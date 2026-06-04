from src.state import HireGraphState, SkillWorkerInput
from src.utils import extract_text_from_file
from src.schema import JDRequirements, SkillScore,DimensionScore
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.types import Send, Command, Literal, interrupt

from src.tools import tavily_search
from src.utils import extract_candidate_name, extract_github_url

load_dotenv()
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=5,
)


###########  Node for file Ingestion ############

def ingest_resume_jd(state: HireGraphState) -> HireGraphState:
    errors = state.get("errors", [])

    try:
        if "resume_path" in state:
            state["raw_resume"] = extract_text_from_file(state["resume_path"])
        else:
            errors.append("Resume path is missing.")

        if "jd_path" in state:
            state["raw_jd"] = extract_text_from_file(state["jd_path"])
        else:
            errors.append("Job description path is missing.")
    except Exception as e:
        errors.append(str(e))

    state["errors"] = errors
    return state


def extract_jd_requirements(state: HireGraphState) -> HireGraphState:
    jd_text = state.get("raw_jd")

    if not jd_text:
        raise ValueError(f"Job description text is missing. Ingestion errors: {state.get('errors', [])}")

    structured_llm = llm.with_structured_output(
        JDRequirements,
        method="function_calling",
    )

    prompt = f"""
   You are an expert technical recruiter and hiring analyst.

    Extract structured hiring requirements from the following job description.

    Rules:
    - Extract only what is supported by the JD.
    - Do not invent skills that are not mentioned.
    - Separate required skills from preferred or nice-to-have skills.
    - If seniority is unclear, return "unknown".
    - Keep each item concise.
    - Required skills should include tools, technologies, frameworks, programming languages, and concrete capabilities.
    - Responsibilities should describe what the person will actually do in the role.
    - Do not summarize detailed technical requirements too broadly.
    - If the JD lists sub-skills, extract them separately.
    - Example: "SQL fluency (window functions, CTEs, joins)" should become:
    ["SQL", "window functions", "CTEs", "joins"].
    - Preserve alternatives such as "Python or R" as one requirement if the JD accepts either.
    - Tools mentioned in responsibilities should be captured either as required or preferred depending on wording.
    - Do not invent requirements.

    Job Description:
    {jd_text}
    """
    result = structured_llm.invoke(prompt)

    return {
        "jd_requirements": result.model_dump(),
        }

### Node for orchestrator fan out of multiple requirement extraction and scoring nodes in the future ##

def skill_worker(state: SkillWorkerInput):
    resume_text = state["resume_text"]
    skill = state["skill"]

    
    structured_llm = llm.with_structured_output(
        SkillScore,
        method="function_calling",
    )

    prompt = f"""
        You are evaluating a candidate resume against one required job skill.

        Evaluate only this skill:
        {skill}

        Scoring rules:
        - Score from 0 to 10.
        - 0 means no evidence.
        - 5 means partial or indirect evidence.
        - 10 means strong direct evidence.
        - Use only the resume text.
        - Do not assume experience that is not present.
        - If the skill contains an alternative such as "Python or R", give credit if either skill is present.
        - Keep evidence concise.

        Resume:
        {resume_text}
        """

    result = structured_llm.invoke(prompt)

    return {
        "skill_evaluations": [result.model_dump()]
    }
  

#### Function for parallel workers for evaluating education,experience and signal

def education_scorer(state: HireGraphState):
    print("[OK] Running education_scorer")
    jd_requirements = state["jd_requirements"]
    resume_text = state["resume_text"]


    structured_llm = llm.with_structured_output(DimensionScore,method="function_calling")

    prompt = f"""
        You are evaluating the candidate's education fit for a job.

        Evaluate only this dimension:
        education

        Scoring rules:
        - Score from 0 to 10.
        - 0 means no relevant education or training.
        - 5 means partially relevant education or alternative training.
        - 10 means strong direct education match.
        - If the JD says the degree is optional, do not penalize heavily.
        - Consider bootcamps, self-taught background, certifications, or equivalent training if mentioned.
        - Use only evidence from the resume.

        JD education requirements:
        {jd_requirements.get("education_requirements", [])}

        Resume:
        {resume_text}
        """
    result = structured_llm.invoke(prompt)
    print("[OK] education_scorer result:", result.model_dump())
    return {
        "dimension_evaluations": [result.model_dump()]
    }

def experience_scorer(state: HireGraphState):
    jd_requirements = state["jd_requirements"]
    resume_text = state["resume_text"]


    structured_llm = llm.with_structured_output(DimensionScore, method="function_calling")

    prompt = f"""
        You are evaluating the candidate's experience fit for a job.

        Evaluate only this dimension:
        experience

        Scoring rules:
        - Score from 0 to 10.
        - 0 means no relevant experience.
        - 5 means partially relevant experience.
        - 10 means strong relevant experience.
        - Compare the resume against the JD's experience requirements.
        - Use only evidence from the resume.
        - Do not assume anything not present.

        JD experience requirements:
        {jd_requirements.get("experience_requirements", [])}

        JD responsibilities:
        {jd_requirements.get("responsibilities", [])}

        Resume:
        {resume_text}

        """
    result = structured_llm.invoke(prompt)

    return {
        "dimension_evaluations": [result.model_dump()]
    }

def signal_scorer(state: HireGraphState):
    resume_text = state["resume_text"]
    jd_requirements = state["jd_requirements"]

    structured_llm = llm.with_structured_output(DimensionScore,method="function_calling")

    prompt = f"""
        You are evaluating the candidate's additional positive signals.

        Evaluate only this dimension:
        signal

        Signals can include:
        - Strong side projects
        - GitHub/open-source activity
        - Communication or presentation evidence
        - Leadership or initiative
        - Domain interest
        - Evidence of learning ability
        - Evidence of ownership

        Scoring rules:
        - Score from 0 to 10.
        - 0 means no positive signals.
        - 5 means some useful signals.
        - 10 means very strong positive signals.
        - Use only evidence from the resume.
        - Compare signals against the JD context.

        JD domain keywords:
        {jd_requirements.get("domain_keywords", [])}

        JD responsibilities:
        {jd_requirements.get("responsibilities", [])}

        Resume:
        {resume_text}
"""

    result = structured_llm.invoke(prompt)

    return {
        "dimension_evaluations": [result.model_dump()]
    }



def research_agent(state):
    print("[OK] Running research_agent")

    resume_text = state["resume_text"]
    jd_requirements = state["jd_requirements"]

    candidate_name = extract_candidate_name(resume_text)
    github_url = extract_github_url(resume_text)

    if github_url:
        query = f"{github_url} GitHub repositories projects"
    else:
        query = f"{candidate_name} GitHub repositories projects"

    search_results = tavily_search(query, max_results=5)

    search_context = "\n\n".join(
        [
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
            for item in search_results
        ]
    )

    structured_llm = llm.with_structured_output(
        DimensionScore,
        method="function_calling",
    )

    prompt = f"""
    You are a research agent evaluating external project and profile evidence for a candidate.

    The dimension field must be exactly:
    research

    Your task:
    - Review the resume.
    - Review the Tavily search results.
    - Evaluate GitHub/project/profile evidence.
    - Score how useful the candidate's external project evidence is for this JD.

    Scoring rules:
    - Score from 0 to 10.
    - 0 means no useful external/project evidence.
    - 5 means some useful but limited evidence.
    - 10 means strong project evidence aligned with the JD.
    - Give credit for relevant repositories, project ownership, stars, technical depth, project usefulness, and alignment with the JD.
    - Do not invent evidence.
    - If Tavily results are weak or empty, rely only on project evidence in the resume.

    JD required skills:
    {jd_requirements.get("required_skills", [])}

    JD preferred skills:
    {jd_requirements.get("preferred_skills", [])}

    JD responsibilities:
    {jd_requirements.get("responsibilities", [])}

    Resume:
    {resume_text}

    Tavily search results:
    {search_context}
"""

    result = structured_llm.invoke(prompt)

    print("[OK] research_agent result:", result.model_dump())

    return {
        "dimension_evaluations": [result.model_dump()],
        "research_results": [
            {
                "query": query,
                "github_url": github_url,
                "results": search_results,
            }
        ],
    }


### Function to call both fixed and dynamic skill workers

def start_parallel_scoring(state: HireGraphState):
    required_skills = state["jd_requirements"]["required_skills"]

    skill_worker_sends = [
        Send(
            "skill_worker",
            {
                "resume_text": state["raw_resume"],
                "skill": skill,
            },
        )
        for skill in required_skills
    ]

    fixed_scorer_sends = [
    Send(
        "experience_scorer",
        {
            "resume_text": state["raw_resume"],
            "jd_requirements": state["jd_requirements"],
        },
    ),
    Send(
        "education_scorer",
        {
            "resume_text": state["raw_resume"],
            "jd_requirements": state["jd_requirements"],
        },
    ),
    Send(
        "signal_scorer",
        {
            "resume_text": state["raw_resume"],
            "jd_requirements": state["jd_requirements"],
        },
    ),
    Send(
        "research_agent",
        {
            "resume_text": state["raw_resume"],
            "jd_requirements": state["jd_requirements"],
        },
    ),
    ]
    
    return [
        *skill_worker_sends,
        *fixed_scorer_sends,
    ]

### Function to find the aggregrate scores 

def aggregate_scores(state: HireGraphState):
    skill_evaluations = state.get("skill_evaluations", [])
    dimension_evaluations = state.get("dimension_evaluations", [])

    skill_scores = [item["score"] for item in skill_evaluations]
    dimension_scores = [item["score"] for item in dimension_evaluations]

    all_scores = skill_scores + dimension_scores

    if not all_scores:
        final_score = 0
    else:
        final_score = round(sum(all_scores) / len(all_scores), 2)

    return {
        "final_score": final_score,
        "score_summary": {
            "skill_count": len(skill_evaluations),
            "dimension_count": len(dimension_evaluations),
            "skill_average": round(sum(skill_scores) / len(skill_scores), 2) if skill_scores else 0,
            "dimension_average": round(sum(dimension_scores) / len(dimension_scores), 2) if dimension_scores else 0,
        },
    }

### Node to draft email based on recommendation

def draft_email(state: HireGraphState):
    jd_requirements = state["jd_requirements"]
    score_summary = state.get("score_summary", {})
    skill_evaluations = state.get("skill_evaluations", [])
    dimension_evaluations = state.get("dimension_evaluations", [])

    prompt = f"""
    You are an HR assistant.

    Write a concise interview invitation email for a candidate who is recommended to advance.

    Rules:
    - Be professional and warm.
    - Do not mention internal scores.
    - Mention that their background appears aligned with the role.
    - Keep it under 180 words.
    - Do not invent interview date or time.
    - Ask them to share availability.

    Job title:
    {jd_requirements.get("job_title")}

    Recommendation reasoning:
    {state.get("recommendation_reasoning")}

    Skill evaluations:
    {skill_evaluations}

    Dimension evaluations:
    {dimension_evaluations}

    Score summary:
    {score_summary}
"""

    response = llm.invoke(prompt)

    return {
        "draft_email": response.content
    }

### Node to draft rejection email based on recommendation

def draft_rejection(state: HireGraphState):
    jd_requirements = state["jd_requirements"]

    
    prompt = f"""
    You are an HR assistant.

    Write a concise rejection email for a candidate who is not moving forward.

    Rules:
    - Be respectful and kind.
    - Do not mention scores.
    - Do not criticize harshly.
    - Do not provide overly specific negative feedback.
    - Keep it under 150 words.
    - Encourage them to apply again in the future.

    Job title:
    {jd_requirements.get("job_title")}

    Recommendation reasoning:
    {state.get("recommendation_reasoning")}
    """

    response = llm.invoke(prompt)

    return {
        "rejection_email": response.content
    }

### Human review node for borderline cases

def human_review(
    state: HireGraphState,
) -> Command[Literal["draft_email", "draft_rejection"]]:

    user_input = interrupt(
        {
            "message": "Borderline candidate requires human review",
            "request": "Please approve or reject this candidate.",
            "candidate_recommendation": state.get("recommendation"),
            "recommendation_reasoning": state.get("recommendation_reasoning"),
            "final_score": state.get("final_score"),
            "score_summary": state.get("score_summary"),
            "skill_evaluations": state.get("skill_evaluations", []),
            "dimension_evaluations": state.get("dimension_evaluations", []),
            "expected_response_format": {
                "decision": "approved or rejected",
                "notes": "optional reviewer notes",
            },
        }
    )

    decision = user_input.get("decision", "").lower().strip()
    notes = user_input.get("notes", "")

    if decision == "approved":
        return Command(
            goto="draft_email",
            update={
                "human_review_decision": "approved",
                "human_review_notes": notes,
                "recommendation_reasoning": (
                    state.get("recommendation_reasoning", "")
                    + " Human reviewer approved the candidate."
                ),
            },
        )

    return Command(
        goto="draft_rejection",
        update={
            "human_review_decision": "rejected",
            "human_review_notes": notes,
            "recommendation_reasoning": (
                state.get("recommendation_reasoning", "")
                + " Human reviewer rejected the candidate."
            ),
        },
    )


### Recommendation router ####

def recommendation_router(
    state: HireGraphState,) -> Command[Literal["draft_email", "draft_rejection", "human_review"]]:

    final_score = state.get("final_score", 0)
    score_summary = state.get("score_summary", {})

    skill_average = score_summary.get("skill_average", 0)
    dimension_average = score_summary.get("dimension_average", 0)

    
    if final_score >= 7:
        recommendation = "advance"
        reasoning = (
            f"Candidate is recommended to advance. Final score is {final_score}, "
            f"with skill average {skill_average} and dimension average {dimension_average}."
        )
        goto = "draft_email"

    elif final_score >= 5:
        recommendation = "borderline"
        reasoning = (
            f"Candidate is borderline. Final score is {final_score}, "
            f"with skill average {skill_average} and dimension average {dimension_average}. "
            "Human review is recommended."
        )
        goto = "human_review"

    else:
        recommendation = "reject"
        reasoning = (
            f"Candidate is recommended for rejection. Final score is {final_score}, "
            f"with skill average {skill_average} and dimension average {dimension_average}."
        )
        goto = "draft_rejection"

    return Command(
        goto=goto,
        update={
            "recommendation": recommendation,
            "recommendation_reasoning": reasoning,
        },
    )
