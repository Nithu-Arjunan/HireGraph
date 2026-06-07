import ast
import smtplib
from email.message import EmailMessage

from src.state import HireGraphState, SkillWorkerInput
from src.utils import extract_text_from_file
from src.schema import JDRequirements, SkillScore,DimensionScore,EmailCritique
from langchain_openai import ChatOpenAI
from langgraph.types import Send, Command, Literal, interrupt
from langchain_core.messages import AIMessage, ToolMessage

from src.config import (
    CRITIC_MODEL,
    EMAIL_MODEL,
    EXTRACT_MODEL,
    OPENAI_API_KEY,
    SCORING_MODEL,
    SMTP_FROM_EMAIL,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USERNAME,
)
from src.tools import tavily_search_tool
from src.utils import extract_candidate_name, extract_github_url, extract_email_from_text


def build_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=OPENAI_API_KEY,
        max_retries=5,
    )


extract_llm = build_llm(EXTRACT_MODEL)
score_llm = build_llm(SCORING_MODEL)
email_llm = build_llm(EMAIL_MODEL)
critic_llm = build_llm(CRITIC_MODEL)


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

    raw_resume = state.get("raw_resume", "")
    candidate_email = extract_email_from_text(raw_resume)
    candidate_name = extract_candidate_name(raw_resume) if raw_resume else "candidate"

    state["candidate_email"] = candidate_email
    state["candidate_name"] = candidate_name
    state["errors"] = errors
    state.setdefault("email_sent", False)
    state.setdefault("ats_updated", False)
    state.setdefault("rejection_logged", False)
    state.setdefault("compensation_done", False)

    return state


def extract_jd_requirements(state: HireGraphState) -> HireGraphState:
    jd_text = state.get("raw_jd")

    if not jd_text:
        raise ValueError(f"Job description text is missing. Ingestion errors: {state.get('errors', [])}")

    structured_llm = extract_llm.with_structured_output(
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
    
    Important skill extraction rules:
    - Extract broad, scorable skill categories.
    - Do not split sub-skills into separate required skills.
    - Example: "SQL fluency (window functions, CTEs, joins)" should become only "SQL fluency" or "SQL".
    - Example: "Python or R for data wrangling" should become "Python or R" and "data wrangling".
    - Preserve alternatives such as "Python or R" as one skill if the JD accepts either.
    - Do not include concepts like CTEs, window functions, joins, REST verbs, indexes, or query plans as separate skills unless the JD clearly lists them as independent requirements.
    - Tools mentioned only as learning exposure should go to preferred_skills, not required_skills.
    Job Description:
    {jd_text}
    """
    result = structured_llm.invoke(prompt)

    jd_requirements = result.model_dump()

    return {
        "jd_requirements": jd_requirements,
        "messages": [
            AIMessage(
                content=(
                    f"Extracted JD requirements: {jd_requirements}"
                )
            )
        ]
    }

### Node for orchestrator fan out of multiple requirement extraction and scoring nodes in the future ##

def skill_worker(state: SkillWorkerInput):
    resume_text = state["resume_text"]
    skill = state["skill"]

    
    structured_llm = score_llm.with_structured_output(
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


    structured_llm = score_llm.with_structured_output(DimensionScore, method="function_calling")

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


    structured_llm = score_llm.with_structured_output(DimensionScore, method="function_calling")

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

    structured_llm = score_llm.with_structured_output(DimensionScore, method="function_calling")

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

    candidate_name = extract_candidate_name(resume_text)
    github_url = extract_github_url(resume_text)

    if github_url:
        default_query = f"{github_url} GitHub repositories projects"
    else:
        default_query = f"{candidate_name} GitHub repositories projects"

    query = state.get("research_query") or default_query

    research_llm = score_llm.bind_tools(
        [tavily_search_tool],
        tool_choice="tavily_search_tool",
    )

    prompt = f"""
    Search for external GitHub, project, and profile evidence for this candidate.

    Candidate:
    {candidate_name}

    GitHub URL, if available:
    {github_url}

    Use this query:
    {query}
    """

    response = research_llm.invoke(prompt)

    return {
        "resume_text": resume_text,
        "research_query": query,
        "messages": [response],
        "audit_trail": [
            {
                "node": "research_agent",
                "status": "tool_call_requested",
                "message": f"Requested Tavily search tool call for query: {query}",
            }
        ],
    }


def parse_tool_results(content) -> list[dict]:
    if isinstance(content, list):
        return content

    if not isinstance(content, str):
        return []

    try:
        parsed = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return [{"title": "Tool output", "url": "", "content": content}]

    return parsed if isinstance(parsed, list) else []


def normalize_tool_error(content) -> str:
    raw_error = str(content).strip()
    if "ValueError('" in raw_error:
        return raw_error.split("ValueError('", 1)[1].split("')", 1)[0]
    if 'ValueError("' in raw_error:
        return raw_error.split('ValueError("', 1)[1].split('")', 1)[0]
    if raw_error.startswith("Error: "):
        raw_error = raw_error.removeprefix("Error: ").strip()
    return raw_error.splitlines()[0].strip()


def research_scorer(
    state: HireGraphState,
) -> Command[Literal["aggregate_scores", "repair_research_query"]]:
    print("[OK] Running research_scorer")

    tool_messages = [
        message
        for message in state.get("messages", [])
        if isinstance(message, ToolMessage)
    ]

    if not tool_messages:
        tool_error = "Research tool did not return a ToolMessage."
        return Command(
            goto="repair_research_query",
            update={
                "tool_error": tool_error,
                "tool_retry_count": state.get("tool_retry_count", 0) + 1,
                "audit_trail": [
                    {
                        "node": "research_scorer",
                        "status": "tool_error",
                        "message": tool_error,
                    }
                ],
            },
        )

    latest_tool_message = tool_messages[-1]
    tool_status = getattr(latest_tool_message, "status", None)

    if tool_status == "error":
        tool_error = normalize_tool_error(latest_tool_message.content)
        return Command(
            goto="repair_research_query",
            update={
                "tool_error": tool_error,
                "tool_retry_count": state.get("tool_retry_count", 0) + 1,
                "audit_trail": [
                    {
                        "node": "research_scorer",
                        "status": "tool_error",
                        "message": tool_error,
                    }
                ],
                "messages": [
                    AIMessage(
                        content=(
                            "Research tool failed and was routed for LLM query repair. "
                            f"Error: {tool_error}"
                        )
                    )
                ],
            },
        )

    search_results = parse_tool_results(latest_tool_message.content)
    resume_text = state["resume_text"]
    jd_requirements = state["jd_requirements"]
    github_url = extract_github_url(resume_text)
    query = state.get("research_query", "")

    search_context = "\n\n".join(
        [
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
            for item in search_results
        ]
    )

    structured_llm = score_llm.with_structured_output(
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

    print("[OK] research_scorer result:", result.model_dump())

    return Command(
        goto="aggregate_scores",
        update={
            "dimension_evaluations": [result.model_dump()],
            "research_results": [
                {
                    "query": query,
                    "github_url": github_url,
                    "results": search_results,
                }
            ],
            "audit_trail": [
                {
                    "node": "research_scorer",
                    "status": "completed",
                    "message": "Research tool results scored.",
                }
            ],
        },
    )


def repair_research_query(
    state: HireGraphState,
) -> Command[Literal["research_agent", "aggregate_scores"]]:
    retry_count = state.get("tool_retry_count", 0)

    if retry_count >= 3:
        return Command(
            goto="aggregate_scores",
            update={
                "dimension_evaluations": [
                    {
                        "dimension": "research",
                        "score": 0,
                        "evidence": "Research tool failed after query repair attempts.",
                        "reasoning": state.get("tool_error", "Unknown tool error."),
                    }
                ],
                "audit_trail": [
                    {
                        "node": "repair_research_query",
                        "status": "exhausted",
                        "message": "Research query repair attempts exhausted.",
                    }
                ],
            },
        )

    prompt = f"""
    A web search tool failed while researching a candidate profile.

    Failed query:
    {state.get("research_query")}

    Tool error:
    {state.get("tool_error")}

    Rewrite the query so it is simpler, shorter, and likely to work.
    Return only the revised search query text.
    """

    response = score_llm.invoke(prompt)
    repaired_query = response.content.strip()

    return Command(
        goto="research_agent",
        update={
            "research_query": repaired_query,
            "audit_trail": [
                {
                    "node": "repair_research_query",
                    "status": "completed",
                    "message": "Research query repaired by LLM.",
                }
            ],
            "messages": [
                AIMessage(content=f"Repaired research query: {repaired_query}")
            ],
        },
    )


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
        "messages": [
            AIMessage(
                content=(
                    f"Aggregated scores. Final score: {final_score}. "
                )
            )
        ]   
    }

### Node to draft email based on recommendation

def draft_email(state: HireGraphState):
    jd_requirements = state["jd_requirements"]
    score_summary = state.get("score_summary", {})
    skill_evaluations = state.get("skill_evaluations", [])
    dimension_evaluations = state.get("dimension_evaluations", [])
    critic_feedback = state.get("critic_feedback", "")

    candidate_name = state.get("candidate_name", "Candidate")

    prompt = f"""
    You are an HR assistant.

    Write a concise interview invitation email for a candidate who is recommended to advance.

    Rules:
    - Be professional and warm.
    - Address the candidate by name.
    - Do not mention internal scores.
    - Do not reveal private evaluation details.
    - Mention that their background appears aligned with the role.
    - Keep it under 180 words.
    - Do not invent interview date or time.
    - Ask them to share availability.

    Candidate name:
    {candidate_name}

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

    Previous critic feedback, if any:
    {critic_feedback}
"""

    response = email_llm.invoke(prompt)

    return {
        "sender_email": "Hiring_Team",
        "draft_email": response.content,
        "audit_trail": [
            {
                "node": "draft_email",
                "status": "completed",
                "message": "Interview invitation email drafted.",
            }
        ],
    }

### Node to draft rejection email based on recommendation

def draft_rejection(state: HireGraphState):
    jd_requirements = state["jd_requirements"]

    candidate_name = state.get("candidate_name", "Candidate")

    
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

    Candidate name:
    {candidate_name}

    Job title:
    {jd_requirements.get("job_title")}

    Recommendation reasoning:
    {state.get("recommendation_reasoning")}
    """

    response = email_llm.invoke(prompt)

    return {
        "sender_email": "Hiring_Team",
        "rejection_email": response.content,
        "audit_trail": [
            {
                "node": "draft_rejection",
                "status": "completed",
                "message": "Rejection email drafted.",
            }
        ],
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
            "messages": [
                AIMessage(
                    content=(
                        f"Recommendation: {recommendation.upper()}. "
                        f"Reasoning: {reasoning}"
                    )
                )
            ],
        },
    )

# Critic node for evaluating the drafted email and providing feedback for improvement

def critic_loop(
    state: HireGraphState,
) -> Command[Literal["draft_email", "send_email_update_ats", "human_review"]]:

    attempts = state.get("critic_attempts", 0)

    structured_llm = critic_llm.with_structured_output(
        EmailCritique,
        method="function_calling",
    )

    prompt = f"""
    You are reviewing an interview invitation email before it is sent.

    Check whether the email is:
    - Professional
    - Warm
    - Clear
    - Free from internal scores
    - Free from harsh or private evaluation details
    - Under 180 words
    - Asking the candidate to share availability
    - Suitable to send from a hiring team

    Email body:
    {state.get("draft_email")}

    Return approved=true only if it is ready to send.
"""

    result = structured_llm.invoke(prompt)

    if result.approved:
        return Command(
            goto="send_email_update_ats",
            update={
                "email_approved_by_critic": True,
                "critic_feedback": result.feedback,
                "critic_attempts": attempts + 1,
                "audit_trail": [
                    {
                        "node": "critic_loop",
                        "status": "approved",
                        "message": result.feedback,
                    }
                ],
            },
        )

    if attempts >= 2:
        return Command(
            goto="human_review",
            update={
                "email_approved_by_critic": False,
                "critic_feedback": result.feedback,
                "critic_attempts": attempts + 1,
                "audit_trail": [
                    {
                        "node": "critic_loop",
                        "status": "escalated",
                        "message": "Email failed critic review after 3 attempts.",
                    }
                ],
            },
        )

    return Command(
        goto="draft_email",
        update={
            "email_approved_by_critic": False,
            "critic_feedback": result.feedback,
            "critic_attempts": attempts + 1,
            "audit_trail": [
                {
                    "node": "critic_loop",
                    "status": "retry",
                    "message": result.feedback,
                }
            ],
        },
    )

### Node to send email and update ATS

def perform_downstream_actions(state: HireGraphState) -> None:
    """
    Sandbox external action:
    - send email through Mailtrap/Ethereal SMTP
    - update ATS after email send succeeds
    """

    candidate_email = state.get("candidate_email")
    draft_email_body = state.get("draft_email")
    candidate_name = state.get("candidate_name", "Candidate")
    jd_title = state.get("jd_requirements", {}).get("job_title", "the role")
    from_email = SMTP_FROM_EMAIL or SMTP_USERNAME

    if not candidate_email:
        raise ConnectionError("Candidate email is missing.")
    if not draft_email_body:
        raise ConnectionError("Draft email body is missing.")
    if not SMTP_HOST or not SMTP_USERNAME or not SMTP_PASSWORD or not from_email:
        raise ConnectionError("SMTP sandbox configuration is incomplete.")

    message = EmailMessage()
    message["From"] = from_email
    message["To"] = candidate_email
    message["Subject"] = f"Interview invitation for {jd_title}"
    message.set_content(draft_email_body)

    print("[OK] Sending sandbox email to:", candidate_email)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise ConnectionError(f"SMTP send failed: {exc}") from exc

    print("[OK] Sandbox email accepted for:", candidate_name)
    print("[OK] Updating ATS for:", state.get("candidate_name"))


def send_email_update_ats(
    state: HireGraphState,
) -> Command[Literal["finalize", "compensate"]]:
    max_attempts = 3
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            perform_downstream_actions(state)
            return Command(
                goto="finalize",
                update={
                    "email_sent": True,
                    "ats_updated": True,
                    "downstream_attempts": attempt,
                    "audit_trail": [
                        {
                            "node": "send_email_update_ats",
                            "status": "completed",
                            "message": "Email sent and ATS updated.",
                        }
                    ],
                },
            )
        except (ConnectionError, TimeoutError) as exc:
            last_error = str(exc)

    return Command(
        goto="compensate",
        update={
            "email_sent": False,
            "ats_updated": False,
            "downstream_error": last_error,
            "downstream_attempts": max_attempts,
            "audit_trail": [
                {
                    "node": "send_email_update_ats",
                    "status": "failed",
                    "message": (
                        "Downstream email/ATS action failed after retries. "
                        "Routing to compensation."
                    ),
                }
            ],
        },
    )

# Node for logging rejection
def log_rejection(state: HireGraphState):
    print("[OK] Logging rejection for:", state.get("candidate_name"))

    return {
        "rejection_logged": True,
        "audit_trail": [
            {
                "node": "log_rejection",
                "status": "completed",
                "message": "Rejection decision logged.",
            }
        ],
    }

### Node for compensation when the retries are exhausted
def compensate(state: HireGraphState):
    print("[WARN] Running compensation flow")

    return {
        "compensation_done": True,
        "audit_trail": [
            {
                "node": "compensate",
                "status": "completed",
                "message": "Compensation completed. Rollback/alert action recorded.",
            }
        ],
    }

### Final Audit trial node
def finalize(state: HireGraphState):
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Hiring workflow completed. "
                    f"Recommendation: {state.get('recommendation')}. "
                    f"Final score: {state.get('final_score')}."
                )
            )
        ],
        "audit_trail": [
            {
                "node": "finalize",
                "status": "completed",
                "message": "Final audit trail written.",
            }
        ]
    }
