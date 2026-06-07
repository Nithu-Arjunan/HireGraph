from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from langgraph.prebuilt import ToolNode

from src.node import (
    ingest_resume_jd,
    parse_resume,
    normalize_resume_skills,
    extract_years_experience,
    extract_jd_requirements, 
    seniority_router,
    junior_scoring_profile,
    mid_scoring_profile,
    senior_scoring_profile,
    executive_scoring_profile,
    start_parallel_scoring_gate,
    skill_worker,
    start_parallel_scoring,
    experience_scorer,
    education_scorer,
    signal_scorer,
    aggregate_scores,
    research_agent,
    research_scorer,
    repair_research_query,
    draft_email,
    draft_rejection,
    human_review,
    recommendation_router,
    critic_loop,
    send_email_update_ats,
    log_rejection,
    compensate,
    finalize,
)
from src.state import HireGraphState
from src.tools import tavily_search_tool


def build_hiregraph():
    builder = StateGraph(HireGraphState)

    # 1. Add nodes for each step in the hiring evaluation process
    builder.add_node("ingest_resume_jd", ingest_resume_jd)
    builder.add_node("parse_resume", parse_resume, retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.05, retry_on=ConnectionError)
    )
    builder.add_node("normalize_resume_skills", normalize_resume_skills, retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.05, retry_on=ConnectionError)
    )
    builder.add_node("extract_years_experience", extract_years_experience, retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.05, retry_on=ConnectionError)
    )
    builder.add_node("extract_jd_requirements", extract_jd_requirements, retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.05, retry_on=ConnectionError)
    )
    builder.add_node("seniority_router", seniority_router)
    builder.add_node("junior_scoring_profile", junior_scoring_profile)
    builder.add_node("mid_scoring_profile", mid_scoring_profile)
    builder.add_node("senior_scoring_profile", senior_scoring_profile)
    builder.add_node("executive_scoring_profile", executive_scoring_profile)

    builder.add_node("skill_worker", skill_worker)
    builder.add_node("start_parallel_scoring", start_parallel_scoring_gate)
    builder.add_node("experience_scorer", experience_scorer)
    builder.add_node("education_scorer", education_scorer)
    builder.add_node("signal_scorer", signal_scorer)
    builder.add_node("research_agent", research_agent, retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.05, retry_on=ConnectionError),
    )
    builder.add_node("research_tools", ToolNode([tavily_search_tool], handle_tool_errors=True))
    builder.add_node("research_scorer", research_scorer)
    builder.add_node("repair_research_query", repair_research_query)

    builder.add_node("aggregate_scores", aggregate_scores)

    builder.add_node("recommendation_router", recommendation_router)
    builder.add_node("draft_email", draft_email, retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.05, retry_on=ConnectionError),
    )
    builder.add_node("draft_rejection", draft_rejection, retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.05, retry_on=ConnectionError),
    )
    builder.add_node("human_review", human_review)
    builder.add_node("critic_loop", critic_loop)

    builder.add_node("send_email_update_ats", send_email_update_ats)

 
    builder.add_node("log_rejection", log_rejection)
    builder.add_node("compensate", compensate)
    builder.add_node("finalize", finalize)


    # 2. Main sequential flow
    builder.add_edge(START, "ingest_resume_jd")
    builder.add_edge("ingest_resume_jd", "parse_resume")
    builder.add_edge("parse_resume", "normalize_resume_skills")
    builder.add_edge("normalize_resume_skills", "extract_years_experience")
    builder.add_edge("extract_years_experience", "extract_jd_requirements")
    builder.add_edge("extract_jd_requirements", "seniority_router")
    # 3. Parallel fan-out
    builder.add_conditional_edges(
        "start_parallel_scoring",
        start_parallel_scoring,
        [
            "skill_worker",
            "experience_scorer",
            "education_scorer",
            "signal_scorer",
            "research_agent",
            
        ],
    )
     # 4. All parallel branches join before aggregation
    builder.add_edge("research_agent", "research_tools")
    builder.add_edge("research_tools", "research_scorer")
    builder.add_edge(
        [
            "skill_worker",
            "experience_scorer",
            "education_scorer",
            "signal_scorer",
            "research_scorer",
        ],
        "aggregate_scores",
    )

    # 5. Recommendation and email drafting flow
    builder.add_edge("aggregate_scores", "recommendation_router")
    builder.add_edge("draft_email", "critic_loop")
    builder.add_edge("draft_rejection", "log_rejection")
    builder.add_edge("log_rejection", "finalize")

    builder.add_edge("compensate", "finalize")

    builder.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
    
