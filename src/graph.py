from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.node import (
    ingest_resume_jd,
    extract_jd_requirements, 
    skill_worker,
    start_parallel_scoring,
    experience_scorer,
    education_scorer,
    signal_scorer,
    aggregate_scores,
    research_agent,
    draft_email,
    draft_rejection,
    human_review,
    recommendation_router,
)
from src.state import HireGraphState


def build_hiregraph():
    builder = StateGraph(HireGraphState)

    # 1. Add nodes for each step in the hiring evaluation process
    builder.add_node("ingest_resume_jd", ingest_resume_jd)
    builder.add_node("extract_jd_requirements", extract_jd_requirements)

    builder.add_node("skill_worker", skill_worker)
    builder.add_node("start_parallel_scoring", start_parallel_scoring)
    builder.add_node("experience_scorer", experience_scorer)
    builder.add_node("education_scorer", education_scorer)
    builder.add_node("signal_scorer", signal_scorer)
    builder.add_node("research_agent", research_agent)

    builder.add_node("aggregate_scores", aggregate_scores)

    builder.add_node("recommendation_router", recommendation_router)
    builder.add_node("draft_email", draft_email)
    builder.add_node("draft_rejection", draft_rejection)
    builder.add_node("human_review", human_review)



    # 2. Main sequential flow
    builder.add_edge(START, "ingest_resume_jd")
    builder.add_edge("ingest_resume_jd", "extract_jd_requirements")
    # 3. Parallel fan-out
    builder.add_conditional_edges(
        "extract_jd_requirements",
        start_parallel_scoring,
        [
            "skill_worker",
            "experience_scorer",
            "education_scorer",
            "signal_scorer",
            "research_agent",
            
        ],
    )
     # 4. All parallel branches flow into aggregation
    builder.add_edge("skill_worker", "aggregate_scores")
    builder.add_edge("experience_scorer", "aggregate_scores")
    builder.add_edge("education_scorer", "aggregate_scores")
    builder.add_edge("signal_scorer", "aggregate_scores")
    builder.add_edge("research_agent", "aggregate_scores")

    # 5. Recommendation and email drafting flow
    builder.add_edge("aggregate_scores", "recommendation_router")
    builder.add_edge("draft_email", END)
    builder.add_edge("draft_rejection", END)
    
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
    

