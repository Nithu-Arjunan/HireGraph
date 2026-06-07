from typing import Annotated, TypedDict,Literal,Any  
import operator  
#from langgraph.types import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class HireGraphState(TypedDict, total=False):
    resume_path: str
    jd_path: str
    raw_resume: str
    raw_jd: str
    resume_text: str

    errors: list[str]

    jd_requirements: dict[str, Any]

    # Orchestration fields
    skill_evaluations:Annotated[list[dict[str,Any]],operator.add]

    # Parallelization fields
    dimension_evaluations: Annotated[list[dict[str,Any]],operator.add]

    # Researcher results
    research_results: Annotated[list[dict[str, Any]], operator.add]
    research_query: str
    tool_error: str
    tool_retry_count: int

    #Final score
    final_score: float
    score_summary: dict[str, Any]

    # Recommendation fields
    recommendation: Literal["advance", "borderline", "reject"]
    recommendation_reasoning: str

    #Email generation fields
    candidate_email: str | None
    candidate_name: str
    sender_email: str
    draft_email: str
    rejection_email: str

    # Human review fields
    human_review_decision: Literal["approved", "rejected"]
    human_review_notes: str

    # Critic fields for email drafting
    critic_attempts: int
    critic_feedback: str
    email_approved_by_critic: bool

    # Action and audit fields
    email_sent: bool
    ats_updated: bool
    rejection_logged: bool
    compensation_done: bool
    
    ## Fields for implementing compensation routing after retry exhaustion
    downstream_error: str
    downstream_attempts: int

    audit_trail: Annotated[list[dict[str, Any]], operator.add]

    ### Messaging field to keep track of messages sent to and from various nodes
    messages: Annotated[list[AnyMessage], add_messages]
    
class SkillWorkerInput(TypedDict):
    resume_text: str
    skill: str
