from typing import Annotated, TypedDict,Literal,Any  
import operator  


class HireGraphState(TypedDict, total=False):
    resume_path: str
    jd_path: str
    raw_resume: str
    raw_jd: str

    errors: list[str]

    jd_requirements: dict[str, Any]

    # Orchestration fields
    skill_evaluations:Annotated[list[dict[str,Any]],operator.add]

    # Parallelization fields
    dimension_evaluations: Annotated[list[dict[str,Any]],operator.add]

    # Researcher results
    research_results: Annotated[list[dict[str, Any]], operator.add]

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
    audit_trail: Annotated[list[dict[str, Any]], operator.add]
    
class SkillWorkerInput(TypedDict):
    resume_text: str
    skill: str
