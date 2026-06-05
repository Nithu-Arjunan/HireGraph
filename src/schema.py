
from typing import Literal
from pydantic import BaseModel, Field
import re

#### Schema for JD requirements extraction ############

class JDRequirements(BaseModel):
    job_title: str = Field(
        description="The main job title from the job description."
    )

    seniority: Literal["junior", "mid", "senior", "executive", "unknown"] = Field(
        description="Estimated seniority level required for the role."
    )

    required_skills: list[str] = Field(
        description=(
        "Core required skills from the JD. Extract broad, scorable skills only. "
        "Do not split subtopics into separate skills. For example, if the JD says "
        "'SQL fluency with window functions, CTEs, and joins', return only 'SQL fluency' "
        "or 'SQL', not separate items for window functions, CTEs, and joins. "
        "Keep alternatives like 'Python or R' as one skill."
        )
    )

    preferred_skills: list[str] = Field(
        default_factory=list,
        description=(
        "Nice-to-have skills, tools, platforms, domains, or technologies. "
        "Include tools mentioned under 'Nice to have' and tools mentioned as "
        "learning exposure, unless they are clearly required."
        )
    )

    experience_requirements: list[str] = Field(
        default_factory=list,
        description="Experience-related requirements such as years of experience, project type, or domain background."
    )

    education_requirements: list[str] = Field(
        default_factory=list,
        description="Education, degree, certification, or academic requirements."
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="Main responsibilities expected from the candidate."
    )

    domain_keywords: list[str] = Field(
        default_factory=list,
        description="Industry or domain-specific keywords such as finance, healthcare, AI agents, RAG, data pipelines, newsletters, etc."
    )


#### Schema for skill score rating ############

class SkillScore(BaseModel):
    skill: str = Field(description="The skill being evaluated.")
    score: int = Field(
        description="A score from 0 to 10 indicating how well the resume matches the skill requirement."
        )
    evidence: str =Field(
        description="Resume evidence supporting the score"
        )
    reasoning: str = Field(
        description="A brief explanation of why the score was given, based on the resume evidence."
        ) 


#### Schema for parallel skill worker output ############

class DimensionScore(BaseModel):
    dimension: Literal["experience", "education", "signal","research"] = Field(
        description="The broad candidate evaluation dimension."
    )

    score: int = Field(
        description="Score from 0 to 10 for this dimension."
    )

    evidence: str = Field(
        description="Resume evidence supporting this score."
    )

    reasoning: str = Field(
        description="Brief explanation for the score."
    )


### Recommendation schema ############

class RecommendationDecision(BaseModel):
    recommendation: Literal["advance", "borderline", "reject"] = Field(
        description="Final hiring recommendation."
    )
    recommendation_reasoning: str = Field(
        description="Brief explanation for the recommendation based on score and evidence."
    )

### Critic schema for email drafting ############

class EmailCritique(BaseModel):
    approved: bool = Field(
        description="Whether the email is good enough to send."
    )
    feedback: str = Field(
        description="Short feedback explaining what should be improved."
    )