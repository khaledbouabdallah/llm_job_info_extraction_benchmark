"""Pydantic schemas for job extraction and evaluation"""

from typing import List, Optional
from pydantic import BaseModel, Field


class JobEnrichment(BaseModel):
    """Schema for extracted job information"""

    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience_years_min: Optional[int] = None
    experience_years_max: Optional[int] = None
    seniority_level: Optional[str] = None
    key_responsibilities: List[str] = Field(default_factory=list)
    team_size: Optional[int] = None
    reports_to: Optional[str] = None
    concrete_benefits: List[str] = Field(default_factory=list)
    education_required: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    remote_details: Optional[str] = None
    red_flags: List[str] = Field(default_factory=list)


class ExtractionQuality(BaseModel):
    """Judge's evaluation of an extraction"""

    completeness_score: int = Field(
        ge=0,
        le=10,
        description="How complete is the extraction? 0=missing most info, 10=extracted everything",
    )

    accuracy_score: int = Field(
        ge=0,
        le=10,
        description="How accurate? 0=many fabrications, 10=all facts correct",
    )

    specificity_score: int = Field(
        ge=0,
        le=10,
        description="How specific? 0=very generic/vague, 10=concrete details",
    )

    hallucination_score: int = Field(
        ge=0,
        le=10,
        description="Hallucinations present? 0=many made up facts, 10=no fabrications",
    )

    # Detailed feedback - ALL REQUIRED for OpenAI compatibility
    missing_critical_info: List[str] = Field(
        description="What important information was missed? Empty list if nothing missed."
    )

    generic_extractions: List[str] = Field(
        description="Which fields are too generic/vague? Empty list if all specific."
    )

    fabricated_info: List[str] = Field(
        description="What information appears fabricated/not in source? Empty list if none."
    )


class ExperimentConfig(BaseModel):
    """Configuration for the experiment"""

    local_models: List[str] = [
        "qwen3:1.7b-q4_K_M",
        "qwen3:4b-q4_K_M",
        "qwen3:4b-instruct-2507-q4_K_M",
        "qwen3:8b-q4_K_M",
        "qwen3:14b-q4_K_M",
    ]
    cloud_models: List[str] = ["gemini"]
    judge_models: List[str] = ["chatgpt", "claude"]
    temperature: float = 0.0
    num_jobs: int = 30
