"""LLM Job Extraction Benchmark - Core modules"""

from .schemas import JobEnrichment, ExtractionQuality, ExperimentConfig
from .extraction import (
    extract_with_model,
    extract_with_cloud_model,
    EXTRACTION_PROMPT_DIRECT,
    EXTRACTION_PROMPT_EXAMPLES,
)
from .evaluation import create_judge_prompt, judge_extraction
from .runner import ExperimentRunner, analyze_results

__all__ = [
    # Schemas
    "JobEnrichment",
    "ExtractionQuality",
    "ExperimentConfig",
    # Extraction
    "extract_with_model",
    "extract_with_cloud_model",
    "EXTRACTION_PROMPT_DIRECT",
    "EXTRACTION_PROMPT_EXAMPLES",
    # Evaluation
    "create_judge_prompt",
    "judge_extraction",
    # Runner
    "ExperimentRunner",
    "analyze_results",
]
