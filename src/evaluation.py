"""Judge evaluation functions"""

import json
from typing import Dict, Any
from .schemas import ExtractionQuality


def create_judge_prompt(job_description: str, extraction: dict) -> str:
    """Create judge prompt with explicit rubric and structured output format

    Args:
        job_description: Original job posting text
        extraction: Extracted information as dict

    Returns:
        Formatted judge prompt
    """

    extraction_json = json.dumps(extraction, indent=2, ensure_ascii=False)

    # Get ExtractionQuality schema for reference
    schema_example = {
        "completeness_score": "int (0-10)",
        "accuracy_score": "int (0-10)",
        "specificity_score": "int (0-10)",
        "hallucination_score": "int (0-10)",
        "missing_critical_info": [
            "list of strings describing missing info, or empty list"
        ],
        "generic_extractions": [
            "list of strings describing generic fields, or empty list"
        ],
        "fabricated_info": [
            "list of strings describing fabricated details, or empty list"
        ],
    }
    schema_json = json.dumps(schema_example, indent=2, ensure_ascii=False)

    return f"""You are evaluating an information extraction from a job posting.

Your task: Rate extraction quality using this rubric and return structured JSON.

ORIGINAL JOB POSTING:
{job_description}

EXTRACTED INFORMATION:
{extraction_json}

EXPECTED OUTPUT FORMAT:
{schema_json}

GRADING RUBRIC:

1. COMPLETENESS (0-10):
   10: Extracted ALL job-critical information (skills, experience, responsibilities, salary/benefits)
   7-9: Extracted most critical info, minor omissions only
   4-6: Extracted some critical info, missing important details
   1-3: Extracted very little, major gaps
   0: Extraction is empty or completely wrong

   NOTE: "Critical" = affects job matching (skills, experience, salary, responsibilities)
         "Non-critical" = company culture, awards, office perks (nice to have, not essential)

2. ACCURACY (0-10):
   10: All extracted facts match source text perfectly
   7-9: Minor interpretation differences, no clear errors
   4-6: Some incorrect or mismatched facts
   1-3: Many errors or mismatches
   0: Completely inaccurate

   NOTE: Semantic equivalence is OK ("maîtrise de Python" → "Python" is accurate)

3. SPECIFICITY (0-10):
   10: Extracted concrete details (Python, SQL, 3-5 years, 50% transport)
   7-9: Mostly concrete, some generic terms
   4-6: Mix of concrete and vague
   1-3: Mostly generic/vague
   0: All generic platitudes

   NOTE: "Python" is specific. "Programming" is generic.
         "3 years" is specific. "Some experience" is generic.

4. HALLUCINATION (0-10):
   10: Zero fabricated information
   7-9: Very minor unsupported inferences
   4-6: Some clearly invented details
   1-3: Many fabrications
   0: Mostly made up

   NOTE: Hallucination = inventing facts not in source
         NOT hallucination = reasonable interpretation or paraphrasing

SCORING PHILOSOPHY:
- Be FAIR, not harsh
- Don't penalize for missing non-critical fluff
- Don't penalize for semantic equivalence
- DO penalize for inventing facts or missing key job requirements

IMPORTANT: Return your response as valid JSON matching the expected output format above.
Provide detailed lists for missing_critical_info, generic_extractions, and fabricated_info."""


def judge_extraction(
    job_description: str, extraction: Dict[str, Any], judge_name: str, judges_dict: Dict
) -> ExtractionQuality:
    """Have judge model evaluate an extraction

    Args:
        job_description: Original job posting
        extraction: Extracted information dict
        judge_name: Name of judge model ("chatgpt", "claude", etc)
        judges_dict: Dictionary mapping judge names to model instances

    Returns:
        ExtractionQuality evaluation
    """

    prompt = create_judge_prompt(job_description, extraction)
    judge = judges_dict[judge_name]

    if judge_name == "claude":
        # Claude uses instructor with different API
        result = judge.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            response_model=ExtractionQuality,
        )
        return result
    else:
        # ChatGPT and Gemini use outlines
        result_json = judge(prompt, ExtractionQuality)
        return ExtractionQuality.model_validate_json(result_json)
