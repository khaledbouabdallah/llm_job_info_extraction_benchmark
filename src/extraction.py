"""Job information extraction functions"""

import time
from typing import Dict, Any
from .schemas import JobEnrichment


# ============================================================================
# EXTRACTION PROMPTS - Two Versions
# ============================================================================

# VERSION 1: DIRECT - Just the facts, minimal instructions
EXTRACTION_PROMPT_DIRECT = """Extract job information from this posting.

JOB POSTING:
{description}

EXTRACT:
- required_skills: Technical skills explicitly required
- preferred_skills: Skills marked as "nice to have" or "preferred"
- experience_years_min: Minimum years if stated
- experience_years_max: Maximum years if stated
- seniority_level: Intern/Junior/Mid/Senior
- key_responsibilities: Main job duties (short phrases)
- team_size: Number of people in the team if mentioned
- reports_to: Position this role reports to (e.g., "CTO", "Data Science Manager")
- concrete_benefits: Specific perks (transport %, salary, etc)
- education_required: Degree level and field
- languages: Required languages
- remote_details: Remote work policy if mentioned
- red_flags: Concerning language (unrealistic expectations, vague scope, unpaid work)

RULES:
- Extract ONLY what's explicitly in the text
- Use null for missing information
- Be specific, not generic
- Do NOT invent information"""


# VERSION 2: EXAMPLES - Shows what good extraction looks like
EXTRACTION_PROMPT_EXAMPLES = """Extract job information as structured JSON.

JOB POSTING:
{description}

OUTPUT FORMAT:
{{
    "required_skills": [...],
    "preferred_skills": [...],
    "experience_years_min": int or null,
    "experience_years_max": int or null,
    "seniority_level": "Intern/Junior/Mid/Senior" or null,
    "key_responsibilities": [...],
    "team_size": int or null,
    "reports_to": string or null,
    "concrete_benefits": [...],
    "education_required": string or null,
    "languages": [...],
    "remote_details": string or null,
    "red_flags": [...]
}}

EXAMPLES OF GOOD vs BAD EXTRACTION:

1. SKILLS
---
Source: "Maîtrise de Python et SQL requise. Connaissance de TensorFlow souhaitée."

✓ GOOD:
  required_skills: ["Python", "SQL"]
  preferred_skills: ["TensorFlow"]

✗ BAD:
  required_skills: ["programming languages", "databases"]  ← Too generic
  required_skills: ["Python 3.x", "SQL databases", "data analysis"]  ← Invented details

2. EXPERIENCE
---
Source: "3 à 5 ans d'expérience en data science"

✓ GOOD:
  experience_years_min: 3
  experience_years_max: 5

✗ BAD:
  required_skills: ["3-5 years data science"]  ← Wrong field

3. RESPONSIBILITIES
---
Source: "Vous développerez des modèles ML, analyserez les résultats et présenterez aux stakeholders"

✓ GOOD:
  key_responsibilities: [
    "Développer des modèles ML",
    "Analyser les résultats",
    "Présenter aux stakeholders"
  ]

✗ BAD:
  key_responsibilities: ["Machine learning tasks"]  ← Too vague
  key_responsibilities: [
    "Développer des modèles de machine learning de haute qualité"  ← Added interpretation
  ]

4. BENEFITS
---
Source: "Tickets restaurant, remboursement transport 60%, télétravail 2j/semaine"

✓ GOOD:
  concrete_benefits: [
    "Tickets restaurant",
    "Remboursement transport 60%",
    "Télétravail 2j/semaine"
  ]

✗ BAD:
  concrete_benefits: ["Good benefits package"]  ← Not concrete
  concrete_benefits: ["Meal vouchers worth 10€"]  ← Invented amount

5. SENIORITY
---
Source: "Stage de 6 mois pour étudiant en dernière année"

✓ GOOD:
  seniority_level: "Intern"

✗ BAD:
  seniority_level: "Entry level"  ← Wrong, it's internship
  seniority_level: "Junior"  ← Intern ≠ Junior

KEY PRINCIPLES:
- Specific > Generic ("Python" not "programming")
- Extract > Infer (use exact terms from source)
- Concrete > Vague ("60% transport" not "transport benefits")
- Honest > Complete (null is better than guessing)

Extract from the job posting above following these examples."""


def extract_with_model(
    job_description: str,
    model,
    model_name: str,
    extraction_prompt: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Extract job info with a specific model (Ollama)

    Args:
        job_description: The job posting text
        model: Outlines model instance
        model_name: Name of the model for logging
        extraction_prompt: Prompt template to use
        temperature: Sampling temperature (default 0.0 for deterministic)

    Returns:
        Dict with extraction results, timing, and token counts
    """

    prompt = extraction_prompt.format(description=job_description)

    try:
        start_time = time.time()

        # Estimate tokens (Ollama doesn't provide token counts)
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(enc.encode(prompt))
        except Exception:
            input_tokens = len(prompt) // 4  # Rough estimate

        result = model(prompt, JobEnrichment, options={"temperature": temperature})
        extraction = JobEnrichment.model_validate_json(result)

        # Estimate output tokens
        try:
            output_tokens = len(enc.encode(result))
        except Exception:
            output_tokens = len(result) // 4

        extraction_time = time.time() - start_time

        return {
            "model": model_name,
            "success": True,
            "extraction": extraction.model_dump(),
            "extraction_time": extraction_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": None,
        }
    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "extraction": None,
            "extraction_time": None,
            "input_tokens": None,
            "output_tokens": None,
            "error": str(e),
        }


def extract_with_cloud_model(
    job_description: str,
    model,
    model_name: str,
    extraction_prompt: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Extract job info with cloud model (ChatGPT, Claude, Gemini)

    Args:
        job_description: The job posting text
        model: Model instance (outlines or instructor client)
        model_name: One of "chatgpt", "claude", "gemini"
        extraction_prompt: Prompt template to use
        temperature: Sampling temperature (default 0.0 for deterministic)

    Returns:
        Dict with extraction results, timing, and token counts
    """

    prompt = extraction_prompt.format(description=job_description)

    try:
        start_time = time.time()

        if model_name == "claude":
            # Claude uses instructor with different API
            response = model.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                response_model=JobEnrichment,
            )
            extraction = response
            # Claude returns usage in _raw_response
            input_tokens = response._raw_response.usage.input_tokens
            output_tokens = response._raw_response.usage.output_tokens

        elif model_name == "chatgpt":
            # ChatGPT with outlines
            result = model(prompt, JobEnrichment)
            extraction = JobEnrichment.model_validate_json(result)
            # Estimate tokens for outlines
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(enc.encode(prompt))
            output_tokens = len(enc.encode(result))

        else:  # gemini
            # Gemini with outlines
            result = model(prompt, JobEnrichment)
            extraction = JobEnrichment.model_validate_json(result)
            # Estimate tokens (Gemini uses different tokenizer)
            input_tokens = len(prompt) // 4
            output_tokens = len(result) // 4

        extraction_time = time.time() - start_time

        return {
            "model": model_name,
            "success": True,
            "extraction": (
                extraction.model_dump()
                if hasattr(extraction, "model_dump")
                else extraction
            ),
            "extraction_time": extraction_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": None,
        }
    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "extraction": None,
            "extraction_time": None,
            "input_tokens": None,
            "output_tokens": None,
            "error": str(e),
        }
