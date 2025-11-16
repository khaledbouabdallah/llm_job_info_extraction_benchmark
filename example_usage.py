"""Example usage of the LLM Job Extraction Benchmark"""

import os
from dotenv import load_dotenv
from src import (
    extract_with_model,
    EXTRACTION_PROMPT_DIRECT,
)
import outlines
from ollama import Client

# Load environment variables
load_dotenv()

# Example job posting
job_description = """
Data Scientist - Machine Learning

We're looking for a Data Scientist to join our ML team.

Requirements:
- 3-5 years of experience in data science
- Strong Python and SQL skills
- Experience with TensorFlow or PyTorch
- Master's degree in Computer Science or related field

Responsibilities:
- Develop and deploy machine learning models
- Analyze large datasets for insights
- Collaborate with engineering teams
- Present findings to stakeholders

Benefits:
- Remote work 3 days/week
- 60% public transport reimbursement
- Health insurance
- Annual salary: $90,000 - $120,000

Location: San Francisco, CA (Hybrid)
"""

# Initialize Ollama model
print("Initializing model...")
ollama_client = Client()
model = outlines.from_ollama(ollama_client, "qwen3:4b-q4_K_M")

# Extract information
print("\nExtracting job information...")
result = extract_with_model(
    job_description=job_description,
    model=model,
    model_name="qwen3:4b",
    extraction_prompt=EXTRACTION_PROMPT_DIRECT,
    temperature=0.0,
)

# Display results
if result["success"]:
    print("\n✓ Extraction successful!")
    print(f"Time: {result['extraction_time']:.2f}s")
    print(f"Tokens: {result['input_tokens']} → {result['output_tokens']}")

    print("\n" + "=" * 60)
    print("EXTRACTED INFORMATION:")
    print("=" * 60)

    extraction = result["extraction"]

    print(f"\nRequired Skills: {', '.join(extraction['required_skills'])}")
    print(f"Preferred Skills: {', '.join(extraction['preferred_skills'])}")
    print(
        f"Experience: {extraction['experience_years_min']}-{extraction['experience_years_max']} years"
    )
    print(f"Seniority: {extraction['seniority_level']}")
    print(f"Education: {extraction['education_required']}")
    print(f"Languages: {', '.join(extraction['languages'])}")
    print(f"Remote: {extraction['remote_details']}")

    print("\nKey Responsibilities:")
    for resp in extraction["key_responsibilities"]:
        print(f"  - {resp}")

    print("\nBenefits:")
    for benefit in extraction["concrete_benefits"]:
        print(f"  - {benefit}")

    if extraction["red_flags"]:
        print("\n⚠ Red Flags:")
        for flag in extraction["red_flags"]:
            print(f"  - {flag}")
else:
    print(f"\n✗ Extraction failed: {result['error']}")


# Example: Use with cloud model (Gemini)
print("\n" + "=" * 60)
print("CLOUD MODEL EXAMPLE")
print("=" * 60)

if os.getenv("GIMINI_API_KEY"):
    from google import genai
    from src import extract_with_cloud_model

    print("\nInitializing Gemini...")
    gemini_client = genai.Client(api_key=os.getenv("GIMINI_API_KEY"))
    gemini_model = outlines.from_gemini(gemini_client, "gemini-2.5-flash-lite")

    print("Extracting with Gemini...")
    cloud_result = extract_with_cloud_model(
        job_description=job_description,
        model=gemini_model,
        model_name="gemini",
        extraction_prompt=EXTRACTION_PROMPT_DIRECT,
        temperature=0.0,
    )

    if cloud_result["success"]:
        print(
            f"✓ Cloud extraction successful in {cloud_result['extraction_time']:.2f}s"
        )
        print(
            f"Required Skills: {', '.join(cloud_result['extraction']['required_skills'])}"
        )
    else:
        print(f"✗ Cloud extraction failed: {cloud_result['error']}")
else:
    print("\n⚠ Skipping cloud example (no GIMINI_API_KEY set)")
