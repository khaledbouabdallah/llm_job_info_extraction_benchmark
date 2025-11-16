"""Experiment runner and analysis"""

import pandas as pd
from datetime import datetime
from typing import List, Dict
from .schemas import ExperimentConfig
from .extraction import extract_with_model, extract_with_cloud_model
from .evaluation import judge_extraction


class ExperimentRunner:
    """Runs full extraction experiment with multiple models, prompts, and judges"""
    
    def __init__(
        self,
        config: ExperimentConfig,
        jobs: List[Dict],
        judges_dict: Dict,
        cloud_extractors: Dict,
        prompt_direct: str,
        prompt_examples: str
    ):
        """Initialize experiment runner
        
        Args:
            config: Experiment configuration
            jobs: List of job dicts with 'title', 'company', 'description', etc
            judges_dict: Dict mapping judge names to model instances
            cloud_extractors: Dict mapping cloud model names to instances
            prompt_direct: Direct prompt template
            prompt_examples: Few-shot examples prompt template
        """
        self.config = config
        self.jobs = jobs[:config.num_jobs]
        self.results = []
        self.judges_dict = judges_dict
        self.cloud_extractors = cloud_extractors
        self.prompt_direct = prompt_direct
        self.prompt_examples = prompt_examples

    def run_experiment(self):
        """Run full experiment with TWO prompts per model per job"""

        # Test both prompt versions
        prompts_to_test = [
            ("direct", self.prompt_direct),
            ("examples", self.prompt_examples)
        ]

        total_combinations = (len(self.config.local_models) + len(self.config.cloud_models)) * len(prompts_to_test)
        current_combination = 0

        # Extract with local models - TEST BOTH PROMPTS
        for model_name in self.config.local_models:
            for prompt_version, extraction_prompt in prompts_to_test:
                current_combination += 1
                print(f"\n{'='*70}")
                print(f"COMBINATION {current_combination}/{total_combinations}: {model_name} + {prompt_version}")
                print(f"{'='*70}")

                try:
                    import outlines
                    from ollama import Client as OllamaClient
                    
                    ollama_client = outlines.from_ollama(
                        OllamaClient(),
                        model_name
                    )

                    # Process all jobs with this model + prompt combination
                    for job_idx, job in enumerate(self.jobs):
                        print(f"\n  [{job_idx + 1}/{len(self.jobs)}] {job['title'][:60]}...")
                        job_description = job['description']

                        local_result = extract_with_model(
                            job_description,
                            ollama_client,
                            model_name,
                            extraction_prompt,
                            temperature=self.config.temperature
                        )

                        if local_result["success"]:
                            print(f"    ✓ Extracted in {local_result['extraction_time']:.2f}s ({local_result['input_tokens']}→{local_result['output_tokens']} tokens)")
                            
                            # Judge the extraction with all judges
                            for judge_name in self.config.judge_models:
                                try:
                                    quality = judge_extraction(
                                        job_description,
                                        local_result["extraction"],
                                        judge_name,
                                        self.judges_dict
                                    )

                                    # Store result with ALL metadata
                                    result_record = {
                                        "job_id": job_idx,
                                        "job_title": job["title"],
                                        "job_company": job["company"],
                                        "job_description": job_description,
                                        "description_length": job["description_length"],
                                        "description_language": job["description_language"],
                                        "extraction_model": model_name,
                                        "prompt_version": prompt_version,
                                        "extraction_time": local_result["extraction_time"],
                                        "input_tokens": local_result["input_tokens"],
                                        "output_tokens": local_result["output_tokens"],
                                        "judge_model": judge_name,
                                        "extraction_success": True,
                                        "extraction": local_result["extraction"],
                                        **quality.model_dump(),
                                        "timestamp": datetime.now()
                                    }
                                    self.results.append(result_record)
                                except Exception as e:
                                    print(f"    ✗ Judge {judge_name} error: {str(e)[:80]}")
                                    self.results.append({
                                        "job_id": job_idx,
                                        "job_title": job["title"],
                                        "job_company": job["company"],
                                        "job_description": job_description,
                                        "description_length": job["description_length"],
                                        "description_language": job["description_language"],
                                        "extraction_model": model_name,
                                        "prompt_version": prompt_version,
                                        "extraction_time": local_result["extraction_time"],
                                        "input_tokens": local_result["input_tokens"],
                                        "output_tokens": local_result["output_tokens"],
                                        "judge_model": judge_name,
                                        "extraction_success": False,
                                        "judge_error": str(e),
                                        "timestamp": datetime.now()
                                    })
                        else:
                            print(f"    ✗ Extraction failed: {local_result['error'][:80]}")
                            # Extraction failed - record for all judges
                            for judge_name in self.config.judge_models:
                                self.results.append({
                                    "job_id": job_idx,
                                    "job_title": job["title"],
                                    "job_company": job["company"],
                                    "job_description": job_description,
                                    "description_length": job["description_length"],
                                    "description_language": job["description_language"],
                                    "extraction_model": model_name,
                                    "prompt_version": prompt_version,
                                    "extraction_time": None,
                                    "input_tokens": None,
                                    "output_tokens": None,
                                    "judge_model": judge_name,
                                    "extraction_success": False,
                                    "error": local_result["error"],
                                    "timestamp": datetime.now()
                                })

                except Exception as e:
                    print(f"\n  ✗ Model initialization failed: {e}")
                    # Record failures for all jobs with this model+prompt combo
                    for job_idx, job in enumerate(self.jobs):
                        job_description = job['description']
                        for judge_name in self.config.judge_models:
                            self.results.append({
                                "job_id": job_idx,
                                "job_title": job["title"],
                                "job_company": job["company"],
                                "job_description": job_description,
                                "description_length": job["description_length"],
                                "description_language": job["description_language"],
                                "extraction_model": model_name,
                                "prompt_version": prompt_version,
                                "extraction_time": None,
                                "input_tokens": None,
                                "output_tokens": None,
                                "judge_model": judge_name,
                                "extraction_success": False,
                                "error": str(e),
                                "timestamp": datetime.now()
                            })

        # Extract with cloud models - TEST BOTH PROMPTS
        for model_name in self.config.cloud_models:
            for prompt_version, extraction_prompt in prompts_to_test:
                current_combination += 1
                print(f"\n{'='*70}")
                print(f"COMBINATION {current_combination}/{total_combinations}: {model_name} + {prompt_version}")
                print(f"{'='*70}")

                try:
                    cloud_model = self.cloud_extractors[model_name]

                    # Process all jobs with this model + prompt combination
                    for job_idx, job in enumerate(self.jobs):
                        print(f"\n  [{job_idx + 1}/{len(self.jobs)}] {job['title'][:60]}...")
                        job_description = job['description']

                        cloud_result = extract_with_cloud_model(
                            job_description,
                            cloud_model,
                            model_name,
                            extraction_prompt,
                            temperature=self.config.temperature
                        )

                        if cloud_result["success"]:
                            print(f"    ✓ Extracted in {cloud_result['extraction_time']:.2f}s ({cloud_result['input_tokens']}→{cloud_result['output_tokens']} tokens)")
                            
                            # Judge the extraction with all judges
                            for judge_name in self.config.judge_models:
                                try:
                                    quality = judge_extraction(
                                        job_description,
                                        cloud_result["extraction"],
                                        judge_name,
                                        self.judges_dict
                                    )

                                    # Store result with ALL metadata
                                    result_record = {
                                        "job_id": job_idx,
                                        "job_title": job["title"],
                                        "job_company": job["company"],
                                        "job_description": job_description,
                                        "description_length": job["description_length"],
                                        "description_language": job["description_language"],
                                        "extraction_model": model_name,
                                        "prompt_version": prompt_version,
                                        "extraction_time": cloud_result["extraction_time"],
                                        "input_tokens": cloud_result["input_tokens"],
                                        "output_tokens": cloud_result["output_tokens"],
                                        "judge_model": judge_name,
                                        "extraction_success": True,
                                        "extraction": cloud_result["extraction"],
                                        **quality.model_dump(),
                                        "timestamp": datetime.now()
                                    }
                                    self.results.append(result_record)
                                except Exception as e:
                                    print(f"    ✗ Judge {judge_name} error: {str(e)[:80]}")
                                    self.results.append({
                                        "job_id": job_idx,
                                        "job_title": job["title"],
                                        "job_company": job["company"],
                                        "job_description": job_description,
                                        "description_length": job["description_length"],
                                        "description_language": job["description_language"],
                                        "extraction_model": model_name,
                                        "prompt_version": prompt_version,
                                        "extraction_time": cloud_result["extraction_time"],
                                        "input_tokens": cloud_result["input_tokens"],
                                        "output_tokens": cloud_result["output_tokens"],
                                        "judge_model": judge_name,
                                        "extraction_success": False,
                                        "judge_error": str(e),
                                        "timestamp": datetime.now()
                                    })
                        else:
                            print(f"    ✗ Extraction failed: {cloud_result['error'][:80]}")
                            # Extraction failed - record for all judges
                            for judge_name in self.config.judge_models:
                                self.results.append({
                                    "job_id": job_idx,
                                    "job_title": job["title"],
                                    "job_company": job["company"],
                                    "job_description": job_description,
                                    "description_length": job["description_length"],
                                    "description_language": job["description_language"],
                                    "extraction_model": model_name,
                                    "prompt_version": prompt_version,
                                    "extraction_time": None,
                                    "input_tokens": None,
                                    "output_tokens": None,
                                    "judge_model": judge_name,
                                    "extraction_success": False,
                                    "error": cloud_result["error"],
                                    "timestamp": datetime.now()
                                })

                except Exception as e:
                    print(f"\n  ✗ Model initialization failed: {e}")
                    # Record failures for all jobs with this model+prompt combo
                    for job_idx, job in enumerate(self.jobs):
                        for judge_name in self.config.judge_models:
                            self.results.append({
                                "job_id": job_idx,
                                "job_title": job["title"],
                                "job_company": job["company"],
                                "job_description": job["description"],
                                "description_length": job["description_length"],
                                "description_language": job["description_language"],
                                "extraction_model": model_name,
                                "prompt_version": prompt_version,
                                "extraction_time": None,
                                "input_tokens": None,
                                "output_tokens": None,
                                "judge_model": judge_name,
                                "extraction_success": False,
                                "error": str(e),
                                "timestamp": datetime.now()
                            })

        return self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        return pd.DataFrame(self.results)


def analyze_results(df: pd.DataFrame):
    """Analyze experiment results with multiple judges
    
    Args:
        df: Results DataFrame from ExperimentRunner
    """

    # Overall model performance BY JUDGE AND PROMPT
    print("\n=== MODEL PERFORMANCE BY JUDGE & PROMPT ===")
    for judge_name in df["judge_model"].unique():
        print(f"\n--- Judge: {judge_name} ---")
        judge_df = df[df["judge_model"] == judge_name]
        summary = judge_df.groupby(["extraction_model", "prompt_version"]).agg({
            "completeness_score": ["mean", "std"],
            "accuracy_score": ["mean", "std"],
            "specificity_score": ["mean", "std"],
            "hallucination_score": ["mean", "std"],
            "extraction_time": ["mean", "std"],
            "input_tokens": ["mean", "sum"],
            "output_tokens": ["mean", "sum"],
            "extraction_success": "mean"
        }).round(2)
        print(summary)

    # Prompt comparison
    print("\n=== PROMPT VERSION COMPARISON ===")
    prompt_comparison = df.groupby("prompt_version").agg({
        "completeness_score": "mean",
        "accuracy_score": "mean",
        "specificity_score": "mean",
        "hallucination_score": "mean",
        "extraction_time": "mean",
        "input_tokens": "mean",
        "output_tokens": "mean"
    }).round(2)
    print(prompt_comparison)

    # Average extraction time per model+prompt
    print("\n=== AVERAGE EXTRACTION TIME (seconds) ===")
    time_summary = df.groupby(["extraction_model", "prompt_version"])["extraction_time"].agg(["mean", "std", "min", "max"]).round(3)
    time_summary = time_summary.sort_values("mean")
    print(time_summary)

    # Token usage summary
    print("\n=== TOKEN USAGE ===")
    token_summary = df.groupby(["extraction_model", "prompt_version"]).agg({
        "input_tokens": ["mean", "sum"],
        "output_tokens": ["mean", "sum"]
    }).round(0)
    print(token_summary)

    # Language breakdown
    print("\n=== DESCRIPTION LANGUAGE BREAKDOWN ===")
    print(df["description_language"].value_counts())

    # Judge agreement analysis
    print("\n=== JUDGE AGREEMENT ===")
    for metric in ["completeness_score", "accuracy_score", "specificity_score", "hallucination_score"]:
        print(f"\n{metric} variance across judges:")
        pivot = df.pivot_table(
            values=metric,
            index=["job_id", "extraction_model", "prompt_version"],
            columns="judge_model",
            aggfunc="mean"
        )
        print(f"  Std dev across judges (mean): {pivot.std(axis=1).mean():.2f}")
        print(f"  Max difference: {(pivot.max(axis=1) - pivot.min(axis=1)).max():.2f}")

    # Best/worst per metric (averaged across judges)
    print("\n=== BEST MODELS PER METRIC (Averaged Across Judges) ===")
    for metric in ["completeness_score", "accuracy_score", "specificity_score", "hallucination_score"]:
        best = df.groupby(["extraction_model", "prompt_version"])[metric].mean().sort_values(ascending=False)
        print(f"\n{metric}:")
        print(best.head(5))

    # Common issues
    print("\n=== COMMON ISSUES ===")

    # Missing info
    all_missing = []
    for missing_list in df["missing_critical_info"].dropna():
        if isinstance(missing_list, list):
            all_missing.extend(missing_list)
    if all_missing:
        print(f"\nMost commonly missed info:")
        print(pd.Series(all_missing).value_counts().head(10))

    # Generic extractions
    all_generic = []
    for generic_list in df["generic_extractions"].dropna():
        if isinstance(generic_list, list):
            all_generic.extend(generic_list)
    if all_generic:
        print(f"\nMost commonly too generic:")
        print(pd.Series(all_generic).value_counts().head(10))

    # Fabrications
    all_fabricated = []
    for fab_list in df["fabricated_info"].dropna():
        if isinstance(fab_list, list):
            all_fabricated.extend(fab_list)
    if all_fabricated:
        print(f"\nMost common fabrications:")
        print(pd.Series(all_fabricated).value_counts().head(10))
