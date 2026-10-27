import argparse
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from tacticalm.dataset.prompts import format_analysis_prompt, format_system_prompt
from tacticalm.dataset.schemas import EvaluationExample
from tacticalm.evaluation.factuality import compute_factuality_score
from tacticalm.inference.model_loader import load_model_for_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def format_prompt(example: EvaluationExample) -> str:
    """Format an evaluation prompt with the same evidence contract as production."""
    user_msg = format_analysis_prompt(
        question=example.question,
        evidence=example.evidence,
        context=example.context.model_dump(),
    )

    return (
        f"<|im_start|>system\n{format_system_prompt()}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def main():
    from tacticalm.evaluation.run_baseline import run_inference

    parser = argparse.ArgumentParser(description="Run tactical model evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset JSON")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter (if testing fine-tuned model)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples for quick testing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    with open(args.dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = [EvaluationExample(**item) for item in data]
    if args.limit:
        examples = examples[:args.limit]
    
    logger.info(f"Loaded {len(examples)} evaluation examples.")

    # Load model
    logger.info("Loading model for inference (this may take a minute)...")
    model, tokenizer = load_model_for_inference(base_model=args.base_model, adapter_path=args.adapter)

    results = []
    
    logger.info("Starting evaluation run...")
    for example in tqdm(examples, desc="Evaluating"):
        prompt = format_prompt(example)
        
        start_time = time.time()
        # Ensure generation doesn't hallucinate too wildly by keeping temp low
        generated_text = run_inference(model, tokenizer, prompt, max_new_tokens=400, temperature=0.1)
        generation_time = time.time() - start_time
        
        # Run automated evaluation checks
        eval_score = compute_factuality_score(
            generated=generated_text,
            evidence=example.evidence,
            reference_facts=example.reference_facts,
            prohibited_claims=example.prohibited_claims
        )
        
        results.append({
            "example_id": example.example_id,
            "category": example.category.value,
            "question": example.question,
            "generated_answer": generated_text,
            "metrics": eval_score,
            "generation_time_seconds": generation_time
        })

    # Save raw results
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Compute aggregates (Factuality, Numerical Accuracy, Unsupported Claims)
    logger.info("Aggregating results...")
    aggregate_scores = {
        "overall_factual_accuracy": sum(r["metrics"]["factual_accuracy"] for r in results) / len(results),
        "overall_numerical_accuracy": sum(r["metrics"]["numerical_accuracy"] for r in results) / len(results),
        "overall_unsupported_claim_rate": sum(r["metrics"]["unsupported_claim_rate"] for r in results) / len(results),
    }
    
    logger.info("=== EVALUATION RESULTS ===")
    for k, v in aggregate_scores.items():
        logger.info(f"{k}: {v:.2%}")
        
    with open(output_dir / "aggregate_metrics.json", 'w') as f:
        json.dump(aggregate_scores, f, indent=2)

    logger.info(f"Full evaluation saved to {output_dir}")

if __name__ == "__main__":
    main()
