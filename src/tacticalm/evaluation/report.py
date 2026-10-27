"""Aggregate and export evaluation results."""

import json
from pathlib import Path
from typing import Dict, List, Any


def aggregate_results(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean, median, std for each metric across all examples."""
    import statistics
    from collections import defaultdict
    
    metrics = defaultdict(list)
    category_metrics = defaultdict(lambda: defaultdict(list))
    
    for eval_res in evaluations:
        cat = eval_res.get("category", "unknown")
        
        if "factual_accuracy" in eval_res:
             metrics["factual_accuracy"].append(eval_res["factual_accuracy"])
             category_metrics[cat]["factual_accuracy"].append(eval_res["factual_accuracy"])
             
    def compute_stats(values):
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }
        
    aggregated = {m: compute_stats(vals) for m, vals in metrics.items()}
    category_aggregated = {
        cat: {m: compute_stats(vals) for m, vals in cat_mets.items()}
        for cat, cat_mets in category_metrics.items()
    }
    
    return {
        "overall": aggregated,
        "by_category": category_aggregated
    }


def generate_comparison_table(baseline_a: Dict[str, Any], baseline_b: Dict[str, Any], finetuned: Dict[str, Any] = None) -> str:
    """Markdown table comparing systems."""
    table = "| Metric | Baseline A | Baseline B |"
    if finetuned:
        table += " Finetuned |\n"
        table += "|--------|------------|------------|-----------|\n"
    else:
        table += "\n|--------|------------|------------|\n"
        
    metrics = ["factual_accuracy", "numerical_accuracy"]
    for m in metrics:
        val_a = baseline_a.get("overall", {}).get(m, {}).get("mean", 0.0)
        val_b = baseline_b.get("overall", {}).get(m, {}).get("mean", 0.0)
        row = f"| {m} | {val_a:.3f} | {val_b:.3f} |"
        if finetuned:
            val_f = finetuned.get("overall", {}).get(m, {}).get("mean", 0.0)
            row += f" {val_f:.3f} |"
        table += row + "\n"
        
    return table


def export_report(results: Dict[str, Any], output_dir: Path) -> Path:
    """Save full report as JSON + summary as markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    md_path = output_dir / "report.md"
    with open(md_path, "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write(json.dumps(results, indent=2))
        
    return json_path
