"""Generate QA datasets (dev_eval and final_test).

Usage:
    py -3.12 scripts/generate_datasets.py
"""

import sys
import json
from pathlib import Path

from tacticalm.data.split import load_splits
from tacticalm.dataset.generate_examples import generate_dev_eval_set, generate_final_test_set, generate_training_examples
from tacticalm.utils.paths import PROCESSED_DIR

def main():
    print("=" * 70)
    print("TacticaLM - Generating QA Datasets")
    print("=" * 70)
    
    splits_dir = PROCESSED_DIR / "splits"
    summaries_dir = PROCESSED_DIR / "summaries"
    
    print("Loading splits...")
    splits = load_splits(splits_dir)
    train_ids = splits.get("train", [])
    dev_ids = splits.get("dev_eval", [])
    test_ids = splits.get("final_test", [])
    
    print(f"Found splits: {len(train_ids)} train, {len(dev_ids)} dev eval, {len(test_ids)} final test.")
    
    print("\nGenerating dev eval set...")
    dev_examples = generate_dev_eval_set(dev_ids, summaries_dir)
    print(f"Generated {len(dev_examples)} dev eval questions.")
    
    print("\nGenerating final test set...")
    test_examples = generate_final_test_set(test_ids, summaries_dir)
    print(f"Generated {len(test_examples)} final test questions.")
    
    print(f"\nGenerating pilot training examples (from {len(train_ids)} train matches)...")
    train_examples = generate_training_examples(train_ids, summaries_dir, num_per_match=8)
    
    train_path = summaries_dir.parent.parent / "evaluation" / "train_pilot.json"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in train_examples], f, indent=2)
    print(f"Generated {len(train_examples)} pilot training questions. Saved to {train_path}.")
    
    print("\nDataset generation complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
