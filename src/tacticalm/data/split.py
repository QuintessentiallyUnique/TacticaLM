import json
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import train_test_split
from tacticalm.utils.logging import get_logger

logger = get_logger(__name__)

def split_matches(match_ids: List[int], seed: int = 42, train_ratio: float = 0.70, dev_eval_ratio: float = 0.15, final_test_ratio: float = 0.15) -> Dict[str, List[int]]:
    """
    Split matches into train, dev_eval, and final_test sets.
    
    Args:
        match_ids: List of match IDs.
        seed: Random seed for reproducibility.
        train_ratio: Proportion for training set.
        dev_eval_ratio: Proportion for dev/eval set.
        final_test_ratio: Proportion for final test set.
        
    Returns:
        Dictionary with keys 'train', 'dev_eval', 'final_test' and list of match IDs as values.
    """
    if abs(train_ratio + dev_eval_ratio + final_test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
        
    logger.info(f"Splitting {len(match_ids)} matches with ratios {train_ratio}/{dev_eval_ratio}/{final_test_ratio}")
    
    # First split into train and temp
    temp_ratio = dev_eval_ratio + final_test_ratio
    
    if len(match_ids) < 3:
        raise ValueError("Need at least 3 matches to perform a 3-way split.")
        
    train_ids, temp_ids = train_test_split(
        match_ids, 
        test_size=temp_ratio, 
        random_state=seed
    )
    
    # Then split temp into dev_eval and final_test
    test_ratio_of_temp = final_test_ratio / temp_ratio
    dev_eval_ids, final_test_ids = train_test_split(
        temp_ids,
        test_size=test_ratio_of_temp,
        random_state=seed
    )
    
    splits = {
        "train": sorted(train_ids),
        "dev_eval": sorted(dev_eval_ids),
        "final_test": sorted(final_test_ids)
    }
    
    verify_no_leakage(splits)
    
    logger.info(f"Split sizes - Train: {len(splits['train'])}, Dev: {len(splits['dev_eval'])}, Test: {len(splits['final_test'])}")
    
    return splits


def save_splits(splits: Dict[str, List[int]], output_dir: Path) -> None:
    """Save split match IDs to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, ids in splits.items():
        file_path = output_dir / f"{split_name}_matches.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
        logger.info(f"Saved {len(ids)} {split_name} matches to {file_path}")


def load_splits(splits_dir: Path) -> Dict[str, List[int]]:
    """Load split match IDs from JSON files."""
    splits = {}
    for split_name in ["train", "dev_eval", "final_test"]:
        file_path = splits_dir / f"{split_name}_matches.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            splits[split_name] = json.load(f)
            
    logger.info(f"Loaded splits - Train: {len(splits['train'])}, Dev: {len(splits['dev_eval'])}, Test: {len(splits['final_test'])}")
    return splits


def verify_no_leakage(splits: Dict[str, List[int]]) -> bool:
    """
    Verify no match appears in multiple splits.
    Raises ValueError if leakage is detected.
    """
    train_set = set(splits.get("train", []))
    dev_set = set(splits.get("dev_eval", []))
    test_set = set(splits.get("final_test", []))
    
    # Check intersections
    if train_set.intersection(dev_set):
        raise ValueError(f"Data leakage detected between train and dev_eval sets: {train_set.intersection(dev_set)}")
    
    if train_set.intersection(test_set):
        raise ValueError(f"Data leakage detected between train and final_test sets: {train_set.intersection(test_set)}")
        
    if dev_set.intersection(test_set):
        raise ValueError(f"Data leakage detected between dev_eval and final_test sets: {dev_set.intersection(test_set)}")
        
    return True
