"""Number extraction and cross-referencing."""

import re
from typing import Dict, List, Any


def extract_all_numbers(text: str) -> List[Dict[str, Any]]:
    """Comprehensive number extraction."""
    pattern = r'\b(\d+\.?\d*)\s*(%|/10|/xG|xG)?\b'
    matches = re.finditer(pattern, text)
    
    results = []
    for match in matches:
        raw = match.group(0)
        val_str = match.group(1)
        suffix = match.group(2)
        
        value = float(val_str)
        num_type = "number"
        if suffix == "%":
            num_type = "percentage"
        elif suffix in ["/10", "/xG", "xG"]:
            num_type = suffix
            
        start_idx = max(0, match.start() - 20)
        end_idx = min(len(text), match.end() + 20)
        context = text[start_idx:end_idx]
        
        results.append({
            "raw": raw,
            "value": value,
            "type": num_type,
            "context": context
        })
    return results


def find_number_in_evidence(number: Dict[str, Any], evidence: List[str], tolerance: float = 0.01) -> bool:
    """Check if a number appears in the evidence (with tolerance for rounding)."""
    for ev in evidence:
        ev_numbers = extract_all_numbers(ev)
        for ev_num in ev_numbers:
            if abs(ev_num["value"] - number["value"]) <= tolerance:
                return True
    return False


def compute_numerical_accuracy(generated: str, evidence: List[str]) -> Dict[str, Any]:
    """Compute numerical accuracy against evidence."""
    numbers = extract_all_numbers(generated)
    if not numbers:
        return {"total_numbers": 0, "verified": 0, "unverified": [], "accuracy": 1.0}
        
    verified = 0
    unverified = []
    for num in numbers:
        if find_number_in_evidence(num, evidence):
            verified += 1
        else:
            unverified.append(num)
            
    accuracy = verified / len(numbers) if numbers else 1.0
    return {
        "total_numbers": len(numbers),
        "verified": verified,
        "unverified": unverified,
        "accuracy": accuracy
    }
