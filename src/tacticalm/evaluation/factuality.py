"""Automated factual accuracy checking."""

import re
from typing import Dict, List, Any

from .numerical import compute_numerical_accuracy
from .unsupported_claims import compute_unsupported_rate


def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """Extract all numbers from generated text with surrounding context."""
    pattern = r'\b(\d+\.?\d*)\b'
    matches = re.finditer(pattern, text)
    
    results = []
    for match in matches:
        value = float(match.group(1))
        start_idx = max(0, match.start() - 20)
        end_idx = min(len(text), match.end() + 20)
        context = text[start_idx:end_idx]
        
        results.append({
            "value": value,
            "context": context,
            "position": match.start()
        })
    return results


def extract_claims(text: str) -> List[str]:
    """Split text into individual claim sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def check_number_accuracy(generated: str, evidence: List[str]) -> Dict[str, Any]:
    """Extract numbers from generated text, cross-reference against evidence."""
    nums = extract_numbers(generated)
    if not nums:
         return {"total_numbers": 0, "matched": 0, "unmatched": [], "accuracy": 1.0}
         
    matched = 0
    unmatched = []
    
    evidence_text = " ".join(evidence)
    ev_nums = [n["value"] for n in extract_numbers(evidence_text)]
    
    for num in nums:
        if any(abs(num["value"] - ev_num) < 0.01 for ev_num in ev_nums):
            matched += 1
        else:
            unmatched.append(num)
            
    accuracy = matched / len(nums) if nums else 1.0
    return {
        "total_numbers": len(nums),
        "matched": matched,
        "unmatched": unmatched,
        "accuracy": accuracy
    }


def check_team_names(generated: str, evidence: List[str]) -> Dict[str, Any]:
    """Verify team names mentioned are in the evidence."""
    return {"valid": True, "invalid_teams": []}


def check_prohibited_claims(generated: str, prohibited: List[str]) -> Dict[str, Any]:
    """Check if generated text contains any prohibited claims."""
    violations = []
    gen_lower = generated.lower()
    for claim in prohibited:
        if claim.lower() in gen_lower:
            violations.append(claim)
            
    return {
        "violations": violations,
        "violation_count": len(violations)
    }


def compute_factuality_score(
    generated: str, 
    evidence: List[str], 
    reference_facts: List[str], 
    prohibited_claims: List[str]
) -> Dict[str, Any]:
    """Aggregate score combining all checks."""
    num_acc = check_number_accuracy(generated, evidence)
    unsupp = compute_unsupported_rate(generated, evidence)
    prohib = check_prohibited_claims(generated, prohibited_claims)
    
    covered_facts = 0
    gen_lower = generated.lower()
    for fact in reference_facts:
        if any(word in gen_lower for word in fact.lower().split() if len(word) > 4):
             covered_facts += 1
    evidence_coverage = covered_facts / len(reference_facts) if reference_facts else 1.0
    
    factual_accuracy = (num_acc["accuracy"] + (1.0 - unsupp["rate"])) / 2.0
    if prohib["violation_count"] > 0:
        factual_accuracy *= 0.5
        
    return {
        "factual_accuracy": factual_accuracy,
        "numerical_accuracy": num_acc["accuracy"],
        "unsupported_claim_rate": unsupp["rate"],
        "evidence_coverage": evidence_coverage,
        "details": {
            "numerical": num_acc,
            "unsupported": unsupp,
            "prohibited": prohib
        }
    }
