"""Detect claims not supported by evidence."""

import re
from typing import Dict, List, Any


def detect_unsupported_claims(generated: str, evidence: List[str]) -> List[Dict[str, Any]]:
    """Use heuristics to find claims that reference facts not in evidence."""
    sentences = re.split(r'(?<=[.!?])\s+', generated)
    unsupported = []
    
    evidence_text = " ".join(evidence).lower()
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_lower = sentence.lower()
        
        comparisons = ['better', 'more', 'highest', 'most', 'worst', 'less']
        has_comparison = any(comp in sentence_lower for comp in comparisons)
        
        words = set(re.findall(r'\b\w+\b', sentence_lower))
        ev_words = set(re.findall(r'\b\w+\b', evidence_text))
        
        novel_words = words - ev_words
        if len(novel_words) > len(words) * 0.7:
            unsupported.append({
                "claim": sentence.strip(),
                "reason": "High ratio of novel words not found in evidence.",
                "confidence": 0.6
            })
            continue
            
        if has_comparison and not any(comp in evidence_text for comp in comparisons):
             unsupported.append({
                "claim": sentence.strip(),
                "reason": "Contains comparative claims without basis in evidence.",
                "confidence": 0.8
            })
             
    return unsupported


def compute_unsupported_rate(generated: str, evidence: List[str]) -> Dict[str, Any]:
    """Compute the rate of unsupported claims."""
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', generated) if s.strip()]
    if not sentences:
        return {"total_claims": 0, "unsupported": 0, "rate": 0.0, "details": []}
        
    unsupported_details = detect_unsupported_claims(generated, evidence)
    unsupported_count = len(unsupported_details)
    
    rate = unsupported_count / len(sentences) if sentences else 0.0
    return {
        "total_claims": len(sentences),
        "unsupported": unsupported_count,
        "rate": rate,
        "details": unsupported_details
    }
