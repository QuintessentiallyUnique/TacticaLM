"""Tactical specificity scoring."""

from typing import Dict, Any, List
import re

TACTICAL_TERMS = {
    "half-space", "pressing", "counter-press", "overload", "underlap",
    "overlap", "block", "low block", "mid block", "high press",
    "transition", "expected goals", "xg", "ppda", "turnover", "regain",
    "possession", "progressive pass", "through ball", "cutback",
    "third man", "build-up", "chance creation"
}

def score_tactical_specificity(text: str) -> Dict[str, Any]:
    """Rule-based heuristic scoring on 1-5 scale."""
    text_lower = text.lower()
    
    found_terms = [term for term in TACTICAL_TERMS if term in text_lower]
    
    numbers_cited = len(re.findall(r'\b\d+\.?\d*\b', text))
    
    score = 1
    reasoning = "Generic or irrelevant"
    
    if len(found_terms) > 0 or numbers_cited > 0:
        score = 2
        reasoning = "Limited tactical content"
        
    if len(found_terms) >= 2 and numbers_cited >= 1:
        score = 3
        reasoning = "Moderately specific"
        
    if len(found_terms) >= 4 and numbers_cited >= 3:
        score = 4
        reasoning = "Strong and evidence-based"
        
    if len(found_terms) >= 6 and numbers_cited >= 5:
        score = 5
        reasoning = "Highly specific, accurate and useful"
        
    return {
        "score": score,
        "reasoning": reasoning,
        "tactical_terms_found": found_terms,
        "numbers_cited": numbers_cited
    }
