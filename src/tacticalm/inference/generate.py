"""
Generation module for tactical analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class TacticalGenerator:
    """
    Generator class for running inference with TacticaLM.
    """
    
    def __init__(self, model: Any, tokenizer: Any):
        """
        Initializes the TacticalGenerator.
        
        Args:
            model: The loaded language model.
            tokenizer: The loaded tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_analysis(self, question: str, match_id: int, team: str) -> dict[str, Any]:
        """
        Generates tactical analysis for a given question and match.
        
        Args:
            question (str): The tactical question.
            match_id (int): The ID of the match.
            team (str): The team name.
            
        Returns:
            Dict[str, Any]: A dictionary containing the response, evidence, and metadata.
        """
        try:
            from tacticalm.features.summary import generate_evidence_statements
        except ImportError as e:
            logger.error(f"Failed to import required modules for generation: {e}")
            raise
            
        summary_path = Path(f"data/processed/summaries/{match_id}_summary.json")
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Match summary not found at {summary_path}")
            
        with open(summary_path, 'r', encoding='utf-8') as f:
            match_summary_dict = json.load(f)
            
        from tacticalm.dataset.schemas import MatchSummary
        match_summary = MatchSummary(**match_summary_dict)
            
        # Resolve team alias
        if team.lower() == "home team":
            team = match_summary.home_team
        elif team.lower() == "away team":
            team = match_summary.away_team
            
        # Generate evidence based on the match summary and team
        evidence = generate_evidence_statements(match_summary, team)
        
        context_dict = {
            "competition": match_summary.competition,
            "season": match_summary.season,
            "match_date": match_summary.match_date,
            "home_team": match_summary.home_team,
            "away_team": match_summary.away_team,
            "score": f"{match_summary.score_home}-{match_summary.score_away}"
        }
        
        response_text = self.generate_from_evidence(
            question=question,
            evidence=evidence,
            context=context_dict,
        )

        return {
            "response": response_text,
            "evidence": evidence,
            "metadata": {
                "match_id": match_id,
                "team": team,
                "question": question
            }
        }

    def generate_from_evidence(
        self,
        question: str,
        evidence: list[str],
        context: dict[str, Any],
        max_new_tokens: int = 1024,
    ) -> str:
        """Generate analysis from already-retrieved evidence."""
        try:
            import torch

            from tacticalm.dataset.prompts import format_analysis_prompt, format_system_prompt
        except ImportError as e:
            logger.error(f"Failed to import required modules for generation: {e}")
            raise

        system_prompt = format_system_prompt()
        base_prompt = format_analysis_prompt(
            question=question,
            evidence=evidence,
            context=context,
        )
        analysis_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{base_prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize prompt
        inputs = self.tokenizer(analysis_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode response, removing the prompt
        input_length = inputs.input_ids.shape[1]
        response_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        return response_text
