"""Prompts and question templates for TacticaLM."""

from tacticalm.dataset.schemas import AnalysisCategory

SYSTEM_PROMPT = """You are an expert football tactical analyst.
Your task is to analyze matches using only the provided statistical evidence.

Rules:
1. ONLY reference facts from the provided evidence. Do not hallucinate or use outside knowledge.
2. Cite specific numbers and statistics when making claims.
3. If the provided evidence is insufficient to answer the question, explicitly acknowledge this limitation.
4. Use appropriate football tactical terminology correctly.
5. Structure your analysis with clear reasoning and logic.
6. Write your analysis in continuous natural language paragraphs. DO NOT repeat the evidence back as a bulleted list. You must provide a cohesive, analytical narrative.
7. Important Context for Spatial Coordinates: The pitch X coordinate goes from 0 to 120 (0 is own goal, 120 is opponent's goal). x=60 is midfield. x=80 to 120 is the attacking third.
8. Important Context for Time Intervals: "0-14'" means the first 15 minutes of the match, NOT the entire first half. The first half is 0-45'. The second half is 45-90'.
9. When reading tables, do not hallucinate missing numbers. For example, if a team has 68% possession in an interval, just state they had 68% possession."""

QUESTION_TEMPLATES = {
    AnalysisCategory.MATCH_SUMMARY: [
        "Provide a comprehensive match summary for {team} against {opponent}.",
        "How did {team} perform overall in their match against {opponent}?",
        "Summarize the tactical performance of {team} in this match.",
        "Give an overview of {team}'s statistics and performance against {opponent}."
    ],
    AnalysisCategory.BUILD_UP_ANALYSIS: [
        "Analyze {team}'s build-up play and passing patterns against {opponent}.",
        "How effectively did {team} progress the ball and maintain possession?",
        "Describe {team}'s passing statistics and build-up effectiveness in this match.",
        "What do the statistics say about {team}'s passing and ball progression?"
    ],
    AnalysisCategory.PRESSING_ANALYSIS: [
        "Evaluate {team}'s pressing intensity and high regains against {opponent}.",
        "How aggressive was {team}'s pressing scheme in this match?",
        "Analyze {team}'s defensive actions in the attacking third.",
        "What do the PPDA and high regain stats indicate about {team}'s pressing?"
    ],
    AnalysisCategory.CHANCE_CREATION: [
        "How did {team} create chances and threaten the goal against {opponent}?",
        "Analyze {team}'s shooting and expected goals (xG) in this match.",
        "Describe {team}'s effectiveness in the final third and chance creation.",
        "What do the statistics reveal about {team}'s attacking threat and shot quality?"
    ],
    AnalysisCategory.DEFENSIVE_PERFORMANCE: [
        "Assess {team}'s defensive performance and actions against {opponent}.",
        "How well did {team} defend in terms of tackles, interceptions, and blocks?",
        "Analyze the defensive solidity of {team} based on their defensive actions.",
        "What do the defensive statistics indicate about {team}'s performance?"
    ],
    AnalysisCategory.TRANSITION_ANALYSIS: [
        "Analyze {team}'s performance in transition and counterattacks.",
        "How effectively did {team} utilize transitions and turnovers against {opponent}?",
        "Describe {team}'s counterattacking threat and shots after regains.",
        "What do the statistics say about {team}'s transitional play?"
    ],
    AnalysisCategory.TEAM_COMPARISON: [
        "Compare the statistical performance of {team} and {opponent} in this match.",
        "How did {team} and {opponent} compare across key tactical metrics?",
        "Provide a comparative analysis of {team} and {opponent} based on the evidence.",
        "Who had the statistical edge between {team} and {opponent} and why?"
    ],
    AnalysisCategory.PLAYER_CONTRIBUTION: [
        "Analyze the individual contribution of players for {team} in this match.",
        "How did key players perform for {team} based on the statistics?",
        "Describe the individual player metrics and their impact for {team}.",
        "What were the standout individual performances for {team}?"
    ],
    AnalysisCategory.EVIDENCE_EXTRACTION: [
        "What specific statistics describe {team}'s possession and passing?",
        "Extract the key metrics related to {team}'s defensive actions.",
        "Identify the specific data points regarding {team}'s expected goals and shots.",
        "What evidence is available regarding {team}'s pressing and regains?"
    ],
    AnalysisCategory.LIMITATION_AWARE: [
        "Based on the evidence, what were the manager's tactical instructions for {team}?",
        "How did the crowd's atmosphere affect {team}'s performance against {opponent}?",
        "Explain the emotional state of {team}'s players during the match.",
        "Why did {team} choose this specific formation against {opponent}?"
    ]
}

def format_system_prompt() -> str:
    """Return the system prompt for the TacticaLM model."""
    return SYSTEM_PROMPT

def format_analysis_prompt(question: str, evidence: list[str], context: dict) -> str:
    """Format a complete analysis prompt with question, context, and evidence.
    
    Args:
        question: The tactical question to answer.
        evidence: A list of factual evidence statements.
        context: Match context dictionary.
        
    Returns:
        The formatted prompt string.
    """
    context_str = (
        f"Match Context:\n"
        f"Competition: {context.get('competition', 'Unknown')}\n"
        f"Season: {context.get('season', 'Unknown')}\n"
        f"Date: {context.get('match_date', 'Unknown')}\n"
        f"Match: {context.get('home_team', 'Unknown')} vs {context.get('away_team', 'Unknown')}\n"
        f"Score: {context.get('score', 'Unknown')}"
    )
    
    evidence_str = "Evidence:\n" + "\n".join(f"- {e}" for e in evidence) if evidence else "Evidence:\nNone"
    
    return f"{context_str}\n\n{evidence_str}\n\nQuestion: {question}\n\nAnswer:"
