"""Local retrieval over processed match summaries.

This is intentionally lightweight: it indexes the existing summary JSON files
at request time and uses TF-IDF when scikit-learn is available, with a token
overlap fallback for minimal environments.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tacticalm.dataset.schemas import MatchSummary
from tacticalm.features.summary import generate_evidence_statements
from tacticalm.utils.paths import PROCESSED_DIR


@dataclass
class RetrievedSummary:
    """A match summary selected for a query."""

    summary: MatchSummary
    score: float
    evidence: list[str]


def load_summaries(summaries_dir: Path | None = None) -> list[MatchSummary]:
    """Load all processed match summaries from disk."""
    root = summaries_dir or PROCESSED_DIR / "summaries"
    if not root.exists():
        return []

    summaries = []
    for path in sorted(root.glob("*_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            summaries.append(MatchSummary(**json.load(f)))
    return summaries


def _team_matches(summary: MatchSummary, team: str | None) -> bool:
    if not team:
        return True
    query = team.lower()
    teams = [summary.home_team.lower(), summary.away_team.lower()]
    return any(query in candidate for candidate in teams)


def _teams_for_evidence(summary: MatchSummary, team: str | None) -> list[str]:
    if team and team in summary.team_stats:
        return [team]
    if team:
        for team_name in summary.team_stats:
            if team.lower() in team_name.lower():
                return [team_name]
    return [summary.home_team, summary.away_team]


def build_retrieval_text(summary: MatchSummary, team: str | None = None) -> str:
    """Build a compact searchable text document for one summary."""
    parts = [
        str(summary.match_id),
        summary.competition,
        summary.season,
        summary.match_date,
        summary.home_team,
        summary.away_team,
        f"{summary.score_home}-{summary.score_away}",
    ]
    for team_name in _teams_for_evidence(summary, team):
        parts.extend(generate_evidence_statements(summary, team_name)[:30])
    return " ".join(parts)


def build_retrieval_evidence(
    summary: MatchSummary,
    team: str | None = None,
    max_evidence_per_team: int = 20,
) -> list[str]:
    """Build evidence statements with match identity attached."""
    header = (
        f"Match {summary.match_id}: {summary.home_team} vs {summary.away_team} "
        f"on {summary.match_date}, score {summary.score_home}-{summary.score_away}."
    )
    evidence = [header]
    for team_name in _teams_for_evidence(summary, team):
        evidence.extend(generate_evidence_statements(summary, team_name)[:max_evidence_per_team])
    return evidence


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _fallback_scores(query: str, documents: list[str]) -> list[float]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0 for _ in documents]
    scores = []
    for document in documents:
        doc_tokens = _tokenize(document)
        overlap = len(query_tokens & doc_tokens)
        scores.append(overlap / max(len(query_tokens), 1))
    return scores


def _tfidf_scores(query: str, documents: list[str]) -> list[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return _fallback_scores(query, documents)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([query, *documents])
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(score) for score in scores]


def retrieve_relevant_summaries(
    query: str,
    summaries_dir: Path | None = None,
    team: str | None = None,
    top_k: int = 5,
) -> list[RetrievedSummary]:
    """Retrieve the most relevant summaries for a multi-match question."""
    top_k = max(1, min(top_k, 20))
    candidates = [
        summary for summary in load_summaries(summaries_dir) if _team_matches(summary, team)
    ]
    if not candidates:
        return []

    documents = [build_retrieval_text(summary, team) for summary in candidates]
    scores = _tfidf_scores(query, documents)
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        RetrievedSummary(
            summary=summary,
            score=round(score, 4),
            evidence=build_retrieval_evidence(summary, team),
        )
        for summary, score in ranked[:top_k]
    ]
