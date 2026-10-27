"""
Pydantic schemas for the FastAPI application.
"""

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Schema for analysis request."""
    match_id: int = Field(..., description="The ID of the match to analyze.")
    question: str = Field(..., description="The tactical question.")
    team: str = Field(..., description="The team to focus on.")

class AnalysisResponse(BaseModel):
    """Schema for analysis response."""
    response: str = Field(..., description="The generated tactical analysis.")
    evidence: list[str] = Field(..., description="The evidence statements used for generation.")
    processing_time: float = Field(..., description="Time taken to process the request in seconds.")

class MatchInfo(BaseModel):
    """Basic match information."""
    match_id: int
    home_team: str | None = None
    away_team: str | None = None

class MatchListResponse(BaseModel):
    """Schema for match list response."""
    matches: list[MatchInfo]

class MatchTimelineResponse(BaseModel):
    """Schema for returning match timeline data."""
    momentum_timeline: list[dict]


class SummarySearchRequest(BaseModel):
    """Schema for multi-match retrieval requests."""

    question: str = Field(..., description="The tactical question to retrieve evidence for.")
    team: str | None = Field(None, description="Optional team filter.")
    top_k: int = Field(5, ge=1, le=20, description="Number of summaries to retrieve.")


class RetrievedMatchInfo(BaseModel):
    """Information about a retrieved match summary."""

    match_id: int
    home_team: str
    away_team: str
    score: str
    match_date: str
    relevance_score: float


class SummarySearchResponse(BaseModel):
    """Response containing retrieved summaries and evidence."""

    retrieved_matches: list[RetrievedMatchInfo]
    evidence: list[str]


class MultiMatchAnalysisRequest(SummarySearchRequest):
    """Schema for retrieval-augmented multi-match analysis."""


class MultiMatchAnalysisResponse(AnalysisResponse):
    """Generated analysis with retrieval metadata."""

    retrieved_matches: list[RetrievedMatchInfo]
