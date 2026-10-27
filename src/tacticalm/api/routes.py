import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from tacticalm.api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    MatchInfo,
    MatchListResponse,
    MatchTimelineResponse,
    MultiMatchAnalysisRequest,
    MultiMatchAnalysisResponse,
    RetrievedMatchInfo,
    SummarySearchRequest,
    SummarySearchResponse,
)
from tacticalm.retrieval.summaries import RetrievedSummary, retrieve_relevant_summaries

logger = logging.getLogger(__name__)

router = APIRouter()

# Global tactical generator instance (can be populated at startup)
_generator = None

def get_generator():
    """Dependency to get the generator."""
    if _generator is None:
        raise HTTPException(status_code=503, detail="Model generator is not loaded.")
    return _generator


GENERATOR_DEPENDENCY = Depends(get_generator)


def _retrieved_match_info(result: RetrievedSummary) -> RetrievedMatchInfo:
    summary = result.summary
    return RetrievedMatchInfo(
        match_id=summary.match_id,
        home_team=summary.home_team,
        away_team=summary.away_team,
        score=f"{summary.score_home}-{summary.score_away}",
        match_date=summary.match_date,
        relevance_score=result.score,
    )


def _retrieve_for_request(request: SummarySearchRequest) -> list[RetrievedSummary]:
    results = retrieve_relevant_summaries(
        query=request.question,
        team=request.team,
        top_k=request.top_k,
    )
    if not results:
        raise HTTPException(status_code=404, detail="No matching summaries found.")
    return results

@router.get("/health", response_model=dict)
def health_check():
    """Health check endpoint."""
    status = "healthy" if _generator is not None else "degraded (model not loaded)"
    return {"status": status}

@router.get("/matches", response_model=MatchListResponse)
def list_matches():
    """Lists available matches based on processed summaries."""
    summary_dir = Path("data/processed/summaries")
    matches = []
    
    if summary_dir.exists():
        for file_path in summary_dir.glob("*_summary.json"):
            try:
                # Expecting format {match_id}_summary.json
                match_id_str = file_path.name.split("_")[0]
                if match_id_str.isdigit():
                    import json
                    with open(file_path, "r", encoding="utf-8") as f:
                        summary_data = json.load(f)
                    home_team = summary_data.get("home_team")
                    away_team = summary_data.get("away_team")
                    matches.append(MatchInfo(match_id=int(match_id_str), home_team=home_team, away_team=away_team))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not parse match from {file_path}: {e}")
                
    return MatchListResponse(matches=matches)


@router.post("/matches/search", response_model=SummarySearchResponse)
def search_matches(request: SummarySearchRequest):
    """Retrieve relevant summaries for a multi-match tactical question."""
    results = _retrieve_for_request(request)
    evidence = [statement for result in results for statement in result.evidence]

    return SummarySearchResponse(
        retrieved_matches=[_retrieved_match_info(result) for result in results],
        evidence=evidence,
    )

@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest, generator=GENERATOR_DEPENDENCY):
    """
    Generates a tactical analysis based on the provided request.
    """
    start_time = time.time()
    
    try:
        result = generator.generate_analysis(
            question=request.question,
            match_id=request.match_id,
            team=request.team
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error during analysis generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during generation.")
        
    processing_time = time.time() - start_time
    
    return AnalysisResponse(
        response=result["response"],
        evidence=result["evidence"],
        processing_time=processing_time
    )


@router.post("/analyze/multi-match", response_model=MultiMatchAnalysisResponse)
def analyze_multi_match(
    request: MultiMatchAnalysisRequest,
    generator=GENERATOR_DEPENDENCY,
):
    """Generate retrieval-augmented analysis across multiple match summaries."""
    start_time = time.time()
    results = _retrieve_for_request(request)
    evidence = [statement for result in results for statement in result.evidence]

    context = {
        "competition": "Multiple matches",
        "season": "Multiple seasons",
        "match_date": "Multiple dates",
        "home_team": request.team or "Retrieved teams",
        "away_team": "Retrieved opponents",
        "score": "Multiple scores",
    }

    try:
        response = generator.generate_from_evidence(
            question=request.question,
            evidence=evidence,
            context=context,
            max_new_tokens=1024,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error during multi-match analysis generation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during multi-match generation.",
        )

    return MultiMatchAnalysisResponse(
        response=response,
        evidence=evidence,
        processing_time=time.time() - start_time,
        retrieved_matches=[_retrieved_match_info(result) for result in results],
    )

@router.get("/matches/{match_id}/timeline", response_model=MatchTimelineResponse)
def get_match_timeline(match_id: int):
    """Fetches the momentum timeline for a specific match."""
    summary_path = Path(f"data/processed/summaries/{match_id}_summary.json")
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Match summary not found.")
    
    import json
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
        
    return MatchTimelineResponse(momentum_timeline=summary_data.get("momentum_timeline", []))
