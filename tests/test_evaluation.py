"""Tests for the evaluation framework."""

from tacticalm.evaluation.numerical import extract_all_numbers, find_number_in_evidence, compute_numerical_accuracy
from tacticalm.evaluation.factuality import check_prohibited_claims
from tacticalm.evaluation.specificity import score_tactical_specificity


def test_extract_all_numbers():
    text = "Player had 55.5% possession and 3 shots, with 0.5 xG."
    nums = extract_all_numbers(text)
    assert len(nums) == 3
    values = [n["value"] for n in nums]
    assert 55.5 in values
    assert 3.0 in values
    assert 0.5 in values


def test_find_number_in_evidence():
    evidence = ["Team A had 55.5% possession."]
    assert find_number_in_evidence({"value": 55.5}, evidence, tolerance=0.01)
    assert not find_number_in_evidence({"value": 60.0}, evidence, tolerance=0.01)


def test_compute_numerical_accuracy():
    generated = "Team A won 2-1 with 55.5% possession."
    evidence = ["Team A won 2-1.", "Possession was 55.5 percent."]
    res = compute_numerical_accuracy(generated, evidence)
    assert res["total_numbers"] == 3
    assert res["accuracy"] == 1.0


def test_check_prohibited_claims():
    generated = "Team A was awful today."
    prohibited = ["Team A was awful today."]
    res = check_prohibited_claims(generated, prohibited)
    assert res["violation_count"] == 1
    assert "Team A was awful today." in res["violations"]


def test_score_tactical_specificity():
    # Text must contain 4+ tactical terms and 3+ numbers for score >= 4
    text = (
        "They used a high press and counter-press to overload the half-space, "
        "completing 12 progressive passes with 65.3% possession and creating "
        "2.5 expected goals from their build-up play."
    )
    res = score_tactical_specificity(text)
    assert res["score"] >= 4
    assert len(res["tactical_terms_found"]) >= 4
