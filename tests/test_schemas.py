import pytest
from pydantic import ValidationError
from jevk5_server.schemas import (
    SystemOneRequest,
    SystemOneResponse,
    NoulQuestion,
    ChoiceQuestion,
    ScoreQuestion,
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    Usage,
    ModelsResponse,
)


def test_noul_question_valid():
    q = NoulQuestion(instructions="Is this time sensitive?")
    assert q.type == "noul"
    assert q.instructions == "Is this time sensitive?"
    assert q.criteria is None


def test_choice_question_validation():
    # Valid choice criteria dict
    q = ChoiceQuestion(
        instructions="Select team",
        criteria={"billing": "Billing", "support": "Support"},
    )
    assert q.type == "choice"
    assert len(q.criteria) == 2

    # Reject if less than 2 options
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            instructions="Select team",
            criteria={"billing": "Billing"},
        )


def test_score_question_validation():
    # Valid score criteria list (2 to 10)
    q = ScoreQuestion(
        instructions="Rate 1-3",
        criteria=["Low", "Medium", "High"],
    )
    assert q.type == "score"
    assert len(q.criteria) == 3

    # Reject if less than 2 levels
    with pytest.raises(ValidationError):
        ScoreQuestion(
            instructions="Rate 1",
            criteria=["Low"],
        )

    # Reject if more than 10 levels
    with pytest.raises(ValidationError):
        ScoreQuestion(
            instructions="Rate 1-11",
            criteria=[f"Level {i}" for i in range(11)],
        )


def test_system_one_request_parsing():
    payload = {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this convey urgency?",
            },
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    }
    req = SystemOneRequest.model_validate(payload)
    assert req.model == "jev-latest"
    assert len(req.questions) == 3
    assert req.questions["is_urgent"].type == "noul"
    assert req.questions["department"].type == "choice"
    assert req.questions["frustration"].type == "score"


def test_system_one_response_structure():
    resp_data = {
        "model": "jev-1.13.0",
        "answers": {
            "is_urgent": NoulAnswer(type="noul", noul=0.95),
            "department": ChoiceAnswer(
                type="choice",
                choice="billing",
                probabilities={"billing": 0.88, "technical": 0.12, "sales": 0.0},
                confidence=0.81,
            ),
            "frustration": ScoreAnswer(
                type="score",
                score=1.05,
                legend={"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                probabilities={"0": 0.0, "1": 0.95, "2": 0.05},
                confidence=0.92,
            ),
        },
        "usage": Usage(input_tokens=304, output_tokens=20),
    }
    resp = SystemOneResponse.model_validate(resp_data)
    d = resp.model_dump()
    assert d["model"] == "jev-1.13.0"
    assert d["answers"]["is_urgent"] == {"type": "noul", "noul": 0.95}
    assert "confidence" not in d["answers"]["is_urgent"]
    assert d["answers"]["department"]["confidence"] == 0.81
    assert d["answers"]["frustration"]["score"] == 1.05
    assert d["answers"]["frustration"]["legend"]["0"] == "Calm"
