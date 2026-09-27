import pytest
from unittest.mock import MagicMock
from jevk5 import JevK5GGUF

from jevk5_server.adapter import JevK5TypeSafeAdapter
from jevk5_server.schemas import SystemOneRequest


def test_adapter_initialization_with_jevk5gguf():
    mock_model = MagicMock(spec=JevK5GGUF)
    adapter = JevK5TypeSafeAdapter(model=mock_model, default_model_name="jevk5-4b-v0.3-Q8_0")
    assert adapter.model is mock_model
    assert adapter.default_model_name == "jevk5-4b-v0.3-Q8_0"


@pytest.mark.anyio
async def test_adapter_evaluates_system_one_request():
    mock_model = MagicMock(spec=JevK5GGUF)

    def mock_probabilities(state, question):
        qtype = question["type"]
        if qtype == "noul":
            return {"true": 0.95, "false": 0.05}, 307
        elif qtype == "choice":
            return {"billing": 0.88, "technical": 0.12, "sales": 0.0}, 318
        elif qtype == "score":
            return {"0": 0.0, "1": 0.95, "2": 0.05}, 304
        return {}, 0

    mock_model.probabilities.side_effect = mock_probabilities

    adapter = JevK5TypeSafeAdapter(model=mock_model, default_model_name="jevk5-4b-v0.3-Q8_0")

    request = SystemOneRequest.model_validate({
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
    })

    resp = await adapter.evaluate(request)

    assert resp.model == "jev-latest"
    assert len(resp.answers) == 3

    # Noul answer check
    noul = resp.answers["is_urgent"]
    assert noul.type == "noul"
    assert noul.noul == 0.95
    assert not hasattr(noul, "confidence")

    # Choice answer check
    choice = resp.answers["department"]
    assert choice.type == "choice"
    assert choice.choice == "billing"
    assert choice.probabilities == {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    # (3 * 0.88 - 1) / 2 = 1.64 / 2 = 0.82
    assert choice.confidence == 0.82

    # Score answer check
    score = resp.answers["frustration"]
    assert score.type == "score"
    assert score.score == 1.05  # 0*0 + 1*0.95 + 2*0.05 = 1.05
    assert score.legend == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert score.probabilities == {"0": 0.0, "1": 0.95, "2": 0.05}
    # (3 * 0.95 - 1) / 2 = 1.85 / 2 = 0.925
    assert score.confidence == pytest.approx(0.925, abs=0.01)

    # Usage check
    assert resp.usage.input_tokens == 307 + 318 + 304
    assert resp.usage.output_tokens == 0
