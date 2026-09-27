import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock
from jevk5 import JevK5GGUF

from jevk5_server.app import create_app
from jevk5_server.adapter import JevK5TypeSafeAdapter
from jevk5_server.config import Settings


@pytest.fixture
def test_app():
    settings = Settings(
        api_key=None,
        llama_server_url="http://mock-llama",
        model_name="jevk5-4b-v0.3-Q8_0",
    )
    mock_model = MagicMock(spec=JevK5GGUF)

    def mock_eval_q(state, question):
        if question["type"] == "noul":
            return {"true": 0.95, "false": 0.05}, 296
        elif question["type"] == "choice":
            return {"billing": 0.88, "technical": 0.12, "sales": 0.0}, 318
        elif question["type"] == "score":
            return {"0": 0.0, "1": 0.95, "2": 0.05}, 304
        return {}, 0

    mock_model.probabilities.side_effect = mock_eval_q
    adapter = JevK5TypeSafeAdapter(model=mock_model, default_model_name=settings.model_name)
    adapter.check_health = MagicMock(return_value=True)
    app = create_app(settings=settings, adapter=adapter)
    return app


@pytest.mark.anyio
async def test_api_health(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["backend_llama_connected"] is True


@pytest.mark.anyio
async def test_api_models(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        resp = await ac.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        names = [m["name"] for m in data["models"]]
        assert "jev-latest" in names
        assert "jevk5-4b-v0.3-Q8_0" in names


@pytest.mark.anyio
async def test_api_systemone_noul(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        payload = {
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Does this convey urgency?",
                }
            },
        }
        resp = await ac.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "jev-latest"
        assert "is_urgent" in data["answers"]
        answer = data["answers"]["is_urgent"]
        assert answer["type"] == "noul"
        assert answer["noul"] == 0.95
        assert "confidence" not in answer
        assert data["usage"]["input_tokens"] == 296


@pytest.mark.anyio
async def test_api_systemone_choice(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        payload = {
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments, invoicing, refunds",
                        "technical": "Bugs, outages, integrations",
                        "sales": "Pricing, upgrades, new accounts",
                    },
                }
            },
        }
        resp = await ac.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answers"]["department"]
        assert answer["type"] == "choice"
        assert answer["choice"] == "billing"
        assert answer["probabilities"]["billing"] == 0.88
        assert "confidence" in answer


@pytest.mark.anyio
async def test_api_systemone_score(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        payload = {
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"],
                }
            },
        }
        resp = await ac.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answers"]["frustration"]
        assert answer["type"] == "score"
        assert answer["score"] == 1.05
        assert answer["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
        assert answer["probabilities"]["1"] == 0.95
        assert "confidence" in answer


@pytest.mark.anyio
async def test_api_auth_required():
    settings = Settings(api_key="secret-token-123")
    mock_model = MagicMock(spec=JevK5GGUF)
    adapter = MagicMock(spec=JevK5TypeSafeAdapter)
    adapter.evaluate = MagicMock()

    async def fake_eval(req):
        return {
            "model": "jev-latest",
            "answers": {},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    adapter.evaluate.side_effect = fake_eval
    app = create_app(settings=settings, adapter=adapter)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Request without token -> 401
        resp = await ac.post("/v1/systemone", json={"state": "hi", "questions": {}})
        assert resp.status_code == 401

        # Request with wrong token -> 401
        resp = await ac.post(
            "/v1/systemone",
            json={"state": "hi", "questions": {}},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

        # Request with correct token -> passes auth (validation of body happens)
        resp = await ac.post(
            "/v1/systemone",
            json={"state": "hi", "questions": {}},
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert resp.status_code != 401


@pytest.mark.anyio
async def test_api_validation_error(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        # Missing state and questions -> 422
        resp = await ac.post("/v1/systemone", json={})
        assert resp.status_code == 422
