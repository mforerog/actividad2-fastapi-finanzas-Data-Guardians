from fastapi.testclient import TestClient

from financial_api.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["service"] == "financial-api"
    assert body["model_available"] is True