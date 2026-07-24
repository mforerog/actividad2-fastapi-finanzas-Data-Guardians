from fastapi.testclient import TestClient

from financial_api.api import app


client = TestClient(app)


def test_market_data_valid_symbol() -> None:
    response = client.get("/market-data/AAPL?limit=3")

    assert response.status_code == 200

    body = response.json()

    assert body["symbol"] == "AAPL"
    assert len(body["records"]) == 3


def test_market_data_invalid_symbol() -> None:
    response = client.get("/market-data/INVALID")

    assert response.status_code == 404