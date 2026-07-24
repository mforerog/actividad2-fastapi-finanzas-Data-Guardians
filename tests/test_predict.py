from fastapi.testclient import TestClient

from financial_api.api import app


client = TestClient(app)


def test_predict_valid_request() -> None:
    payload = {
        "symbol": "AAPL",
        "prediction_horizon": 1,
        "use_cached_data": True,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["symbol"] == "AAPL"
    assert body["prediction"] in {"up", "down"}
    assert 0.0 <= body["probability_up"] <= 1.0
    assert body["model_version"] == "random_forest_v1"


def test_predict_invalid_symbol() -> None:
    payload = {
        "symbol": "INVALID",
        "prediction_horizon": 1,
        "use_cached_data": True,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_without_cached_data() -> None:
    payload = {
        "symbol": "AAPL",
        "prediction_horizon": 1,
        "use_cached_data": False,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400