"""Carga del modelo y generación de predicciones."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from financial_api.features import (
    FEATURE_COLUMNS,
    load_processed_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "model.joblib"
METADATA_FILE = ARTIFACTS_DIR / "model_metadata.json"

DISCLAIMER = (
    "Predicción generada únicamente con fines educativos. "
    "No constituye asesoría financiera."
)


def load_model() -> RandomForestClassifier:
    """Carga el modelo serializado desde disco."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "No se encontró artifacts/model.joblib. "
            "Ejecute primero el entrenamiento."
        )

    model = joblib.load(MODEL_FILE)
    return model


def load_model_metadata() -> dict:
    """Carga los metadatos almacenados en JSON."""

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "No se encontró artifacts/model_metadata.json."
        )

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_latest_features(symbol: str) -> pd.DataFrame:
    """Obtiene la fila más reciente de variables para un activo."""

    processed_data = load_processed_data()

    symbol_data = processed_data[
        processed_data["Symbol"].str.upper() == symbol.upper()
    ].copy()

    if symbol_data.empty:
        raise ValueError(
            f"No existen datos procesados para el símbolo {symbol}."
        )

    symbol_data["Date"] = pd.to_datetime(symbol_data["Date"])
    latest_row = symbol_data.sort_values("Date").iloc[-1]

    return latest_row[FEATURE_COLUMNS].to_frame().T


def predict_symbol(symbol: str) -> dict[str, str | float]:
    """Genera la predicción del siguiente día."""

    model = load_model()
    metadata = load_model_metadata()
    latest_features = get_latest_features(symbol)

    predicted_class = int(model.predict(latest_features)[0])
    probabilities = model.predict_proba(latest_features)[0]

    class_positions = {
        int(class_value): index
        for index, class_value in enumerate(model.classes_)
    }

    probability_up = float(
        probabilities[class_positions[1]]
    )

    return {
        "symbol": symbol.upper(),
        "prediction": "up" if predicted_class == 1 else "down",
        "probability_up": round(probability_up, 4),
        "model_version": metadata["model_version"],
        "prediction_horizon": "next_day",
        "disclaimer": DISCLAIMER,
    }