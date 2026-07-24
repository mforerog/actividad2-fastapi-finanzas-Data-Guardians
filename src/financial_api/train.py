"""Entrenamiento y almacenamiento del modelo predictivo."""

from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from financial_api.features import FEATURE_COLUMNS, load_processed_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "model.joblib"
METADATA_FILE = ARTIFACTS_DIR / "model_metadata.json"

MODEL_VERSION = "random_forest_v1"
RANDOM_STATE = 42


def prepare_training_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y variable objetivo."""

    missing_columns = set(FEATURE_COLUMNS + ["target"]).difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas para entrenamiento: {sorted(missing_columns)}"
        )

    features = data[FEATURE_COLUMNS].copy()
    target = data["target"].astype(int)

    return features, target


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[RandomForestClassifier, dict[str, float | int]]:
    """Entrena y evalúa un modelo Random Forest."""

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics: dict[str, float | int] = {
        "accuracy": round(float(accuracy), 4),
        "training_records": int(len(x_train)),
        "test_records": int(len(x_test)),
    }

    print("\nResultados del modelo:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, predictions))

    return model, metrics


def save_artifacts(
    model: RandomForestClassifier,
    metrics: dict[str, float | int],
    data: pd.DataFrame,
) -> None:
    """Guarda el modelo entrenado y sus metadatos."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    metadata = {
        "model_name": "RandomForestClassifier",
        "model_version": MODEL_VERSION,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": sorted(data["Symbol"].unique().tolist()),
        "feature_columns": FEATURE_COLUMNS,
        "target_definition": (
            "1 si el retorno del siguiente día es positivo; 0 en caso contrario"
        ),
        "prediction_horizon": "next_day",
        "main_metric": "accuracy",
        "metrics": metrics,
        "dataset_records": int(len(data)),
        "random_state": RANDOM_STATE,
        "disclaimer": (
            "Modelo desarrollado únicamente con fines educativos. "
            "No constituye asesoría financiera."
        ),
    }

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado en: {MODEL_FILE}")
    print(f"Metadatos guardados en: {METADATA_FILE}")


def main() -> None:
    """Ejecuta el entrenamiento completo."""

    processed_data = load_processed_data()
    features, target = prepare_training_data(processed_data)
    model, metrics = train_model(features, target)
    save_artifacts(model, metrics, processed_data)


if __name__ == "__main__":
    main()