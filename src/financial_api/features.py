"""Construcción de variables financieras para entrenamiento y predicción."""

from pathlib import Path

import pandas as pd

from financial_api.data import load_local_market_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "features.csv"

FEATURE_COLUMNS = [
    "return_1d",
    "return_lag_1",
    "return_lag_2",
    "sma_5_ratio",
    "sma_10_ratio",
    "volatility_5",
    "volume_change",
]


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Crea variables financieras y la variable objetivo."""

    required_columns = {"Date", "Close", "Volume", "Symbol"}

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing_columns)}"
        )

    featured_frames: list[pd.DataFrame] = []

    for symbol, symbol_data in data.groupby("Symbol"):
        current = symbol_data.copy()

        current["Date"] = pd.to_datetime(current["Date"])
        current = current.sort_values("Date").reset_index(drop=True)

        # Retorno porcentual diario.
        current["return_1d"] = current["Close"].pct_change()

        # Retornos rezagados.
        current["return_lag_1"] = current["return_1d"].shift(1)
        current["return_lag_2"] = current["return_1d"].shift(2)

        # Medias móviles.
        current["sma_5"] = current["Close"].rolling(window=5).mean()
        current["sma_10"] = current["Close"].rolling(window=10).mean()

        # Relación entre el precio actual y las medias móviles.
        current["sma_5_ratio"] = current["Close"] / current["sma_5"]
        current["sma_10_ratio"] = current["Close"] / current["sma_10"]

        # Volatilidad móvil de cinco días.
        current["volatility_5"] = (
            current["return_1d"].rolling(window=5).std()
        )

        # Cambio porcentual del volumen.
        current["volume_change"] = current["Volume"].pct_change()

        # Retorno del día siguiente.
        current["future_return"] = current["return_1d"].shift(-1)

        # Objetivo: 1 si el retorno siguiente es positivo, 0 si no.
        current["target"] = (current["future_return"] > 0).astype(int)

        current["Symbol"] = symbol

        featured_frames.append(current)

    featured_data = pd.concat(featured_frames, ignore_index=True)

    columns_to_keep = [
        "Date",
        "Symbol",
        "Close",
        *FEATURE_COLUMNS,
        "future_return",
        "target",
    ]

    featured_data = featured_data[columns_to_keep]

    # Elimina filas incompletas generadas por rezagos y ventanas móviles.
    featured_data = featured_data.dropna().reset_index(drop=True)

    return featured_data


def save_processed_data(data: pd.DataFrame) -> Path:
    """Guarda las variables procesadas en un archivo local."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA_FILE, index=False)

    print(f"Datos procesados guardados en: {PROCESSED_DATA_FILE}")
    print(f"Registros procesados: {len(data)}")

    return PROCESSED_DATA_FILE


def load_processed_data() -> pd.DataFrame:
    """Carga el dataset procesado desde almacenamiento local."""

    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            "No existe el archivo de variables procesadas. "
            "Ejecute primero el módulo financial_api.features."
        )

    return pd.read_csv(
        PROCESSED_DATA_FILE,
        parse_dates=["Date"],
    )


def main() -> None:
    """Ejecuta la construcción y almacenamiento de variables."""

    raw_data = load_local_market_data()
    processed_data = create_features(raw_data)
    save_processed_data(processed_data)

    print("\nRegistros procesados por activo:")
    print(processed_data.groupby("Symbol").size())

    print("\nDistribución de la variable objetivo:")
    print(processed_data["target"].value_counts(normalize=True))


if __name__ == "__main__":
    main()