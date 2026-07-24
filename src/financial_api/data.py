"""Descarga y almacenamiento local de datos financieros."""

from pathlib import Path

import pandas as pd
import yfinance as yf


SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "market_data.csv"


def download_market_data(
    symbols: list[str] | None = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Descarga datos históricos de varios activos desde Yahoo Finance."""

    selected_symbols = symbols or SYMBOLS
    frames: list[pd.DataFrame] = []

    for symbol in selected_symbols:
        print(f"Descargando datos de {symbol}...")

        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            raise ValueError(f"No se obtuvieron datos para {symbol}.")

        data = data.reset_index()

        # Algunas versiones de yfinance generan columnas multinivel.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                column[0] if isinstance(column, tuple) else column
                for column in data.columns
            ]

        data["Symbol"] = symbol
        frames.append(data)

    market_data = pd.concat(frames, ignore_index=True)

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Symbol",
    ]

    available_columns = [
        column for column in required_columns
        if column in market_data.columns
    ]

    return market_data[available_columns]


def save_market_data(data: pd.DataFrame) -> Path:
    """Guarda los datos en un archivo CSV local."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_DATA_FILE, index=False)

    print(f"Datos guardados en: {RAW_DATA_FILE}")
    print(f"Registros guardados: {len(data)}")

    return RAW_DATA_FILE


def load_local_market_data() -> pd.DataFrame:
    """Carga los datos financieros guardados localmente."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            "No existe el archivo local de datos. "
            "Ejecute primero el módulo financial_api.data."
        )

    return pd.read_csv(RAW_DATA_FILE, parse_dates=["Date"])


def main() -> None:
    """Ejecuta la descarga y el almacenamiento local."""

    market_data = download_market_data()
    save_market_data(market_data)

    print("\nRegistros por activo:")
    print(market_data.groupby("Symbol").size())


if __name__ == "__main__":
    main()