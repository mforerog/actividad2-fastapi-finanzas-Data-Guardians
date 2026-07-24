"""Contratos de entrada y salida de la API financiera."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ALLOWED_SYMBOLS = {"AAPL", "MSFT", "GOOGL"}


class PredictionRequest(BaseModel):
    """Datos necesarios para solicitar una predicción."""

    symbol: str = Field(
        ...,
        description="Símbolo financiero que se desea analizar.",
        examples=["AAPL"],
    )
    prediction_horizon: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Horizonte de predicción en días. El modelo predice el día siguiente.",
    )
    use_cached_data: bool = Field(
        default=True,
        description="Indica si se utilizarán las variables guardadas localmente.",
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        """Normaliza y valida el símbolo financiero."""

        normalized = value.strip().upper()

        if normalized not in ALLOWED_SYMBOLS:
            allowed = ", ".join(sorted(ALLOWED_SYMBOLS))
            raise ValueError(
                f"Símbolo no permitido. Los símbolos disponibles son: {allowed}."
            )

        return normalized


class PredictionResponse(BaseModel):
    """Respuesta generada por el modelo."""

    symbol: str
    prediction: Literal["up", "down"]
    probability_up: float = Field(ge=0.0, le=1.0)
    model_version: str
    prediction_horizon: Literal["next_day"]
    disclaimer: str


class HealthResponse(BaseModel):
    """Respuesta del endpoint de disponibilidad."""

    status: Literal["ok", "degraded"]
    service: str
    model_available: bool


class MarketDataRecord(BaseModel):
    """Último registro de variables financieras de un activo."""

    date: str
    symbol: str
    close: float
    return_1d: float
    return_lag_1: float
    return_lag_2: float
    sma_5_ratio: float
    sma_10_ratio: float
    volatility_5: float
    volume_change: float


class MarketDataResponse(BaseModel):
    """Respuesta con datos procesados recientes."""

    symbol: str
    records: list[MarketDataRecord]


class ModelMetadataResponse(BaseModel):
    """Metadatos principales del modelo."""

    model_name: str
    model_version: str
    training_date_utc: str
    symbols: list[str]
    feature_columns: list[str]
    prediction_horizon: str
    main_metric: str
    metrics: dict[str, float | int]
    dataset_records: int
    disclaimer: str