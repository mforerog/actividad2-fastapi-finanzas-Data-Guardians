"""Aplicación principal de la API financiera educativa."""

from fastapi import FastAPI, HTTPException, Query

from financial_api.features import load_processed_data
from financial_api.predict import (
    MODEL_FILE,
    METADATA_FILE,
    load_model_metadata,
    predict_symbol,
)
from financial_api.schemas import (
    ALLOWED_SYMBOLS,
    HealthResponse,
    MarketDataRecord,
    MarketDataResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(
    title="API Financiera Educativa - Data Guardians",
    description=(
        "API académica para consultar variables financieras y generar "
        "predicciones educativas de tendencia. "
        "No constituye asesoría financiera ni recomendación de inversión."
    ),
    version="0.2.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Devuelve información general del servicio."""

    return {
        "message": "API Financiera Educativa - Data Guardians",
        "documentation": "/docs",
        "disclaimer": "No constituye asesoría financiera.",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """Verifica que la API y los artefactos del modelo estén disponibles."""

    model_available = MODEL_FILE.exists() and METADATA_FILE.exists()

    return HealthResponse(
        status="ok" if model_available else "degraded",
        service="financial-api",
        model_available=model_available,
    )


@app.get(
    "/market-data/{symbol}",
    response_model=MarketDataResponse,
)
def get_market_data(
    symbol: str,
    limit: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Número de registros recientes que se desean consultar.",
    ),
) -> MarketDataResponse:
    """Devuelve variables financieras procesadas de un activo."""

    normalized_symbol = symbol.strip().upper()

    if normalized_symbol not in ALLOWED_SYMBOLS:
        allowed = ", ".join(sorted(ALLOWED_SYMBOLS))
        raise HTTPException(
            status_code=404,
            detail=f"Símbolo no disponible. Use uno de estos: {allowed}.",
        )

    data = load_processed_data()

    symbol_data = data[
        data["Symbol"].str.upper() == normalized_symbol
    ].copy()

    if symbol_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No existen datos para {normalized_symbol}.",
        )

    symbol_data["Date"] = symbol_data["Date"].astype(str)
    recent_data = symbol_data.sort_values("Date").tail(limit)

    records = [
        MarketDataRecord(
            date=row["Date"],
            symbol=row["Symbol"],
            close=float(row["Close"]),
            return_1d=float(row["return_1d"]),
            return_lag_1=float(row["return_lag_1"]),
            return_lag_2=float(row["return_lag_2"]),
            sma_5_ratio=float(row["sma_5_ratio"]),
            sma_10_ratio=float(row["sma_10_ratio"]),
            volatility_5=float(row["volatility_5"]),
            volume_change=float(row["volume_change"]),
        )
        for _, row in recent_data.iterrows()
    ]

    return MarketDataResponse(
        symbol=normalized_symbol,
        records=records,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Genera una predicción para el activo solicitado."""

    if not request.use_cached_data:
        raise HTTPException(
            status_code=400,
            detail=(
                "Esta versión académica utiliza datos locales para garantizar "
                "la reproducibilidad."
            ),
        )

    try:
        result = predict_symbol(request.symbol)
        return PredictionResponse(**result)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@app.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
)
def get_model_metadata() -> ModelMetadataResponse:
    """Devuelve los metadatos principales del modelo."""

    try:
        metadata = load_model_metadata()
        return ModelMetadataResponse(**metadata)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error