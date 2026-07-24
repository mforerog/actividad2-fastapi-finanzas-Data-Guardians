from fastapi import FastAPI

app = FastAPI(
    title="API Financiera Educativa - Data Guardians",
    description=(
        "API académica para consultar datos financieros y generar "
        "predicciones educativas. No constituye asesoría financiera."
    ),
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Devuelve un mensaje de bienvenida."""
    return {
        "message": "API Financiera Educativa - Data Guardians"
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Verifica que la API esté disponible."""
    return {
        "status": "ok",
        "service": "financial-api"
    }