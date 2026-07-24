# syntax=docker/dockerfile:1

# ============================================================
# ETAPA 1: construcción del entorno con Poetry
# ============================================================

FROM python:3.12-slim AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Instala Poetry únicamente para construir el entorno.
RUN pip install --no-cache-dir "poetry>=2.0,<3.0"

# Copiamos primero las dependencias para aprovechar la caché de Docker.
COPY pyproject.toml poetry.lock ./

# Instala solamente las dependencias necesarias para ejecutar la aplicación.
RUN poetry install --only main --no-root \
    && rm -rf "$POETRY_CACHE_DIR"


# ============================================================
# ETAPA 2: imagen final de ejecución
# ============================================================

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app

# Crea un usuario sin privilegios para ejecutar la API.
RUN addgroup --system app \
    && adduser --system --ingroup app app

# Copia el entorno virtual creado en la etapa anterior.
COPY --from=builder /app/.venv /app/.venv

# Copia el código fuente, los datos locales y el modelo entrenado.
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data
COPY --chown=app:app artifacts ./artifacts

USER app

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn financial_api.api:app --host 0.0.0.0 --port ${PORT}"]