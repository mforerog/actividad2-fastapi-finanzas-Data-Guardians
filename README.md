# API Financiera Educativa — Data Guardians

API financiera desarrollada con FastAPI para consultar datos históricos de activos, construir variables financieras y generar una predicción educativa sobre la tendencia del retorno del siguiente día.

El proyecto fue desarrollado como parte de la Actividad Integradora 2 del módulo de MLOps en Python.

> *Advertencia:* este proyecto tiene fines exclusivamente académicos y educativos. No constituye asesoría financiera ni una recomendación de compra o venta de activos.

## Integrantes

- Martha Forero
- Maria Beltran

La descripción detallada de los roles y responsabilidades se encuentra en [TEAM.md](TEAM.md).

## Objetivo

Implementar un flujo reproducible de Machine Learning que permita:

1. Descargar datos históricos con yfinance.
2. Guardar una copia local para garantizar reproducibilidad.
3. Construir variables financieras.
4. Entrenar un modelo predictivo.
5. Serializar el modelo y guardar sus metadatos.
6. Exponer predicciones mediante FastAPI.
7. Validar los endpoints con pruebas automatizadas.
8. Ejecutar la aplicación localmente y mediante Docker.

## Tarea predictiva

El modelo busca predecir si el retorno del siguiente día será:

- up: retorno positivo.
- down: retorno no positivo.

Se implementó un modelo RandomForestClassifier.

## Activos incluidos

El proyecto utiliza datos históricos de:

- AAPL: Apple Inc.
- MSFT: Microsoft Corporation.
- GOOGL: Alphabet Inc.

## Estructura del proyecto

```text
actividad2-fastapi-finanzas-Data-Guardians/
├── artifacts/
│   ├── model.joblib
│   └── model_metadata.json
├── data/
│   ├── raw/
│   │   └── market_data.csv
│   └── processed/
│       └── features.csv
├── reports/
├── src/
│   └── financial_api/
│       ├── _init_.py
│       ├── api.py
│       ├── data.py
│       ├── features.py
│       ├── predict.py
│       ├── schemas.py
│       └── train.py
├── tests/
│   ├── test_health.py
│   ├── test_market_data.py
│   └── test_predict.py
├── .dockerignore
├── .gitignore
├── compose.yaml
├── Dockerfile
├── poetry.lock
├── pyproject.toml
├── README.md
└── TEAM.md
