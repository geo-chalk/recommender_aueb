import io
import logging
from datetime import datetime

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File
from mlflow import MlflowClient
from starlette.responses import RedirectResponse

# Import project-specific variables
from recommender_app.utils.variables import (
    REGISTERED_MODEL_NAME,
    ALIAS,
    TARGET_VARIABLE,
    MLFLOW_TRACKING_URI
)

# Ensure logs are visible in the terminal
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Restaurant Recommender CSV API")

# Global container to keep the model in memory
MODEL_HOLDER = {"model": None}


@app.on_event("startup")
def load_champion_model():
    """
    Startup logic mirroring 'skip-training' in main.py.
    Loads the Champion model once so it is ready for fast inference.
    """
    try:
        # Resolve the model URI using the 'champion' alias
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}"
        logger.info(f"Attempting to load model: {model_uri}")

        # Load the model directly from MLflow
        MODEL_HOLDER["model"] = mlflow.sklearn.load_model(model_uri)
        logger.info("Successfully loaded Champion model.")
    except Exception as e:
        logger.error(f"FATAL: Could not load model from MLflow: {e}")


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """
    Comprehensive health check returning app status and champion model version.
    """
    client = MlflowClient()
    health_status = "UP"
    model_version = "unknown"

    try:
        # Resolve the actual version number for the 'champion' alias
        model_metadata = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)
        model_version = model_metadata.version

        # Check if the local model is actually loaded
        if MODEL_HOLDER["model"] is None:
            health_status = "degraded (model not loaded)"

    except Exception as e:
        logger.error(f"Health check failed to fetch model metadata: {e}")
        health_status = "unhealthy (registry unreachable)"

    return {
        "status": health_status,
        "timestamp": datetime.utcnow().isoformat(),
        "model_details": {
            "name": REGISTERED_MODEL_NAME,
            "alias": ALIAS,
            "version": model_version
        },
        "app_version": "1.0.0"
    }

@app.post("/api/v1/predict-parquet")
async def predict_from_parquet(file: UploadFile = File(...)):
    """
    Accepts a parquet file (like api_input.parquet), converts it to a DataFrame,
    and returns predictions for every row.
    """
    if MODEL_HOLDER["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    if not file.filename.endswith(".parquet"):
        raise HTTPException(status_code=400, detail="Please upload a .parquet file.")

    try:
        # Read the file, convert to dataframe and predict
        contents = await file.read()
        df = pd.read_parquet(io.BytesIO(contents))
        X = df.drop(columns=[TARGET_VARIABLE], errors='ignore')
        predictions = MODEL_HOLDER["model"].predict(X)

        return {"status": "success", "predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"Parquet processing error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
