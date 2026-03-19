import io
import logging
from datetime import datetime
from typing import Tuple, Union

import mlflow.sklearn
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from mlflow import MlflowClient
from pydantic import BaseModel
from starlette.responses import JSONResponse, RedirectResponse

from recommender_app.generators import Restaurant
from recommender_app.generators.restaurant_generator import generate_restaurants
# Import project-specific variables
from recommender_app.utils.variables import (
    REGISTERED_MODEL_NAME,
    TARGET_VARIABLE,
    MLFLOW_TRACKING_URI
)
import asyncio
from contextlib import asynccontextmanager

# Ensure logs are visible in the terminal
logger = logging.getLogger("uvicorn.error")


# Global container to keep the model in memory
MODEL_HOLDER = {"model": None, "metadata": None}
ALIAS: str = "latest"


def _load_model_by_alias(registered_model_name: str = REGISTERED_MODEL_NAME,
                         alias: str = ALIAS):
    """
    Startup logic mirroring 'skip-training' in main.py.
    Loads the Champion model once so it is ready for fast inference.
    """
    try:
        # Resolve the model URI using the 'champion' alias
        model_uri = f"models:/{registered_model_name}@{alias}"
        logger.info(f"Attempting to load model: {model_uri}")

        # Load the model directly from MLflow
        MODEL_HOLDER["model"] = mlflow.sklearn.load_model(model_uri)

        # Resolve the actual version number for the 'champion' alias
        model_metadata = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
        model_version = model_metadata.version

        MODEL_HOLDER["metadata"] = {
            "name": REGISTERED_MODEL_NAME,
            "alias": alias,
            "version": model_version
        }

        logger.info("Successfully loaded Champion model.")
    except Exception as e:
        logger.error(f"FATAL: Could not load model from MLflow: {e}")

async def model_refresher():
    """Worker that triggers the reload every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        _load_model_by_alias(REGISTERED_MODEL_NAME, ALIAS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model immediately
    _load_model_by_alias(REGISTERED_MODEL_NAME, ALIAS)
    # Start the background task
    refresh_task = asyncio.create_task(model_refresher())
    yield
    # Shutdown: Clean up task
    refresh_task.cancel()

app = FastAPI(title="Restaurant Recommender CSV API", lifespan=lifespan)

# Initialize mlflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()




@app.on_event("startup")
def load_model():
    """Initial load of the champion model."""
    _load_model_by_alias(REGISTERED_MODEL_NAME, ALIAS)


@app.post("/api/v1/switch-model")
async def switch_model(model_name: str = REGISTERED_MODEL_NAME, alias: str = ALIAS):
    """Dynamically switch the active model."""
    global ALIAS
    _load_model_by_alias(model_name, alias)
    if not MODEL_HOLDER["model"]:
        raise HTTPException(status_code=404, detail="Model not found")
    ALIAS = alias
    return {"message": "Model switched successfully", "details": MODEL_HOLDER["metadata"]}


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """
    Comprehensive health check returning app status and champion model version.
    """
    health_status = "UP"

    try:
        # Check if the local model is actually loaded
        if MODEL_HOLDER["model"] is None:
            health_status = "degraded (model not loaded)"

    except Exception as e:
        logger.error(f"Health check failed to fetch model metadata: {e}")
        health_status = "unhealthy (registry unreachable)"

    return {
        "status": health_status,
        "timestamp": datetime.utcnow().isoformat(),
        "model_details": MODEL_HOLDER["metadata"],
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


from typing import List, Dict, Any


# ... existing imports ...

@app.post("/api/v1/predict")
async def predict_json(data: List[Dict[str, Any]]):
    """
    Accepts a JSON list of records to simulate real-time or small-batch inference.
    Each dictionary in the list should match the columns in test.csv.
    """
    if MODEL_HOLDER["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    try:
        df = pd.DataFrame(data)
        X = df.drop(columns=[TARGET_VARIABLE], errors='ignore')
        predictions = MODEL_HOLDER["model"].predict(X)

        return {
            "status": "success",
            "model_used": MODEL_HOLDER["metadata"]["name"],
            "version": MODEL_HOLDER["metadata"]["version"],
            "predictions": predictions.tolist()
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Define configuration model for restaurant generator
class RestaurantConfig(BaseModel):
    cuisines: List[str] = ['Pizza', 'Burger', 'Pasta', 'Souvlaki', 'Sushi', 'Chinese']
    max_cuisines: int = 3
    price_samples: List[float] = [0.5, 0.3, 0.15, 0.05]
    has_extra_del_cost_prob: float = 0.2
    min_cost: List[int] = [6, 3]
    avg_rating: Tuple[float, int] = [3.7, 1]
    avg_del_time: Tuple[int, int] = [30, 7]
    payment_methods: Tuple[List[str], List[str]] = [["CASH", "CARD"], ['COUPON']]
    rest_num: int = 100


# Initialize default config
restaurant_config = RestaurantConfig()

# Endpoint to generate restaurants
@app.post("/api/v1/generate_restaurants/",
          response_model=List[Restaurant])
async def restaurant_generator(config: RestaurantConfig) -> Union[List[Restaurant], JSONResponse]:
    """Generate restaurants based on an input configuration"""
    try:
        # generate and return restaurants
        restaurants: List[Restaurant] = generate_restaurants(**config.dict())
        logger.info(f"Generated {len(restaurants)} restaurants successfully.")
        return restaurants

    except Exception as e:
        logger.error(f"Error occurred while generating restaurants: {str(e)}")
        return JSONResponse({"status": "FAILURE", "message": f"Error message: {str(e)}"})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
