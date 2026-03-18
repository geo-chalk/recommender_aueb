# Import necessary modules
import logging
from typing import List, Tuple, Union

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from starlette.responses import JSONResponse, RedirectResponse

# Import local modules
from recommender_app.generators.restaurant_generator import generate_restaurants
from recommender_app.generators import Restaurant

# Initialize FastAPI app
app = FastAPI(
    title="Restaurant Generator API",
    description="This is a simple API that generates a list of restaurants based on a given configuration.",
    version="1.0.0",
    docs_url="/docs",  # Default value
    redoc_url=None  # Disables ReDoc documentation
)

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rec_sys_app")


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


# redirect to docs
@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


# Health check endpoint
@app.get("/health")
async def health() -> JSONResponse:
    logger.info("Health check request received.")
    return JSONResponse({"status": "UP"})


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


# Main function to run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
