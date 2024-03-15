"""
Host variables used by more than 1 function. Easy to change.
"""
from typing import Tuple, List
from pathlib import Path

# General
CUISINES: Tuple = ('Pizza', 'Burger', 'Pasta', 'Souvlaki', 'Sushi', 'Chinese')
TARGET_VARIABLE: str = 'rating'

# Paths
RAW_DATA_DIR: Path = Path("data") / "raw"
PROCESSED_DATA_DIR: Path = Path("data") / "processed"
MODEL_REGISTRY_DIR: Path = Path("data") / "model_registry"
SEGMENTS_FILE_NAME: str = "segments.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
BEST_MODEL_FILE_NAME: str = "best_model.pkl"

# Create Paths if needed
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# restaurant
RESTAURANT = dict(
    cuisines=CUISINES,
    max_cuisines=3,
    price_samples=(0.5, 0.3, 0.15, 0.05),
    has_extra_del_cost_prob=0.2,
    min_cost=(6, 3),
    avg_rating=(3.7, 1),
    avg_del_time=(30, 7),
    payment_methods=(["CASH", "CARD"], ['COUPON']),
    rest_num=100)

# segments
SEGMENT_1 = dict(
    segment_id="segment_1",
    user_num=1000,
    postcode_num=50,
    usr_age=(60, 7),
    usr_cuisines=CUISINES,
    usr_cuisines_num=(1, 1)
)

SEGMENT_2 = dict(
    segment_id="segment_2",
    user_num=2000,
    postcode_num=50,
    usr_age=(20, 7),
    usr_cuisines=CUISINES,
    usr_cuisines_num=(1, 6)
)

# Model params
MODEL_OPTIMIZE_PARAMS = dict(
    n_iter=10,
    n_jobs=-1,
    cv=5,
    verbose=2
)

MODEL_SEARCH_PARAMS: dict = {
    'iterations': [500, 1000],
    'depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.5, 0.1]
}
