"""
Host variables used by more than 1 function. Easy to change.
"""
from typing import Tuple, List
from pathlib import Path

CUISINES: Tuple = ('Pizza', 'Burger', 'Pasta', 'Souvlaki', 'Sushi', 'Chinese')

# Paths
RAW_DATA_DIR: Path = Path("data") / "raw"
PROCESSED_DATA_DIR: Path = Path("data") / "processed"

# Create Paths if needed
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

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
