from typing import List
import pandas as pd

from recommender_app.generators import Restaurant, User
from recommender_app.generators.restaurant_generator import generate_restaurants
from recommender_app.generators.users_generator import (
    RatingsGenerator
)
from recommender_app.utils.variables import (
    SEGMENT_1, SEGMENT_2,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    SEGMENTS_FILE_NAME, RESTAURANT
)
from recommender_app.preprocessing.process import DataPreProcessor


def generate_segments(restaurant_dict: dict,
                      segment_list: List[dict]
                      ) -> pd.DataFrame:
    """
    Generates restaurant data, applies user preference segments, processes the data and returns it as a DataFrame.

    * Generates restaurants using predefined parameters.
    * Applies defined user preference segments (SEGMENT_1 and SEGMENT_2) to the data.
    * Defines categorical columns from User and Restaurant classes.
    * Processes the segmented data into a suitable format for machine learning models.
    * Saves the processed data to a CSV file.

    Returns:
        A DataFrame containing the processed segmented data.
    """
    # generate restaurants
    restaurants = generate_restaurants(**restaurant_dict)

    # generate segments
    file_paths = []
    for segment in segment_list:
        segment_path = RatingsGenerator(output_dir=RAW_DATA_DIR).generate_segment(restaurants=restaurants,
                                                                                  **segment)
        file_paths.append(segment_path)

    # Define categorical columns
    categorical_columns: List[str] = User.get_categorical_cols() + Restaurant.get_categorical_cols()

    # Create df
    df: pd.DataFrame = DataPreProcessor(file_paths).process(categorical_columns)
    # df.to_csv(PROCESSED_DATA_DIR / SEGMENTS_FILE_NAME, index=False)

    return df
