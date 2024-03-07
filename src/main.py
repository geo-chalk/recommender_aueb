"""
Main module of the library
"""
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
from catboost import CatBoostClassifier
import pickle
import os
print(os.getcwd())
from recommender_app.generators.restaurant_generator import generate_restaurants
from recommender_app.generators.users_generator import (
    RatingsGenerator
)
from recommender_app.utils.variables import (
    CUISINES, SEGMENT_1, SEGMENT_2,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)
from recommender_app.preprocessing.process import DataPreProcessor
from recommender_app.ml.modelling import ModelTrainer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


def main():
    try:
        df: pd.DataFrame = pd.read_csv(PROCESSED_DATA_DIR / "segments.csv")
        print("Loaded existing file.")
    except FileNotFoundError:
        print("File Not found. Generating user ratings.")
        # generate restaurants
        restaurants = generate_restaurants(rest_num=100, cuisines=CUISINES)

        # generate segments
        file_paths = []
        for segment in [SEGMENT_1, SEGMENT_2]:
            segment_path = RatingsGenerator(output_dir=RAW_DATA_DIR).generate_segment(restaurants=restaurants,
                                                                                      **segment)
            file_paths.append(segment_path)

        # Create df
        df: pd.DataFrame = DataPreProcessor(file_paths).process()
        df.to_csv(PROCESSED_DATA_DIR / "segments.csv")

    # Split the data into features and target
    X = df.drop('rating', axis=1)
    y = df['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(
        use_best_model=True,
        random_seed=42,
        auto_class_weights='Balanced',  # Helps if the target classes are imbalanced
        verbose=True,
        allow_writing_files=False
    )

    params = {'iterations': [500, 1000],
              'depth': [3, 4, 5, 6],
              'learning_rate': [0.01, 0.5, 0.1]
              }

    # Identify categorical features in the dataset
    categorical_features_indices = np.where(X_train.dtypes != np.float64)[0]
    scorer = make_scorer(accuracy_score)
    clf_grid = RandomizedSearchCV(estimator=model,
                                  param_distributions=params,
                                  n_iter=5,
                                  n_jobs=-1,
                                  scoring=scorer,
                                  cv=5,
                                  verbose=2)
    # Fit GridSearchCV
    clf_grid.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50  # Stops training if the validation metric is not improving
    )

    model = CatBoostClassifier(**clf_grid.best_params_)
    # Train the model
    model.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50  # Stops training if the validation metric is not improving
    )

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
