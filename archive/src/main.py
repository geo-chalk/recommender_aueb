"""
Main module of the library
"""
import ast

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

from archive.src.helper_functions.restaurant_generator import generate_restaurants
from archive.src.helper_functions.users_generator import (
    generate_users_segment1, generate_ratings_segment1,
    generate_users_segment2, generate_ratings_segment2
)


def convert_string_to_list(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def main():
    restaurants = generate_restaurants(rest_num=100)
    users_segment1 = generate_users_segment1(user_num=1000)
    generate_ratings_segment1(users_segment1, restaurants)

    users_segment2 = generate_users_segment2(user_num=2000)
    generate_ratings_segment2(users_segment2, restaurants)

    file_paths = ['segment1.csv', 'segment2.csv']  # Add as many paths as needed

    # Read each CSV file into a DataFrame and store them in a list
    dfs = [pd.read_csv(file_path) for file_path in file_paths]

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Shuffle the rows of the combined DataFrame
    df = combined_df.sample(frac=1).reset_index(drop=True)

    # shuffled_df now contains all your data, shuffled
    print(df.shape)

    # Function to convert string representation of lists to actual lists

    # Convert string representations of lists to actual lists for all list-type columns
    list_columns = ['favorite_cuisine', 'restaurant_cuisine', 'payment_methods']
    for column in list_columns:
        convert_string_to_list(df, column)

    # One-hot encode each list-type column
    for column in list_columns:
        mlb = MultiLabelBinarizer()
        expanded = mlb.fit_transform(df[column])
        encoded_df = pd.DataFrame(expanded, columns=[f"{column}_{cls}" for cls in mlb.classes_])
        df = df.join(encoded_df)

    # Drop the original list-type columns if they are no longer needed
    df.drop(list_columns, axis=1, inplace=True)

    # Split the data into features and target
    X = df.drop('rating', axis=1)
    y = df['rating']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical features in the dataset
    categorical_features_indices = np.where(X.dtypes != np.float64)[0]

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=3,
        eval_metric='Accuracy',
        use_best_model=True,
        random_seed=42,
        auto_class_weights='Balanced',  # Helps if the target classes are imbalanced
        verbose=False
    )

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
