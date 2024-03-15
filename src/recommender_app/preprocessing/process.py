import ast
from typing import List
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from recommender_app.generators import Restaurant, User


class DataPreProcessor:

    def __init__(self, file_paths: List[Path]):
        self.df = self._create_df_from_paths(file_paths)

    @staticmethod
    def _create_df_from_paths(file_paths: List[Path]) -> pd.DataFrame:
        """
        Reads a list of CSV files and concatenates them into a single DataFrame.
        The rows of the resulting DataFrame are shuffled randomly.

        Args:
            file_paths: A list of paths to CSV files.

        Returns:
            pd.DataFrame: A shuffled DataFrame containing all the data from the input CSV files.
        """
        # Read each CSV file into a DataFrame and store them in a list
        dfs: List[pd.DataFrame] = [pd.read_csv(file_path) for file_path in file_paths]

        # Concatenate all DataFrames into a single DataFrame
        combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

        # Shuffle the rows of the combined DataFrame
        df: pd.DataFrame = combined_df.sample(frac=1).reset_index(drop=True)

        # shuffled_df now contains all your data, shuffled
        print(df.shape)

        return df

    def _convert_string_to_list(self, column_name):
        """
        Converts a column of string representations of lists to actual lists.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be converted.
            column_name (str): The name of the column to be converted.

        Returns:
            pd.DataFrame: The input DataFrame with the specified column converted to actual lists.

        Examples:
            >>> df = pd.DataFrame({'A': ['[1, 2, 3]', '[4, 5]', '[]', '[1]']})
            >>> df = convert_string_to_list(df, 'A')
            >>> df['A'].tolist()
            [[1, 2, 3], [4, 5], [], [1]]]
        """
        self.df[column_name] = self.df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def process(self, categorical_columns: List[str]):
        """
        Processes the data frame by converting string representations of lists to actual lists,
        one-hot encoding each list-type column.
        """
        # Convert string representations of lists to actual lists for all list-type columns
        for column in categorical_columns:
            self._convert_string_to_list(column)

        # One-hot encode each list-type column
        for column in categorical_columns:
            mlb = MultiLabelBinarizer()
            expanded = mlb.fit_transform(self.df[column])
            encoded_df = pd.DataFrame(expanded, columns=[f"{column}_{cls}" for cls in mlb.classes_])
            self.df = self.df.join(encoded_df)

        # Drop the original list-type columns if they are no longer needed
        self.df.drop(categorical_columns, axis=1, inplace=True)

        # Split the data into training and testing sets
        return self.df
