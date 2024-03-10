from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


class ModelTrainer(CatBoostClassifier):

    def __init__(self,
                 df: pd.DataFrame,
                 predicted_col: str = 'rating',
                 test_size: float = 0.2,
                 iterations=1000,
                 learning_rate=0.1,
                 depth=3,
                 eval_metric='Accuracy',
                 random_seed=42,
                 auto_class_weights='Balanced',
                 verbose=False):
        super().__init__()
        self.df = df
        self.predicted_col = predicted_col
        self.test_size = test_size
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            eval_metric=eval_metric,
            random_seed=random_seed,
            auto_class_weights=auto_class_weights,  # Helps if the target classes are imbalanced
            verbose=verbose,
            allow_writing_files=False)
        self.X_train, self.X_valid, self.y_train, self.y_valid = self.split()

    @property
    def categorical_features_indices(self):
        """Returns the input of the categorical indices"""
        return np.where(self.X_train.dtypes != np.float64)[0]

    @property
    def X(self) -> pd.DataFrame:
        """Defines the input data"""
        return self.df.drop(self.predicted_col, axis=1)

    @property
    def y(self) -> pd.DataFrame:
        """Defines the label"""
        return self.df[self.predicted_col]

    def split(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Splits the data into training and validation sets.

        Returns
                A tuple containing the training data, validation data, target values for the training data,
                and target values for the validation data.
        """
        # Split the data into features and target
        return train_test_split(self.X, self.y,
                                test_size=0.2,
                                random_state=42)

    def fit(self, **kwargs):
        """Trains the CatBoostClassifier model."""
        # Train the model
        self.model.fit(
            self.X_train, self.y_train,
            cat_features=self.categorical_features_indices,
            eval_set=(self.X_valid, self.y_valid),
            early_stopping_rounds=50  # Stops training if the validation metric is not improving
        )

    def predict(self, data: pd.DataFrame):
        """Makes predictions on the given data."""
        return self.model.predict(data)

    def optimize(self,
                 **kwargs):
        """
        Optimizes the CatBoostClassifier model using RandomizedSearchCV.
        Once the best model is found, the model is re-trained using the entire dataset

        Parameters
            **kwargs: Additional keyword arguments for the RandomizedSearchCV.
        """
        scorer = make_scorer(accuracy_score)
        clf_grid = RandomizedSearchCV(estimator=self.model,
                                      scoring=scorer,
                                      **kwargs)
        # Fit GridSearchCV
        clf_grid.fit(
            self.X_train, self.y_train,
            cat_features=self.categorical_features_indices,
            eval_set=(self.X_valid, self.y_valid),
            early_stopping_rounds=50  # Stops training if the validation metric is not improving
        )

        # re-train using all the data
        clf_grid.best_estimator_.fit(self.X, self.y,
                                     cat_features=self.categorical_features_indices)
        self.model = clf_grid.best_estimator_
