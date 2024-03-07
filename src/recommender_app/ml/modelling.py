from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV


class ModelTrainer(CatBoostClassifier):

    def __init__(self,
                 df: pd.DataFrame,
                 test_size: float = 0.2,
                 iterations=1000,
                 learning_rate=0.1,
                 depth=3,
                 eval_metric='Accuracy',
                 use_best_model=True,
                 random_seed=42,
                 auto_class_weights='Balanced',
                 verbose=False):
        super().__init__()
        self.df = df
        self.test_size = test_size
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            eval_metric=eval_metric,
            use_best_model=use_best_model,
            random_seed=random_seed,
            auto_class_weights=auto_class_weights,  # Helps if the target classes are imbalanced
            verbose=verbose)
        self.X_train, self.X_valid, self.y_train, self.y_valid = self.split()

    def split(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # Split the data into features and target
        X: pd.DataFrame = self.df.drop('rating', axis=1)
        y: pd.DataFrame = self.df['rating']
        return train_test_split(X, y,
                                test_size=0.2,
                                random_state=42)

    def fit(self, **kwargs):

        # Identify categorical features in the dataset
        categorical_features_indices = np.where(self.X_train.dtypes != np.float64)[0]

        # Train the model
        self.model.fit(
            self.X_train, self.y_train,
            cat_features=categorical_features_indices,
            eval_set=(self.X_valid, self.y_valid),
            early_stopping_rounds=50  # Stops training if the validation metric is not improving
        )

    def predict(self, **kwargs):
        # Make predictions on the test set
        predictions = self.model.predict(self.X_test)

        # Calculate the accuracy
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")

        return predictions

    def fit_predict(self):
        self.fit()
        return self.predict()
