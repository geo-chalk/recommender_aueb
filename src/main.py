"""
Main module of the library
"""
from argparse import ArgumentParser, Namespace
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split


from recommender_app.generators.segment_generator import generate_segments
from recommender_app.utils.variables import (
    PROCESSED_DATA_DIR, MODEL_SEARCH_PARAMS,
    BEST_MODEL_FILE_NAME, TRAIN_FILE_NAME, TEST_FILE_NAME,
    MODEL_OPTIMIZE_PARAMS, MODEL_REGISTRY_DIR
)
from recommender_app.ml.modelling import ModelTrainer
from recommender_app.utils.input_args import RecSysArgumentParser, BaseArguments


def main(input_args: BaseArguments):
    # --- Load test data and model ---
    if input_args.skip_training:

        try:
            test: pd.DataFrame = pd.read_csv(PROCESSED_DATA_DIR / TEST_FILE_NAME)
            print("Test data loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(f"`skip-training` is True but `{TEST_FILE_NAME}` is missing. "
                                    f"Please run again the process without the `skip-training` flag.")
        try:
            model: ModelTrainer = pickle.load((MODEL_REGISTRY_DIR / BEST_MODEL_FILE_NAME).open(mode='rb'))
            print("Best Model loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(f"`skip-training` is True but `{BEST_MODEL_FILE_NAME}` is missing. "
                                    f"Please run again the process without the `skip-training` flag.")

    # Define segments
    else:
        df: pd.DataFrame = generate_segments()

        # Split the data into features and target
        train, test = train_test_split(df,
                                       test_size=0.2,
                                       random_state=42)
        train.to_csv(PROCESSED_DATA_DIR / TRAIN_FILE_NAME, index=False)
        test.to_csv(PROCESSED_DATA_DIR / TEST_FILE_NAME, index=False)

        # Optimize model
        model = ModelTrainer(df=train,
                             predicted_col='rating',
                             test_size=0.2,
                             random_seed=42,
                             auto_class_weights='Balanced',  # Helps if the target classes are imbalanced
                             verbose=True)
        model.optimize(**{**MODEL_OPTIMIZE_PARAMS,
                       **{"param_distributions": MODEL_SEARCH_PARAMS}})
        pickle.dump(model.model, (MODEL_REGISTRY_DIR / BEST_MODEL_FILE_NAME).open(mode='wb'))

    # Make predictions on the test set
    print("Running prediction on the test set.")
    predictions = model.predict(test.drop('rating', axis=1))

    # Calculate the accuracy
    accuracy = accuracy_score(test['rating'], predictions)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    _args: Namespace = RecSysArgumentParser().parse()

    # Parse command-line arguments
    arguments: BaseArguments = BaseArguments.from_dict(vars(_args))

    main(arguments)
