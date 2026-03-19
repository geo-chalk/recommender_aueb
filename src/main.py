"""
Main module of the library with MLflow integration
"""
from argparse import Namespace

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from recommender_app.generators.segment_generator import generate_segments
from recommender_app.ml.modelling import ModelTrainer
from recommender_app.utils.input_args import RecSysArgumentParser, BaseArguments
from recommender_app.utils.mlflow_utils import save_json_artifact
from recommender_app.utils.variables import (
    PROCESSED_DATA_DIR, MODEL_SEARCH_PARAMS,
    TEST_FILE_NAME,
    MODEL_OPTIMIZE_PARAMS,
    SEGMENT_1, SEGMENT_2, RESTAURANT,
    MLFLOW_TRACKING_URI, ALIAS, REGISTERED_MODEL_NAME
)


def main(input_args: BaseArguments):
    # Mlflow stuff
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Restaurant_Recommender")
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_datasets=True,
        # exclusive=False,
        max_tuning_runs=None
    )

    # --- Load test data and model ---
    if input_args.skip_training:
        print("Skipping training. Attempting to load best model and data from MLflow...")

        client = mlflow.MlflowClient()

        # Attempt to load best model from registry
        try:
            # Use Alias (champion) to load the model and get the run_id to access artifacts
            model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)
            run_id = model_version.run_id
            model_uri = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}"

            # Load the Model
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Model loaded from MLflow: {model_uri} (Run ID: {run_id})")

        except Exception as e:
            print(f"Could not load from MLflow: {e}")
            raise RuntimeError("Registry is empty. Run training first.")

        # Load test data
        try:
            # Download the EXACT test set used by this champion
            # We use the run_id and the relative artifact path defined in training
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=f"data/{TEST_FILE_NAME}"
            )
            test = pd.read_parquet(local_path)
            print(f"Test set loaded successfully from MLflow artifacts.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {TEST_FILE_NAME}. Cannot run skip-training.")

        # Evaluate
        print("Running prediction on the test set.")
        predictions = model.predict(test.drop('rating', axis=1))
        accuracy = accuracy_score(test['rating'], predictions)
        print(f"Final Accuracy: {accuracy:.4f}")

    # --- Run Training and Log to MLflow ---
    else:
        df: pd.DataFrame = generate_segments(restaurant_dict=RESTAURANT,
                                             segment_list=[SEGMENT_1, SEGMENT_2])
        train, test = train_test_split(df, test_size=0.2, random_state=42)

        # # we can either save locally of (better) log the datset to mlflow
        # train.to_csv(PROCESSED_DATA_DIR / TRAIN_FILE_NAME, index=False)
        # test.to_csv(PROCESSED_DATA_DIR / TEST_FILE_NAME, index=False)

        with mlflow.start_run() as run:
            # Optimize model
            trainer = ModelTrainer(
                df=train,
                predicted_col='rating',
                test_size=0.2,
                random_seed=42,
                auto_class_weights='Balanced',
                verbose=True
            )

            trainer.optimize(**{**MODEL_OPTIMIZE_PARAMS,
                                **{"param_distributions": MODEL_SEARCH_PARAMS}})

            # Use the optimized model inside the trainer
            model = trainer.model

            # Evaluate
            print("Running prediction on the test set.")
            predictions = trainer.predict(test.drop('rating', axis=1))
            accuracy = accuracy_score(test['rating'], predictions)
            print(f"Final Accuracy: {accuracy:.4f}")

            # Log and Register the model (this handles Versioning)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME,
                serialization_format="skops",
                skops_trusted_types=["catboost.core.CatBoostClassifier"]
            )
            print(f"Model trained and logged to MLflow as Version {REGISTERED_MODEL_NAME}")

            # Log parameters, metrics, and the model
            save_json_artifact(SEGMENT_1, "segment_1")
            save_json_artifact(SEGMENT_2, "segment_2")
            save_json_artifact(RESTAURANT, "restaurant")

            mlflow.log_params(MODEL_OPTIMIZE_PARAMS)
            mlflow.log_metric("test_accuracy", accuracy)
            test.to_parquet(PROCESSED_DATA_DIR / TEST_FILE_NAME)
            mlflow.log_artifact(str(PROCESSED_DATA_DIR / TEST_FILE_NAME), artifact_path="data")


if __name__ == "__main__":
    _args: Namespace = RecSysArgumentParser().parse()
    arguments: BaseArguments = BaseArguments.from_dict(vars(_args))
    main(arguments)
