import json
import tempfile
import mlflow
from pathlib import Path
from typing import Any, Dict


def save_json_artifact(data_dict: Dict[str, Any], file_name: str, artifact_path: str = "metadata"):
    """
    Saves a dictionary as a JSON file and logs it as an MLflow artifact.

    Args:
        data_dict: The dictionary to save.
        file_name: Name of the file (without .json extension).
        artifact_path: The relative path within the MLflow run to store the file.
    """
    # Create a temporary directory to host the file before logging
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Ensure filename has the correct extension
        local_file_path = Path(tmp_dir) / f"{file_name}.json"

        with open(local_file_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)

        # Log the file to the specific run
        mlflow.log_artifact(local_path=str(local_file_path), artifact_path=artifact_path)
        print(f"Artifact {file_name}.json successfully logged to path: {artifact_path}")