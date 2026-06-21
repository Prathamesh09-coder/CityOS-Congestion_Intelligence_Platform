"""MLflow utility wrappers for experiment setup, model logging, and report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow


def setup_experiment(name: str, tracking_uri: str = "mlruns") -> str:
    """Create or get an MLflow experiment by name.

    Args:
        name: Experiment name (e.g., "m1_closure_classifier").
        tracking_uri: MLflow tracking URI. Defaults to local "mlruns/" directory.

    Returns:
        The experiment ID as a string.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(name)
    return experiment_id


def log_dict_as_params(d: dict[str, Any], prefix: str = "") -> None:
    """Log a (possibly nested) dict as flat MLflow params.

    Args:
        d: Dictionary of parameters.
        prefix: Key prefix for nested dicts.
    """
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            log_dict_as_params(v, prefix=key)
        elif isinstance(v, (list, tuple)):
            mlflow.log_param(key, json.dumps(v))
        else:
            mlflow.log_param(key, v)


def log_model_artifact(
    model: Any,
    name: str,
    artifact_dir: str = "models",
) -> Path:
    """Serialize and log a model artifact to MLflow.

    Saves the model using joblib (sklearn-compatible) or torch.save (PyTorch),
    then logs it as an MLflow artifact.

    Args:
        model: The trained model object.
        name: Filename for the serialized model (without extension).
        artifact_dir: Subdirectory within MLflow artifacts.

    Returns:
        Path to the saved model file.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / name

        # Try torch first
        try:
            import torch

            if isinstance(model, torch.nn.Module):
                model_path = model_path.with_suffix(".pt")
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(str(model_path), artifact_dir)
                return model_path
        except ImportError:
            pass

        # Try CatBoost
        try:
            from catboost import CatBoost

            if isinstance(model, CatBoost):
                model_path = model_path.with_suffix(".cbm")
                model.save_model(str(model_path))
                mlflow.log_artifact(str(model_path), artifact_dir)
                return model_path
        except ImportError:
            pass

        # Default: joblib (sklearn-compatible)
        import joblib

        model_path = model_path.with_suffix(".pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_dir)
        return model_path


def log_markdown_report(report_path: str | Path) -> None:
    """Log a markdown report file as an MLflow artifact.

    Args:
        report_path: Path to the .md report file.
    """
    mlflow.log_artifact(str(report_path), "reports")
