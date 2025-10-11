"""Model persistence and tracking utilities for ALS training.

Provides helpers to save/load Spark ALS models and to log MLflow runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
import os
import re

from loguru import logger


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    logger.info(f"Saved JSON: {path}")


def save_model(model, path: Path) -> None:
    """Save a Spark MLlib model, overwriting any existing folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import shutil
        shutil.rmtree(path)  # remove old model dir
    model.write().overwrite().save(str(path))
    logger.info(f"Saved model to: {path}")


def load_model(path: Path):
    from pyspark.ml.recommendation import ALSModel

    return ALSModel.load(str(path))


def log_mlflow(
    params: Dict,
    metrics: Dict,
    artifacts_dir: Optional[Path] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Optional[str]:
    """Log params/metrics to MLflow, return run_id if successful.

    This function is resilient: if MLflow is unavailable, it logs a warning and
    returns None without raising.
    """

    try:
        import mlflow
    except Exception as e:  # noqa: BLE001
        logger.warning(f"MLflow not available: {e}")
        return None

    try:
        # Configure tracking and experiment
        mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        def _sanitize_mlflow_name(name: str) -> str:
            # Make common metric names readable and MLflow-safe
            name = name.replace("@", "_at_")
            return re.sub(r"[^A-Za-z0-9_\-\. :/]", "_", name)

        with mlflow.start_run():
            for k, v in params.items():
                mlflow.log_param(_sanitize_mlflow_name(k), v)
            for k, v in metrics.items():
                mlflow.log_metric(_sanitize_mlflow_name(k), float(v))
            run_id = mlflow.active_run().info.run_id
        logger.info(f"Logged MLflow run: {run_id}")

        if artifacts_dir:
            (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "metrics" / "mlflow_run_id.txt").write_text(run_id)
        return run_id
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to log to MLflow: {e}")
        return None


__all__ = ["save_model", "load_model", "save_json", "log_mlflow"]
