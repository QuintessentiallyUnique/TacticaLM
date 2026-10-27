"""Tracking and logging setup for training."""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_mlflow(experiment_name: str, tracking_uri: str = "./mlruns"):
    """Set up MLflow tracking.
    
    Args:
        experiment_name: The name of the MLflow experiment.
        tracking_uri: The tracking URI for MLflow.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed. Skipping MLflow setup.")
        return

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking set up: uri={tracking_uri}, experiment={experiment_name}")
