"""Project configuration utilities.

This module loads environment variables (from a local .env if present) and
exposes a simple `get_config()` helper returning common project paths and
constants used across the codebase. Keep this lightweight and import-safe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Immutable configuration container for the project.

    Attributes
    ----------
    project_root: Path
        Absolute path to the project repository root.
    data_raw_dir: Path
        Directory where raw source data is stored.
    data_processed_dir: Path
        Directory for processed/feature-engineered datasets.
    mlflow_tracking_uri: str
        MLflow tracking backend URI. Defaults to a local file store.
    log_level: str
        Default log level (e.g., "INFO", "DEBUG").
    environment: str
        Arbitrary environment label (e.g., "local", "dev", "prod").
    """

    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    mlflow_tracking_uri: str
    log_level: str
    environment: str
    raw_reviews_path: Path
    min_interactions: int
    raw_reviews_url: str


def _resolve_project_root() -> Path:
    """Resolve the project root directory.

    Assumes this file lives under `src/utils/config.py` within the repo. We walk
    up two directories from this file to get the repository root.
    """

    return Path(__file__).resolve().parents[2]


def get_config(env_file: Optional[Path] = None) -> Config:
    """Load environment and return a `Config` object.

    Parameters
    ----------
    env_file: Optional[Path]
        Optional explicit path to a `.env` file. If not provided, we attempt to
        load `.env` from the project root if it exists.
    """

    project_root = _resolve_project_root()

    # Load environment variables from the provided .env or a default at root
    candidate_env = env_file if env_file is not None else project_root / ".env"
    if candidate_env.exists():
        load_dotenv(dotenv_path=candidate_env)
    else:
        # Fallback to default search behavior without erroring if missing
        load_dotenv()

    data_dir = project_root / "data"
    data_raw_dir = data_dir / "raw"
    data_processed_dir = data_dir / "processed"

    # Collect environment overrides with sensible defaults
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    environment = os.getenv("ENVIRONMENT", "local")
    raw_reviews_path = Path(
        os.getenv("RAW_REVIEWS_PATH", str(project_root / "data/raw/reviews_electronics.json.gz"))
    )
    min_interactions = int(os.getenv("MIN_INTERACTIONS", "5"))
    raw_reviews_url = os.getenv(
        "RAW_REVIEWS_URL",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
    )

    return Config(
        project_root=project_root,
        data_raw_dir=data_raw_dir,
        data_processed_dir=data_processed_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        log_level=log_level,
        environment=environment,
        raw_reviews_path=raw_reviews_path,
        min_interactions=min_interactions,
        raw_reviews_url=raw_reviews_url,
    )


__all__ = ["Config", "get_config"]
