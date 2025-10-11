"""ALS model training, evaluation, and recommendation utilities.

Trains a Spark MLlib ALS model on processed interactions and computes RMSE and
ranking metrics. Saves model artifacts and metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .metrics import precision_at_k, ndcg_at_k
from .utils import save_model, save_json, log_mlflow


@dataclass
class AlsTrainingResult:
    model_path: Path
    params: Dict
    metrics: Dict
    run_id: Optional[str] = None


def _split_data(df: DataFrame, seed: int) -> tuple[DataFrame, DataFrame]:
    train, val = df.randomSplit([0.8, 0.2], seed=seed)
    return train.cache(), val.cache()


def _evaluate_rmse(model, val_df: DataFrame) -> float:
    preds = model.transform(val_df)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(preds)
    return float(rmse)


def _evaluate_ranking(model, val_df: DataFrame, k: int = 10) -> Dict[str, float]:
    preds = model.transform(val_df).select("user_idx", "item_idx", "rating", "prediction")
    return {
        "precision@k": float(precision_at_k(preds, k)),
        "ndcg@k": float(ndcg_at_k(preds, k)),
    }


def train_als(
    spark: SparkSession,
    interactions_df: DataFrame,
    artifacts_dir: Path,
    rank: int = 50,
    reg: float = 0.1,
    alpha: float = 1.0,
    maxIter: int = 10,
    seed: int = 42,
) -> AlsTrainingResult:
    """Train ALS with a simple grid search and save the best model.

    Returns training result with paths and metrics.
    """

    train_df, val_df = _split_data(interactions_df, seed)

    param_grid = [
        {"rank": r, "regParam": rp}
        for r in [32, rank, 64]
        for rp in [0.05, reg, 0.2]
    ]

    best = None
    metrics_best = None
    params_best = None
    grid_results: List[Dict] = []

    for params in param_grid:
        logger.info(f"Training ALS with params: {params}")
        als = ALS(
            userCol="user_idx",
            itemCol="item_idx",
            ratingCol="rating",
            implicitPrefs=False,
            rank=int(params["rank"]),
            regParam=float(params["regParam"]),
            alpha=float(alpha),
            maxIter=int(maxIter),
            seed=int(seed),
            coldStartStrategy="drop",
        )
        model = als.fit(train_df)

        rmse = _evaluate_rmse(model, val_df)
        rank_metrics = _evaluate_ranking(model, val_df, k=10)
        logger.info(f"Validation RMSE={rmse:.4f} P@10={rank_metrics['precision@k']:.4f} NDCG@10={rank_metrics['ndcg@k']:.4f}")

        score = (rmse, -rank_metrics["precision@k"])  # primary: lower RMSE, then higher precision
        # Track this candidate's results
        candidate = {"params": params, "metrics": {"rmse": rmse, **rank_metrics}}
        grid_results.append(candidate)
        # Log candidate as nested MLflow run (if available)
        try:
            log_mlflow(
                params={"model": "als", **params},
                metrics=candidate["metrics"],
                artifacts_dir=None,
                tracking_uri=None,
                experiment_name="ALS_Recommender",
                run_name=f"candidate_rank={params['rank']}_reg={params['regParam']}",
                tags={"candidate": "true"},
                nested=True,
            )
        except Exception:  # noqa: BLE001
            pass
        if best is None or score < best:
            best = score
            metrics_best = {"rmse": rmse, **rank_metrics}
            params_best = params
            best_model = model

    # Save best model
    model_dir = artifacts_dir / "als_model"
    save_model(best_model, model_dir)
    metadata = {"params": params_best, "metrics": metrics_best}
    save_json(metadata, model_dir / "metadata.json")

    # Persist metrics JSON and attempt MLflow logging
    metrics_dir = artifacts_dir / "metrics"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_path = metrics_dir / f"als_metrics_{timestamp}.json"
    save_json({"model": "als", "params": params_best, "metrics": metrics_best}, metrics_path)
    grid_path = metrics_dir / f"als_grid_{timestamp}.json"
    save_json(
        {
            "model": "als",
            "k": 10,
            "selection": "rmse_then_precision@k",
            "grid_results": grid_results,
            "best": {"params": params_best, "metrics": metrics_best},
        },
        grid_path,
    )

    run_id = log_mlflow(
        params={"model": "als", **params_best},
        metrics=metrics_best,
        artifacts_dir=artifacts_dir,
        tracking_uri=None,  # falls back to env or file:./mlruns
        experiment_name="ALS_Recommender",
        artifact_paths=[metrics_path, grid_path, model_dir / "metadata.json"],
        run_name=f"best_rank={params_best['rank']}_reg={params_best['regParam']}",
        tags={"best": "true"},
    )
    
    return AlsTrainingResult(model_path=model_dir, params=params_best, metrics=metrics_best, run_id=run_id)


def recommend_for_user(model_path: Path, user_id: int, n: int = 10) -> List[Dict]:
    """Load a saved model and return top-N item indices for a user_idx.

    Note: user_id here refers to `user_idx` as produced in Stage 2.
    """
    from .utils import load_model

    model = load_model(model_path)
    spark = SparkSession.builder.getOrCreate()
    try:
        users = spark.createDataFrame([(int(user_id),)], ["user_idx"])
        recs = model.recommendForUserSubset(users, n).select("recommendations").first()[0]
        return [{"item_idx": int(r[0]), "score": float(r[1])} for r in recs]
    finally:
        spark.stop()


__all__ = ["train_als", "recommend_for_user", "AlsTrainingResult"]
