"""ALS model training, evaluation, and recommendation utilities.

Trains a Spark MLlib ALS model on processed interactions and computes RMSE and
ranking metrics. Saves model artifacts and metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from loguru import logger
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .metrics import precision_at_k, ndcg_at_k
from .utils import save_model, save_json


@dataclass
class AlsTrainingResult:
    model_path: Path
    params: Dict
    metrics: Dict


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

    return AlsTrainingResult(model_path=model_dir, params=params_best, metrics=metrics_best)


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
