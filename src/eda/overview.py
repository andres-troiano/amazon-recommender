"""High-level dataset overview utilities for Spark DataFrames."""

from __future__ import annotations

from typing import Dict

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def basic_stats(df: DataFrame) -> Dict[str, float]:
    """Compute basic counts and sparsity for a ratings dataset.

    Expected columns: user_id, item_id, rating
    """
    logger.info("Computing basic stats")
    num_rows = df.count()
    num_users = df.select("user_id").distinct().count()
    num_items = df.select("item_id").distinct().count()
    sparsity = 1.0 - min(1.0, num_rows / float(max(1, num_users * num_items)))
    return {
        "num_rows": float(num_rows),
        "num_users": float(num_users),
        "num_items": float(num_items),
        "sparsity": float(sparsity),
    }


def rating_summary(df: DataFrame) -> Dict[str, float]:
    logger.info("Computing rating summary")
    agg = df.agg(
        F.count("rating").alias("count"),
        F.avg("rating").alias("mean"),
        F.expr("percentile(rating, 0.5)").alias("median"),
        F.min("rating").alias("min"),
        F.max("rating").alias("max"),
    ).first()
    return {k: float(agg[i]) for i, k in enumerate(["count", "mean", "median", "min", "max"])}


__all__ = ["basic_stats", "rating_summary"]
