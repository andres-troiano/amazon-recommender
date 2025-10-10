"""Feature engineering helpers for ETL outputs.

Provides utilities to compute popularity statistics and other aggregates used
for cold-start recommendations and monitoring.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_popularity(interactions: DataFrame) -> DataFrame:
    """Compute item popularity metrics from interactions DataFrame.

    Expects columns: item_idx (optional), item_id (optional), rating
    At least one of item_idx or item_id should be present.
    """

    item_col = "item_idx" if "item_idx" in interactions.columns else "item_id"

    popularity = (
        interactions.groupBy(item_col)
        .agg(
            F.count(F.lit(1)).alias("count_ratings"),
            F.avg("rating").alias("avg_rating"),
        )
        .orderBy(F.col("count_ratings").desc())
    )
    return popularity


def save_popularity(popularity: DataFrame, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    popular_path = output_dir / "popular_items.parquet"
    popularity.write.mode("overwrite").parquet(str(popular_path))
    return str(popular_path)


__all__ = ["compute_popularity", "save_popularity"]


