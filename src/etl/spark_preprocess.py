"""Spark ETL pipeline for Amazon Reviews (Electronics subset).

Responsibilities:
- Load raw reviews CSV/TSV
- Clean and filter by minimum interactions per user/item
- Index `user_id` and `item_id` to integer indices
- Save processed interactions and index maps as Parquet

This module is intentionally self-contained for Stage 2 and avoids any
modeling logic. It will be extended in later stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.feature import StringIndexer


def get_spark(app_name: str = "amazon-recommender-etl") -> SparkSession:
    """Create or retrieve a SparkSession with sensible defaults."""

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    return spark


def _read_reviews(spark: SparkSession, input_path: Path) -> DataFrame:
    """Read raw reviews file (CSV/TSV) with header inference.

    The dataset is expected to include at least: user_id, item_id, rating.
    """

    path_str = str(input_path)
    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        df = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("sep", "\t")
            .csv(path_str)
        )
    elif path_str.endswith(".json") or path_str.endswith(".json.gz"):
        # Amazon review files are often JSON lines, sometimes gzipped
        df = spark.read.json(path_str)
    else:
        df = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(path_str)
        )

    # Normalize expected columns to string/double
    # Try common alternative column names
    candidates = {
        "user_id": ["user_id", "reviewerID", "user", "userId"],
        "item_id": ["item_id", "asin", "item", "itemId"],
        "rating": ["rating", "overall", "stars", "score"],
    }

    def pick(col_group):
        for c in col_group:
            if c in df.columns:
                return c
        return None

    user_col = pick(candidates["user_id"]) or "user_id"
    item_col = pick(candidates["item_id"]) or "item_id"
    rating_col = pick(candidates["rating"]) or "rating"

    df = df.withColumn("user_id", F.col(user_col).cast(StringType()))
    df = df.withColumn("item_id", F.col(item_col).cast(StringType()))
    df = df.withColumn("rating", F.col(rating_col).cast(DoubleType()))

    return df.select("user_id", "item_id", "rating").dropna(subset=["user_id", "item_id", "rating"])


def _filter_min_interactions(df: DataFrame, min_interactions: int) -> DataFrame:
    """Keep only users and items having at least `min_interactions`."""

    user_counts = df.groupBy("user_id").agg(F.count("item_id").alias("cnt"))
    item_counts = df.groupBy("item_id").agg(F.count("user_id").alias("cnt"))

    df_users = df.join(user_counts, on="user_id", how="inner").where(F.col("cnt") >= min_interactions).drop("cnt")
    df_items = df_users.join(item_counts, on="item_id", how="inner").where(F.col("cnt") >= min_interactions).drop("cnt")
    return df_items


def _index_ids(df: DataFrame) -> Dict[str, DataFrame]:
    """Index `user_id` and `item_id` into consecutive integer IDs.

    Returns a dict containing:
      - interactions: DataFrame with user_idx, item_idx, rating
      - user_map: DataFrame with user_idx, user_id
      - item_map: DataFrame with item_idx, item_id
    """

    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol="item_id", outputCol="item_idx", handleInvalid="skip")

    model_user = user_indexer.fit(df)
    df_user_indexed = model_user.transform(df)

    model_item = item_indexer.fit(df_user_indexed)
    df_indexed = model_item.transform(df_user_indexed)

    # Ensure integer type for indices
    df_indexed = df_indexed.withColumn("user_idx", F.col("user_idx").cast("int"))
    df_indexed = df_indexed.withColumn("item_idx", F.col("item_idx").cast("int"))

    user_labels = model_user.labels
    item_labels = model_item.labels

    # Build mapping DataFrames
    spark = df.sparkSession
    user_map = spark.createDataFrame([(i, uid) for i, uid in enumerate(user_labels)], ["user_idx", "user_id"]).orderBy("user_idx")
    item_map = spark.createDataFrame([(i, iid) for i, iid in enumerate(item_labels)], ["item_idx", "item_id"]).orderBy("item_idx")

    interactions = df_indexed.select("user_idx", "item_idx", "rating")
    return {"interactions": interactions, "user_map": user_map, "item_map": item_map}


def preprocess_reviews(input_path: Path, output_dir: Path, min_interactions: int = 5) -> Dict[str, str]:
    """Run the preprocessing pipeline end-to-end and write outputs as Parquet.

    Returns a dict of written paths.
    """

    spark = get_spark()
    try:
        logger.info(f"Loading raw reviews from: {input_path}")
        df = _read_reviews(spark, input_path)
        total_rows = df.count()
        logger.info(f"Loaded {total_rows} rows")

        logger.info(f"Filtering users/items with < {min_interactions} interactions")
        df_filt = _filter_min_interactions(df, min_interactions)

        # Persist counts
        n_users = df_filt.select("user_id").distinct().count()
        n_items = df_filt.select("item_id").distinct().count()
        logger.info(f"Post-filter: {n_users} users, {n_items} items")

        logger.info("Indexing user and item IDs")
        parts = _index_ids(df_filt)

        output_dir.mkdir(parents=True, exist_ok=True)

        interactions_path = output_dir / "interactions.parquet"
        user_map_path = output_dir / "user_map.parquet"
        item_map_path = output_dir / "item_map.parquet"

        parts["interactions"].write.mode("overwrite").parquet(str(interactions_path))
        parts["user_map"].write.mode("overwrite").parquet(str(user_map_path))
        parts["item_map"].write.mode("overwrite").parquet(str(item_map_path))

        logger.info(
            f"✅ Processed {total_rows} rows → {n_users} users × {n_items} items"
        )

        return {
            "interactions": str(interactions_path),
            "user_map": str(user_map_path),
            "item_map": str(item_map_path),
        }
    finally:
        spark.stop()


__all__ = ["get_spark", "preprocess_reviews"]
