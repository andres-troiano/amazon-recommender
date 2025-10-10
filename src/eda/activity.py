"""User and item activity distributions for Spark DataFrames."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def user_activity(df: DataFrame) -> DataFrame:
    """Return per-user counts of ratings."""
    logger.info("Computing user activity counts")
    return df.groupBy("user_id").agg(F.count("item_id").alias("num_ratings")).orderBy(
        F.col("num_ratings").desc()
    )


def item_popularity(df: DataFrame) -> DataFrame:
    """Return per-item counts of ratings."""
    logger.info("Computing item popularity counts")
    return df.groupBy("item_id").agg(F.count("user_id").alias("num_ratings")).orderBy(
        F.col("num_ratings").desc()
    )


def plot_histogram(series: pd.Series, title: str, bins: int = 50) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.histplot(series, bins=bins)
    plt.title(title)
    plt.xlabel("count")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()


def plot_user_activity(df: DataFrame, sample: int = 100_000) -> None:
    """Plot histogram of user activity (ratings per user)."""
    pdf = user_activity(df).limit(sample).toPandas()
    plot_histogram(pdf["num_ratings"], title="User activity distribution")


def plot_item_popularity(df: DataFrame, sample: int = 100_000) -> None:
    """Plot histogram of item popularity (ratings per item)."""
    pdf = item_popularity(df).limit(sample).toPandas()
    plot_histogram(pdf["num_ratings"], title="Item popularity distribution")


__all__ = [
    "user_activity",
    "item_popularity",
    "plot_user_activity",
    "plot_item_popularity",
]
