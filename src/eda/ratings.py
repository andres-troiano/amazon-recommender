"""Rating distribution utilities for Spark DataFrames."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def distribution(df: DataFrame) -> DataFrame:
    """Return counts per rating value (rounded to nearest 0.5)."""
    logger.info("Computing rating distribution")
    return (
        df.withColumn("rating_bin", F.round(F.col("rating") * 2) / 2.0)
        .groupBy("rating_bin")
        .agg(F.count("rating").alias("count"))
        .orderBy("rating_bin")
    )


def plot_distribution(df: DataFrame, sample: int = 1_000_000) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    pdf = distribution(df).limit(sample).toPandas()
    sns.barplot(x="rating_bin", y="count", data=pdf)
    plt.title("Rating distribution")
    plt.xlabel("rating")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


__all__ = ["distribution", "plot_distribution"]
