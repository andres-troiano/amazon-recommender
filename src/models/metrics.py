"""Ranking metrics implemented with Spark DataFrame operations.

Implements precision@k, recall@k, and ndcg@k for recommendation evaluation.
Expects a predictions DataFrame with columns:
  - user_idx: int
  - item_idx: int
  - rating: actual relevance (implicit/explicit)
  - prediction: model score for ranking
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def _topk(predictions: DataFrame, k: int) -> DataFrame:
    w = Window.partitionBy("user_idx").orderBy(F.col("prediction").desc())
    return predictions.withColumn("rank", F.row_number().over(w)).where(F.col("rank") <= k)


def precision_at_k(predictions: DataFrame, k: int, threshold: float = 4.0) -> float:
    topk = _topk(predictions, k)
    rel = topk.withColumn("is_rel", (F.col("rating") >= F.lit(threshold)).cast("int"))
    per_user = rel.groupBy("user_idx").agg(F.avg("is_rel").alias("prec"))
    return per_user.agg(F.avg("prec")).first()[0] or 0.0


def recall_at_k(predictions: DataFrame, k: int, threshold: float = 4.0) -> float:
    rel_all = predictions.where(F.col("rating") >= F.lit(threshold)).select("user_idx", "item_idx")
    topk = _topk(predictions, k).select("user_idx", "item_idx")

    hits = topk.join(rel_all, ["user_idx", "item_idx"], "inner").groupBy("user_idx").agg(F.count("item_idx").alias("hits"))
    totals = rel_all.groupBy("user_idx").agg(F.count("item_idx").alias("total_rel"))

    joined = totals.join(hits, "user_idx", "left").fillna({"hits": 0})
    per_user = joined.withColumn("rec", F.when(F.col("total_rel") > 0, F.col("hits") / F.col("total_rel")).otherwise(0.0))
    return per_user.agg(F.avg("rec")).first()[0] or 0.0


def ndcg_at_k(predictions: DataFrame, k: int, threshold: float = 4.0) -> float:
    # Mark relevance (binary) for simplicity
    preds = predictions.withColumn("rel", (F.col("rating") >= F.lit(threshold)).cast("int"))

    # DCG@k
    w = Window.partitionBy("user_idx").orderBy(F.col("prediction").desc())
    ranked = preds.withColumn("rank", F.row_number().over(w)).where(F.col("rank") <= k)
    dcg = ranked.withColumn("gain", (F.col("rel") / F.log2(F.col("rank") + F.lit(1)))).groupBy("user_idx").agg(F.sum("gain").alias("dcg"))

    # IDCG@k (ideal ranking)
    w2 = Window.partitionBy("user_idx").orderBy(F.col("rel").desc())
    ideal = preds.withColumn("rank", F.row_number().over(w2)).where(F.col("rank") <= k)
    idcg = ideal.withColumn("gain", (F.col("rel") / F.log2(F.col("rank") + F.lit(1)))).groupBy("user_idx").agg(F.sum("gain").alias("idcg"))

    joined = dcg.join(idcg, "user_idx", "left").fillna({"idcg": 0.0})
    per_user = joined.withColumn("ndcg", F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(0.0))
    return per_user.agg(F.avg("ndcg")).first()[0] or 0.0


__all__ = ["precision_at_k", "recall_at_k", "ndcg_at_k"]
