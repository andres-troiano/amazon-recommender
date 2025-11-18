"""Recommendation orchestration for the serving layer.

Loads saved ALS artifacts and provides helpers for:
- Personalized recommendations for known users
- Item-to-item similarity recommendations
- Cold-start fallbacks (delegated to cold_start.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.utils.config import get_config
from src.serving.cold_start import recommend_from_popularity


class RecommenderService:
	"""Holds model and lookup tables for serving requests."""

	def __init__(self) -> None:
		self.spark: Optional[SparkSession] = None
		self.als_model: Optional[ALSModel] = None
		self.item_idx_to_vec: Optional[Dict[int, np.ndarray]] = None
		self.item_idx_to_id: Optional[Dict[int, str]] = None
		self.user_id_to_idx: Optional[Dict[str, int]] = None
		self.popular_items: Optional[List[Dict]] = None

	def _ensure_spark(self) -> SparkSession:
		if self.spark is None:
			self.spark = (
				SparkSession.builder.appName("recommender-serving")
				.config("spark.sql.session.timeZone", "UTC")
				.getOrCreate()
			)
		return self.spark

	def load_artifacts(self) -> None:
		"""Load ALS model, item/user maps, and popularity table."""
		cfg = get_config()
		spark = self._ensure_spark()

		model_dir = cfg.artifacts_dir / "als_model"
		if model_dir.exists():
			logger.info(f"Loading ALS model from {model_dir}")
			self.als_model = ALSModel.load(str(model_dir))
			# Load item factors into memory for cosine similarity
			logger.info("Loading item factors for similarity search")
			item_factors_df = self.als_model.itemFactors
			item_factors_pd = item_factors_df.toPandas()
			self.item_idx_to_vec = {
				int(row["id"]): self._normalize(np.array(row["features"], dtype=np.float32))
				for _, row in item_factors_pd.iterrows()
			}
		else:
			logger.warning(f"ALS model directory not found: {model_dir}")

		# Load mapping tables
		user_map_path = cfg.data_processed_dir / "user_map.parquet"
		item_map_path = cfg.data_processed_dir / "item_map.parquet"
		if user_map_path.exists():
			user_map_df = spark.read.parquet(str(user_map_path))
			self.user_id_to_idx = {
				row["user_id"]: int(row["user_idx"]) for row in user_map_df.select("user_id", "user_idx").collect()
			}
		else:
			logger.warning(f"user_map not found at {user_map_path}")
		if item_map_path.exists():
			item_map_df = spark.read.parquet(str(item_map_path))
			self.item_idx_to_id = {
				int(row["item_idx"]): row["item_id"] for row in item_map_df.select("item_idx", "item_id").collect()
			}
		else:
			logger.warning(f"item_map not found at {item_map_path}")

		# Load popularity
		pop_path = cfg.data_processed_dir / "popular_items.parquet"
		if pop_path.exists():
			pop_df = spark.read.parquet(str(pop_path)).orderBy(F.col("count_ratings").desc())
			pop_pd = pop_df.toPandas()
			# prefer item_id if present, else item_idx (map back)
			items: List[Dict] = []
			for _, row in pop_pd.iterrows():
				if "item_id" in pop_pd.columns:
					items.append({"item_id": row.get("item_id"), "score": float(row.get("count_ratings", 0))})
				else:
					idx = int(row.get("item_idx"))
					item_id = self.item_idx_to_id.get(idx, str(idx)) if self.item_idx_to_id else str(idx)
					items.append({"item_id": item_id, "score": float(row.get("count_ratings", 0))})
			self.popular_items = items
		else:
			logger.warning(f"popular_items not found at {pop_path}")

	def recommend_for_user(self, user_id: str, n: int = 10) -> List[Dict]:
		"""Return top-N personalized recommendations when possible; else fall back to popularity."""
		if self.als_model and self.user_id_to_idx and user_id in self.user_id_to_idx:
			user_idx = self.user_id_to_idx[user_id]
			spark = self._ensure_spark()
			users_df = spark.createDataFrame([(int(user_idx),)], ["user_idx"])
			recs = self.als_model.recommendForUserSubset(users_df, n).select("recommendations").first()
			if recs and recs[0]:
				items: List[Dict] = []
				for item_idx, score in recs[0]:
					item_id = self.item_idx_to_id.get(int(item_idx), str(item_idx)) if self.item_idx_to_id else str(item_idx)
					items.append({"item_id": item_id, "score": float(score)})
				return items

		# cold start / fallback: popularity
		return recommend_from_popularity(self.popular_items, n)

	def similar_items(self, item_id: str, n: int = 5) -> List[Dict]:
		"""Return top-N similar items using cosine similarity over ALS item factors."""
		if not self.item_idx_to_vec or not self.item_idx_to_id:
			logger.warning("Item similarity requested but item factors not loaded; falling back to popularity.")
			return recommend_from_popularity(self.popular_items, n)
		# Map item_id to item_idx
		# Build reverse map id->idx
		id_to_idx = {v: k for k, v in self.item_idx_to_id.items()}
		if item_id not in id_to_idx:
			logger.warning(f"Item {item_id} not found; falling back to popularity.")
			return recommend_from_popularity(self.popular_items, n)
		target_idx = id_to_idx[item_id]
		target_vec = self.item_idx_to_vec.get(target_idx)
		if target_vec is None:
			logger.warning(f"Item vector for idx {target_idx} missing; falling back to popularity.")
			return recommend_from_popularity(self.popular_items, n)
		# Compute cosine similarity
		scores: List[Tuple[str, float]] = []
		for idx, vec in self.item_idx_to_vec.items():
			if idx == target_idx:
				continue
			sim = float(np.dot(target_vec, vec))
			scores.append((self.item_idx_to_id.get(idx, str(idx)), sim))
		scores.sort(key=lambda x: x[1], reverse=True)
		return [{"item_id": iid, "score": sc} for iid, sc in scores[:n]]

	@staticmethod
	def _normalize(x: np.ndarray) -> np.ndarray:
		n = np.linalg.norm(x)
		return x / n if n > 0 else x


# Singleton instance used by API
service = RecommenderService()


__all__ = ["service", "RecommenderService"]
