"""Cold-start recommendation strategies."""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger


def recommend_from_popularity(popular_items: Optional[List[Dict]], n: int) -> List[Dict]:
	"""Return top-N items from precomputed popularity list."""
	if not popular_items:
		logger.warning("Popularity list unavailable; returning empty list.")
		return []
	return popular_items[:n]


__all__ = ["recommend_from_popularity"]
