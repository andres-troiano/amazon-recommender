"""FastAPI service exposing recommendation endpoints."""

from __future__ import annotations

from typing import Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from loguru import logger

from src.utils.logging import setup_logging
from src.serving.recommender import service


@asynccontextmanager
async def lifespan(app: FastAPI):
	# Startup
	setup_logging()
	logger.info("Starting API; loading artifacts...")
	service.load_artifacts()
	logger.info("Artifacts loaded.")
	yield
	# Shutdown
	try:
		if service.spark:
			service.spark.stop()
			logger.info("Spark session stopped.")
	except Exception as e:  # noqa: BLE001
		logger.warning(f"Error during shutdown: {e}")

app = FastAPI(title="Amazon Recommender API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
	return {"status": "ok"}


@app.get("/recommendations")
def get_recommendations(user_id: str = Query(...), n: int = Query(10, ge=1, le=100)) -> Dict[str, Any]:
	items = service.recommend_for_user(user_id=user_id, n=n)
	return {"user_id": user_id, "items": items}


@app.get("/similar-items")
def get_similar_items(item_id: str = Query(...), n: int = Query(5, ge=1, le=100)) -> Dict[str, Any]:
	items = service.similar_items(item_id=item_id, n=n)
	return {"item_id": item_id, "items": items}


@app.post("/feedback")
def post_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
	# Minimal feedback logging to JSONL under artifacts
	try:
		from src.utils.config import get_config
		import json
		cfg = get_config()
		out_dir = cfg.artifacts_dir
		out_dir.mkdir(parents=True, exist_ok=True)
		with open(out_dir / "feedback.jsonl", "a") as f:
			f.write(json.dumps(payload) + "\n")
		logger.info(f"Logged feedback: {payload}")
	except Exception as e:  # noqa: BLE001
		logger.warning(f"Failed to log feedback: {e}")
	return {"status": "ok"}


__all__ = ["app"]
