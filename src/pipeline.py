"""Command-line interface for orchestrating the project workflow.

Stage 2: ETL is implemented (PySpark preprocessing and popularity stats).
Subcommands for training/evaluation/deployment remain placeholders for now.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.config import get_config
from utils.logging import setup_logging, logger
from utils.data import download_file
from etl.spark_preprocess import preprocess_reviews
from etl.feature_engineering import compute_popularity, save_popularity
from models.als import train_als
import uvicorn


def _handle_etl(_: argparse.Namespace) -> int:
    logger.info("ETL step not yet implemented.")
    return 0


def _handle_train(_: argparse.Namespace) -> int:
    logger.info("Training step not yet implemented.")
    return 0


def _handle_eval(_: argparse.Namespace) -> int:
    logger.info("Evaluation step not yet implemented.")
    return 0


def _handle_deploy(_: argparse.Namespace) -> int:
    logger.info("Deployment step not yet implemented.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Amazon Recommender System — pipeline orchestrator (Stage 2: ETL ready)"
        )
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    etl_parser = subparsers.add_parser("etl", help="Run ETL & feature engineering")
    etl_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw reviews file (CSV/TSV). Defaults to RAW_REVIEWS_PATH in config",
    )
    etl_parser.add_argument(
        "--min-interactions",
        type=int,
        default=None,
        help="Minimum interactions threshold per user/item (default from config)",
    )
    etl_parser.set_defaults(func=_handle_etl)

    train_parser = subparsers.add_parser("train", help="Train recommendation models")
    train_parser.set_defaults(func=_handle_train)

    train_als_parser = subparsers.add_parser("train_als", help="Train ALS model")
    train_als_parser.add_argument("--rank", type=int, default=50)
    train_als_parser.add_argument("--reg", type=float, default=0.1)
    train_als_parser.add_argument("--alpha", type=float, default=1.0)
    train_als_parser.add_argument("--maxiter", type=int, default=10)
    train_als_parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Optional fraction (0-1] to subsample interactions for faster dev",
    )
    train_als_parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for subsampling",
    )
    train_als_parser.set_defaults(func=_handle_train)

    serve_parser = subparsers.add_parser("serve_api", help="Run FastAPI serving layer")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=_handle_deploy)
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained models")
    eval_parser.set_defaults(func=_handle_eval)

    deploy_parser = subparsers.add_parser("deploy", help="Serve models via API")
    deploy_parser.set_defaults(func=_handle_deploy)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Initialize config and logging as early as possible
    cfg = get_config()
    setup_logging(log_level=cfg.log_level)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "etl":
        # Resolve config and parameters
        input_path = Path(args.input) if args.input else cfg.raw_reviews_path
        min_interactions = (
            args.min_interactions if args.min_interactions is not None else cfg.min_interactions
        )

        # Auto-download dataset if not present locally
        if not input_path.exists():
            logger.info(
                f"Input file not found at {input_path}. Attempting download from configured URL."
            )
            download_file(cfg.raw_reviews_url, input_path)

        logger.info(
            f"Starting ETL with input={input_path}, min_interactions={min_interactions}"
        )
        outputs = preprocess_reviews(
            input_path=input_path, output_dir=cfg.data_processed_dir, min_interactions=min_interactions
        )

        # Compute popularity from interactions parquet
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        try:
            interactions_df = spark.read.parquet(outputs["interactions"])
            popularity_df = compute_popularity(interactions_df)
            popular_path = save_popularity(popularity_df, cfg.data_processed_dir)
            logger.info(f"Saved popularity table to: {popular_path}")
        finally:
            spark.stop()

        return 0

    if args.command == "train_als":
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        try:
            interactions_path = cfg.data_processed_dir / "interactions.parquet"
            interactions_df = spark.read.parquet(str(interactions_path))
            if 0.0 < getattr(args, "sample_fraction", 1.0) < 1.0:
                frac = float(args.sample_fraction)
                seed = int(args.sample_seed)
                logger.info(f"Subsampling interactions with fraction={frac}, seed={seed}")
                interactions_df = interactions_df.sample(withReplacement=False, fraction=frac, seed=seed)
            result = train_als(
                spark=spark,
                interactions_df=interactions_df,
                artifacts_dir=cfg.artifacts_dir,
                rank=args.rank,
                reg=args.reg,
                alpha=args.alpha,
                maxIter=args.maxiter,
            )
            logger.info(
                f"✅ Trained ALS: RMSE={result.metrics['rmse']:.4f}, P@10={result.metrics['precision@k']:.4f}, NDCG@10={result.metrics['ndcg@k']:.4f}"
            )
            if result.run_id:
                logger.info(f"MLflow run_id: {result.run_id}")
        finally:
            spark.stop()

        return 0
    if args.command == "serve_api":
        logger.info("Starting serving API on FastAPI (http://localhost:8000/docs)")
        # Import the app object with fallbacks for both invocation styles:
        # - `python -m src.pipeline serve_api` (package import works)
        # - `python src/pipeline.py serve_api` (use local module import)
        try:
            from src.serving.api import app as fastapi_app  # type: ignore
        except ModuleNotFoundError:
            try:
                from serving.api import app as fastapi_app  # type: ignore
            except ModuleNotFoundError:
                import sys as _sys
                from pathlib import Path as _Path
                # Ensure project root on sys.path so `src` is importable
                _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
                from src.serving.api import app as fastapi_app  # type: ignore
        uvicorn.run(fastapi_app, host=args.host, port=args.port, reload=False)
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
