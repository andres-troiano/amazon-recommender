"""Command-line interface for orchestrating the project workflow.

This CLI intentionally provides only stubbed subcommands for Stage 1.
Future stages will implement ETL, model training, evaluation, and deployment.
"""

from __future__ import annotations

import argparse
import sys

from utils.config import get_config
from utils.logging import setup_logging, logger
from etl.spark_preprocess import preprocess_reviews
from etl.feature_engineering import compute_popularity, save_popularity


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
            "Amazon Recommender System — pipeline orchestrator (Stage 1 skeleton)"
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

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
