# Amazon Product Recommender System

An end-to-end recommender system built on the Amazon Product Reviews dataset (Electronics subset).

- Stage 1: scaffold, environment, and CLI skeleton — ✅ completed
- Stage 2: PySpark ETL with popularity stats — ✅ completed
- Next stages: modeling, evaluation, serving, and demo UI

## Architecture Overview

<p align="center">
  <img src="docs/pipeline.svg" width="500"/><br/>
  <em>Diagram source: <a href="docs/pipeline.mmd">docs/pipeline.mmd</a></em>
</p>

## Repository Layout

```
amazon-recommender/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── etl/
│   ├── models/
│   ├── serving/
│   ├── utils/
│   │   ├── config.py
│   │   └── logging.py
│   └── pipeline.py
├── notebooks/
│   └── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── Makefile
└── README.md
```

## Getting Started

### Prerequisites
- Docker and Docker Compose
- (Optional) Python 3.10+ if running locally without Docker

### Quickstart (Docker)
```bash
docker compose build
docker compose up -d
# Exec into container if needed
docker exec -it amazon-recommender-app bash
```

### Local (No Docker)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `requirements.txt` is configured with the CPU-only PyTorch wheel; no CUDA/GPU is required.

## Pipeline CLI

```bash
python src/pipeline.py --help
python src/pipeline.py etl [--input PATH] [--min-interactions N]
python src/pipeline.py train
python src/pipeline.py eval
python src/pipeline.py deploy
```

The `etl` command is implemented in Stage 2. Other commands are placeholders for now.

### ETL (Stage 2)

Inputs (CSV/TSV/JSON/JSON.GZ): must contain fields equivalent to `user_id`, `item_id`, `rating`.

Run with Docker:
```bash
docker exec -it amazon-recommender-app \
  python src/pipeline.py etl \
  --input data/raw/reviews_electronics.json.gz \
  --min-interactions 5
```

Run locally (no Docker):
```bash
python src/pipeline.py etl --input data/raw/reviews_electronics.json.gz --min-interactions 5
```

Outputs are written to `data/processed/`:
- `interactions.parquet` — columns: `user_idx`, `item_idx`, `rating`
- `user_map.parquet` — columns: `user_idx`, `user_id`
- `item_map.parquet` — columns: `item_idx`, `item_id`
- `popular_items.parquet` — columns: `item_idx` or `item_id`, `count_ratings`, `avg_rating`

Example log on success:
```
✅ Processed X rows → Y users × Z items
Saved popularity table to: data/processed/popular_items.parquet
```

If the input file does not exist, the ETL will automatically download it from `RAW_REVIEWS_URL` into `RAW_REVIEWS_PATH`.

## Configuration & Logging
- Environment variables can be set in a `.env` at repo root.
- An example file is provided: `.env.example` (copy to `.env`).
- `src/utils/config.py` loads configuration and common paths.
- `src/utils/logging.py` sets up a project-wide Loguru logger.

Key variables (with defaults):
- `RAW_REVIEWS_PATH`: `data/raw/reviews_electronics.json.gz`
- `RAW_REVIEWS_URL`: `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz`
- `MIN_INTERACTIONS`: `5`
- `LOG_LEVEL`: `INFO`
- `ENVIRONMENT`: `local`
- `MLFLOW_TRACKING_URI`: `file:./mlruns`

## Roadmap
1. Model Training: ALS (Spark) and NCF (PyTorch)
2. Evaluation & MLflow tracking
3. Serving API with FastAPI + Docker
4. Optional Streamlit/Gradio demo and cold-start strategies

## License
This project is for educational and portfolio demonstration purposes.
