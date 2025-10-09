# Amazon Product Recommender System (Scaffold)

An end-to-end recommender system built on the Amazon Product Reviews dataset (Electronics subset). This repository currently contains the Stage 1 scaffold: project structure, environment, and CLI skeleton. Modeling and API serving will be implemented in subsequent stages.

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
docker-compose build
docker-compose up -d
# Exec into container if needed
docker exec -it amazon-recommender-app bash
```

### Local (No Docker)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline CLI

```bash
python src/pipeline.py --help
python src/pipeline.py etl
python src/pipeline.py train
python src/pipeline.py eval
python src/pipeline.py deploy
```

Each command is a placeholder in Stage 1.

## Configuration & Logging
- Environment variables can be set in a `.env` at repo root.
- `src/utils/config.py` loads configuration and common paths.
- `src/utils/logging.py` sets up a project-wide Loguru logger.

## Roadmap
1. ETL & Feature Engineering with PySpark
2. Model Training: ALS (Spark) and NCF (PyTorch)
3. Evaluation & MLflow tracking
4. Serving API with FastAPI + Docker
5. Optional Streamlit/Gradio demo and cold-start strategies

## License
This project is for educational and portfolio demonstration purposes.
