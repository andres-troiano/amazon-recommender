# 🧭 Project Description — Amazon Product Recommender System

## 🎯 Goal

Build an **end-to-end recommender system** using the **Amazon Product Reviews dataset (Electronics subset)**.
The system should demonstrate:

* **High business value:** personalized product recommendations for an e-commerce platform.
* **Strong ML engineering:** scalable data processing with **PySpark**, reproducible experiments with **MLflow**, and modern **deployment with FastAPI + Docker**.
* **Senior-level completeness:** from raw data to interactive API/UI with support for toy demo users and cold-start logic.

---

## 🏗️ Architecture Overview

**High-level pipeline:**

```mermaid
flowchart LR
    A[Amazon Reviews (raw CSV)] --> B[ETL & Feature Engineering (Spark)]
    B --> C[ALS Model (Spark MLlib)]
    B --> D[Deep Model (NCF or Two-Tower, PyTorch)]
    C & D --> E[Evaluation (Precision@K, NDCG)]
    E --> F[Model Registry (MLflow)]
    F --> G[Serving API (FastAPI)]
    G --> H[UI (Streamlit / Gradio)]
    G --> I[Feedback Logging (Simulated)]
    I --> B
```

---

## 🧰 Tech Stack

| Layer                   | Tools / Frameworks                                                       |
| ----------------------- | ------------------------------------------------------------------------ |
| **Data & ETL**          | PySpark, Pandas, Parquet, AWS S3 / MinIO                                 |
| **Modeling**            | Spark MLlib (ALS), PyTorch / TensorFlow (Neural Collaborative Filtering) |
| **Experiment Tracking** | MLflow                                                                   |
| **Deployment**          | FastAPI, Docker, Docker Compose                                          |
| **Frontend (Optional)** | Streamlit or Gradio                                                      |
| **Storage**             | Postgres or SQLite for metadata, Redis (optional caching)                |
| **Monitoring**          | Matplotlib/Plotly dashboards, simulated logs                             |

---

## 📂 Repository Structure

```
amazon-recommender/
├── data/
│   ├── raw/                     # downloaded reviews subset
│   ├── processed/               # parquet feature tables
│   ├── demo_users.json          # toy profiles for UI
│   └── popular_items.parquet
├── src/
│   ├── etl/                     # spark_preprocess.py, feature_engineering.py
│   ├── models/
│   │   ├── als.py               # collaborative filtering
│   │   ├── ncf.py               # deep learning model
│   │   └── metrics.py           # precision@k, ndcg
│   ├── serving/
│   │   ├── api.py               # FastAPI app
│   │   ├── recommender.py       # business logic
│   │   ├── cold_start.py        # popularity + content fallback
│   │   └── toy_users.json       # demo profiles
│   ├── pipeline.py              # CLI orchestrator: etl/train/eval/deploy
│   └── utils/                   # config loader, logging, paths
├── notebooks/
│   ├── eda.ipynb
│   ├── model_comparison.ipynb
│   └── evaluation.ipynb
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🔄 Workflow Stages

### 1. **ETL & Feature Engineering (Spark)**

* Load subset of Amazon Electronics reviews.
* Filter users/items with ≥ 5 interactions.
* Generate numeric IDs for users and items.
* Save processed tables as Parquet.
* Optional: compute global popularity stats.

### 2. **Model Training**

**(a) Collaborative Filtering – ALS)**

* Train Spark ALS model (implicit feedback).
* Optimize rank, regParam, and alpha.
* Log metrics (RMSE, Precision@K) with MLflow.

**(b) Neural Recommender – NCF / Two-Tower)**

* Train small PyTorch model using user/item embeddings.
* Compare metrics to ALS.
* Export model + embeddings.

### 3. **Evaluation**

* Offline metrics: Precision@K, Recall@K, NDCG, Coverage, Diversity.
* Business metrics simulation: expected CTR / conversion lift.

### 4. **Serving Layer**

* **API Endpoints:**

  * `GET /recommendations?user_id=U&n=10`
  * `GET /similar-items?item_id=I&n=5`
  * `POST /feedback`
* **Cold-start fallback:**

  * Known user → personalized.
  * Known interests → content-based embedding similarity.
  * Unknown user → top-popular items.

### 5. **Demo Frontend (Optional)**

* Streamlit app:

  * Dropdown: select toy user (Alice, Bob, etc.)
  * Show personalized grid of recommended items.
  * Allow feedback simulation (like / skip).

### 6. **Monitoring & Retraining**

* Log API calls and simulated feedback.
* Periodically retrain using latest interactions.

---

## 🧠 Business Framing (for README & LinkedIn)

> “This project demonstrates how personalization increases user engagement and revenue for an e-commerce platform. Using Amazon’s product review data, we built a scalable recommender system in Spark, compared classical ALS with deep learning approaches, and deployed an interactive API that serves personalized recommendations — including fallback strategies for cold users.”

---

## 💡 Key Deliverables

* **End-to-end reproducible pipeline** (Spark → model → API).
* **Comprehensive README** with diagrams, results, and demo instructions.
* **Interactive demo** (Streamlit or video).
* **LinkedIn-ready summary** emphasizing scalability, ML engineering, and business insight.

---

## 🧠 Stretch Goals (if time allows)

* Integrate Sentence-BERT embeddings for content-based hybrid recommendations.
* Deploy API to AWS ECS / Lambda / GCP Cloud Run.
* Add FAISS for approximate nearest neighbor retrieval.
* Add simple CI/CD workflow (GitHub Actions).
