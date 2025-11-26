Housing Price Prediction
========================

End-to-end ML project that cleans housing data, trains and tunes regression models, tracks experiments with MLflow, and serves the best model via a FastAPI API. Includes drift monitoring scaffolding and CI/CD-ready structure.

Features
--------
- Data cleaning and preprocessing (`src/preprocessing.py`)
- Model training with preprocessing pipelines (GBM, RF, LightGBM) and Optuna tuning (`src/train_model.py`)
- Experiment tracking and model registry with MLflow
- FastAPI inference service with Swagger UI (`/docs`) (`src/serve_model.py`)
- Basic drift monitoring via KS-test (`src/monitor_drift.py`)
- Docker Compose setup for one-command deployment
- GitHub Actions workflow scaffold

Setup
-----
1. Python 3.12 recommended (or 3.11+)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configuration: edit `config.yaml` if needed (paths, MLflow, API host/port).

Data
----
Place raw data at `data/raw/Housing.csv`. Clean it:

```bash
python src/preprocessing.py
```

This writes `data/processed/Housing_clean.csv`.

Workflows
=========

## Option 1: Manual Workflow (Step-by-Step)

Run each script individually for full control over the pipeline.

### Step 1: Preprocess Data
```bash
python src/preprocessing.py
```

### Step 2: Train Models
```bash
python src/train_model.py
```

**Outputs:**
- MLflow runs under `artifacts/mlruns`
- Tuned pipeline at `artifacts/models/GradientBoosting_optuna_pipeline.pkl`
- Training stats at `data/processed/train_stats.csv`
- Best model info at `artifacts/models/best_model.json`

### Step 3: Register Model
Register the best model in MLflow Model Registry:

```bash
python src/register_model.py
```

### Step 4: Promote Model
Promote the registered model to Production stage:

```bash
python src/promote_model.py
```

### Step 5: Start MLflow UI (Optional)
View experiments and model registry:

```bash
mlflow server --host 127.0.0.1 --port 5001
```

Then open: http://127.0.0.1:5001

### Step 6: Serve API
Start the FastAPI server:

```bash
uvicorn src.serve_model:app --host 127.0.0.1 --port 8000 --reload
```

**Access:**
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

---

## Option 2: Docker Workflow (Automated)

Run the entire pipeline automatically with Docker Compose. All steps (train → register → promote → serve) execute in sequence when the container starts.

### Prerequisites
- Docker and Docker Compose installed
- Data file at `data/raw/Housing.csv`

### Start Everything
```bash
docker compose up --build
```

This will:
1. **Build** the API container with all dependencies
2. **Start MLflow server** on port 5001
3. **Start API container** which automatically:
   - Trains models (logs experiments to MLflow)
   - Registers the best model
   - Promotes it to Production stage
   - Starts the API server on port 8000

### Access Services
Once containers are running:

- **MLflow UI**: http://127.0.0.1:5001
  - View experiments in the "Experiments" tab
  - View registered models in the "Models" tab
- **Swagger UI**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

### Useful Docker Commands

**View logs:**
```bash
docker compose logs -f api        # API container logs
docker compose logs -f mlflow     # MLflow container logs
```

**Stop services:**
```bash
docker compose down
```

**Rebuild and restart:**
```bash
docker compose up --build --force-recreate
```

**Check container status:**
```bash
docker compose ps
```

### Notes
- The API container runs the full pipeline on startup. If you need to retrain, restart the container.
- All artifacts are persisted in `./artifacts` and `./data` directories (mounted as volumes).
- MLflow data is stored in `./artifacts/mlruns` (accessible from both containers).

---

Prediction Example
------------------

Send a POST request to `/predict` with:

```json
{
  "area": 4000,
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 2,
  "parking": 1,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "prefarea": "no",
  "furnishingstatus": "semi-furnished"
}
```

Response:
```json
{
  "predicted_price": 5673054.43
}
```

Drift Monitoring
----------------
Compare new data against training statistics:

```bash
python src/monitor_drift.py
```

Requires:
- Training stats at `data/processed/train_stats.csv` (saved during training)
- New data at `data/new/Housing_new.csv`

Model Registry
--------------
The model registry workflow is integrated into both manual and Docker workflows:

**Manual:**
1. `python src/register_model.py` - Registers best model
2. `python src/promote_model.py` - Promotes to Production

**Docker:**
- Registration and promotion happen automatically during container startup

**View in MLflow UI:**
- Navigate to http://127.0.0.1:5001
- Click "Models" tab (not "Experiments")
- You should see "HousingPriceModel" with version 1 in Production stage

Testing
-------
Run the test suite:

```bash
# Run all tests
pytest -q

# Run specific test file
python tests/test_api.py
pytest tests/test_preprocessing.py
pytest tests/test_config.py
```

Project Structure
-----------------
```
.
├── src/                    # Source code
│   ├── preprocessing.py    # Data cleaning
│   ├── train_model.py      # Model training & tuning
│   ├── register_model.py  # Model registration
│   ├── promote_model.py   # Model promotion
│   ├── serve_model.py     # FastAPI server
│   ├── monitor_drift.py   # Drift detection
│   └── batch_infer.py      # Batch inference
├── tests/                  # Unit tests
├── notebooks/              # EDA and prototyping
├── artifacts/              # Models and MLflow runs
│   ├── models/             # Saved model pipelines
│   └── mlruns/             # MLflow tracking data
├── data/                   # Datasets
│   ├── raw/                # Raw input data
│   └── processed/          # Cleaned data
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── Dockerfile              # API container definition
├── docker-compose.yml      # Docker Compose configuration
└── docker-entrypoint.sh   # Container startup script
```

Configuration
-------------
Edit `config.yaml` to customize:

- **MLflow settings**: tracking URI, experiment name, model registry name
- **Paths**: data locations, model artifacts
- **API settings**: host and port

For Docker, ensure paths in `config.yaml` match the container's working directory (`/app`).
