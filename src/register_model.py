import json
import mlflow
import mlflow.sklearn
import yaml

# ===============================
# 1️⃣ Setup (from config)
# ===============================
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
model_registry_name = cfg["mlflow"]["model_registry_name"]

# ===============================
# 2️⃣ Load best model info
# ===============================
with open("artifacts/models/best_model.json", "r") as f:
    best_model_info = json.load(f)

best_run_id = best_model_info["best_run_id"]
best_model_name = best_model_info["best_model"]

print(f"Registering best model: {best_model_name}, Run ID: {best_run_id}")

# ===============================
# 3️⃣ Register model
# ===============================
model_uri = f"runs:/{best_run_id}/model"

# Register model in MLflow Model Registry
result = mlflow.register_model(model_uri=model_uri, name=model_registry_name)

print(f"Model registered: {result.name}, version: {result.version}")