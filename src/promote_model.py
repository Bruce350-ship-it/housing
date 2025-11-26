import mlflow
from mlflow.tracking import MlflowClient
import yaml

# ===============================
# 1️⃣ Setup (from config)
# ===============================
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
model_registry_name = cfg["mlflow"]["model_registry_name"]
target_stage = "Production"  # Stage you want to promote to

client = MlflowClient()

# ===============================
# 2️⃣ Get latest model version
# ===============================
# Get all versions of the registered model
all_versions = client.get_latest_versions(model_registry_name)

if not all_versions:
    raise ValueError(f"No registered versions found for model '{model_registry_name}'")

# Sort by version number and pick the latest
latest_version = max(all_versions, key=lambda v: int(v.version))
print(f"Latest version: {latest_version.version}, current stage: {latest_version.current_stage}")

# ===============================
# 3️⃣ Promote model
# ===============================
client.transition_model_version_stage(
    name=model_registry_name,
    version=latest_version.version,
    stage=target_stage,
    archive_existing_versions=True  # Automatically archive other Production versions
)

print(f"✅ Model version {latest_version.version} promoted to {target_stage}")