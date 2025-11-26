import os
import sys
import pandas as pd
from scipy.stats import ks_2samp
import joblib
import yaml

# Load config
def load_config(path: str = "config.yaml") -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)

cfg = load_config()
model_path = cfg["paths"]["best_model_pickle"]
train_stats_path = cfg["paths"].get("train_stats", "data/processed/train_stats.csv")
new_data_path = cfg["paths"].get("new_data", "data/new/Housing_new.csv")
alpha = float(cfg.get("monitoring", {}).get("ks_pvalue_threshold", 0.05))

if not os.path.exists(model_path):
	print(f"Model not found: {model_path}", file=sys.stderr)
	sys.exit(2)
if not os.path.exists(train_stats_path):
	print(f"Train stats not found: {train_stats_path}", file=sys.stderr)
	sys.exit(3)
if not os.path.exists(new_data_path):
	print(f"New data not found: {new_data_path}", file=sys.stderr)
	sys.exit(4)

# Load model & training data stats
pipeline = joblib.load(model_path)
train_stats = pd.read_csv(train_stats_path)  # saved during training

# Load new batch
new_data = pd.read_csv(new_data_path)

# Check numeric feature drift
drift_found = False
common_cols = [c for c in train_stats.columns if c in new_data.columns]
for col in common_cols:
	try:
		stat, p_value = ks_2samp(train_stats[col].dropna(), new_data[col].dropna())
		if p_value < alpha:
			print(f"DRIFT: {col} | p={p_value:.5f} < {alpha}")
			drift_found = True
	except Exception as e:
		print(f"ERROR comparing {col}: {e}", file=sys.stderr)

if drift_found:
	sys.exit(10)  # non-zero to flag in schedulers/CI
else:
	print("No drift detected")
	sys.exit(0)