import os
import sys
import yaml

# Ensure project root is on sys.path when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_config_has_required_keys():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "mlflow" in cfg
    assert "paths" in cfg
    assert "api" in cfg
    assert "tracking_uri" in cfg["mlflow"]
    assert "experiment_name" in cfg["mlflow"]
    assert "processed_data" in cfg["paths"]
    assert "best_model_pickle" in cfg["paths"]
    assert "host" in cfg["api"]
    assert "port" in cfg["api"]

