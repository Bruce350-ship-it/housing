import os
import sys
from fastapi.testclient import TestClient

# Ensure project root is on sys.path when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serve_model import app


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-s", __file__]))


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_endpoint_sample():
    client = TestClient(app)
    payload = {
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
    resp = client.post("/predict", json=payload)
    # Print response so it is visible with `pytest -s`
    print("predict status:", resp.status_code, "body:", resp.json())
    # If model is missing, we expect 503; otherwise 200
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "predicted_price" in body


