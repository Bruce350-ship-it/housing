import os
import json
import joblib
import uvicorn
from typing import Optional, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import logging
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware import Middleware
from starlette.responses import Response

# Configure structured logging early
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("serve_model")

# ===============================
# 0. Load configuration
# ===============================
def load_config(config_path: str = "config.yaml") -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
MODEL_PATH = cfg["paths"]["best_model_pickle"]

# ===============================
# 1. Load trained pipeline (with error handling)
# ===============================
pipeline = None
try:
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        logger.info("Model pipeline loaded | path=%s", MODEL_PATH)
    else:
        logger.warning("Model path does not exist | path=%s", MODEL_PATH)
except Exception as exc:
    pipeline = None  # will be surfaced in health/predict
    logger.exception("Failed to load model pipeline | path=%s", MODEL_PATH)

# ===============================
# 2. Define request body with validation
# ===============================
YesNo = Literal["yes", "no"]
Furnishing = Literal["unfurnished", "semi-furnished", "furnished"]

class HousingData(BaseModel):
    area: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=0)
    bathrooms: int = Field(..., ge=0)
    stories: int = Field(..., ge=0)
    parking: int = Field(..., ge=0)
    mainroad: YesNo = "no"
    guestroom: YesNo = "no"
    basement: YesNo = "no"
    hotwaterheating: YesNo = "no"
    airconditioning: YesNo = "no"
    prefarea: YesNo = "no"
    furnishingstatus: Furnishing = "unfurnished"

# ===============================
# 3. Initialize FastAPI
# ===============================
app = FastAPI(
    title="Housing Price Prediction API",
    description="Predict housing prices using a tuned GradientBoosting model",
    version="1.1"
)

# Security headers middleware (simple)
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Note: For strict environments add Content-Security-Policy tailored to your frontend
    return response

# CORS (locked down by default; adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.get("cors", {}).get("allow_origins", []),
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    payload = {
        "name": "Housing Price Prediction API",
        "version": "1.1",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }
    return payload

# ===============================
# 4. Prediction endpoint
# ===============================
@app.post("/predict")
def predict(data: HousingData, request: Request):
    if pipeline is None:
        logger.warning("Predict called but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = pipeline.predict(df)[0]
        response = {"predicted_price": round(float(prediction), 2)}
        logger.info(
            "predict ok | area=%s bedrooms=%s bathrooms=%s stories=%s parking=%s | pred=%.2f | ua=%s",
            data.area,
            data.bedrooms,
            data.bathrooms,
            data.stories,
            data.parking,
            response["predicted_price"],
            request.headers.get("user-agent", "-"),
        )
        return response
    except Exception as exc:
        logger.exception("predict error during inference")
        raise HTTPException(status_code=400, detail=str(exc))

# ===============================
# 5. Health check endpoint
# ===============================
@app.get("/health")
def health_check():
    status = "alive" if pipeline is not None else "degraded"
    return {"status": status, "model_loaded": pipeline is not None}

# ===============================
# 6. Run server
# ===============================
if __name__ == "__main__":
    uvicorn.run(app, host=cfg["api"]["host"], port=int(cfg["api"]["port"]))