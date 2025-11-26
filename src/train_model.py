import os
import json
import joblib
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
import yaml

# ============================================================
# 1️⃣ Setup (from config)
# ============================================================
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

# ============================================================
# 2️⃣ Feature definitions
# ============================================================
numeric_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
categorical_features = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"
]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

joblib.dump(numeric_transformer, "artifacts/scalers/scaler.pkl")
joblib.dump(categorical_transformer, "artifacts/encoders/encoder.pkl")

# ============================================================
# 3️⃣ Load data
# ============================================================
processed_path = cfg["paths"]["processed_data"]
df = pd.read_csv(processed_path)
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save numeric training stats for drift monitoring (store raw samples)
train_stats_path = cfg["paths"]["train_stats"]
os.makedirs(os.path.dirname(train_stats_path), exist_ok=True)
X_train[numeric_features].to_csv(train_stats_path, index=False)

# ============================================================
# 4️⃣ Define base models
# ============================================================
models = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=500, max_depth=10, random_state=42
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
    )
}

metadata_list = []
best_r2 = -np.inf
best_run_id = None
best_model_name = None

# ============================================================
# 5️⃣ Train & Log to MLflow
# ============================================================
for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    with mlflow.start_run(run_name=model_name) as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"--- {model_name} ---")
        print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}\n")

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})
        mlflow.sklearn.log_model(pipeline, "model")

        # Save locally
        models_dir = cfg["paths"]["models_dir"]
        os.makedirs(models_dir, exist_ok=True)
        pipeline_path = os.path.join(models_dir, f"{model_name}_pipeline.pkl")
        joblib.dump(pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path)

        # Metadata
        metadata = {
            "model_name": model_name,
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": numeric_features + categorical_features,
            "hyperparameters": model.get_params(),
            "metrics": {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2},
            "run_id": run.info.run_id
        }
        metadata_list.append(metadata)

        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name
            best_run_id = run.info.run_id

# Save metadata
with open("artifacts/models/all_models_metadata.json", "w") as f:
    json.dump(metadata_list, f, indent=4)

# ============================================================
# 6️⃣ Optuna tuning (Gradient Boosting)
# ============================================================
print("\n🔍 Starting Optuna tuning for Gradient Boosting...")

X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "random_state": 42
    }

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(**params))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
    return -np.mean(r2_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_trial.params
print("✅ Best trial parameters:", best_params)
print(f"Best (negative R²): {study.best_value:.4f}")

# ============================================================
# 7️⃣ Retrain tuned Gradient Boosting & Log
# ============================================================
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(**best_params))
])

with mlflow.start_run(run_name="GradientBoosting_Optuna") as run:
    final_pipeline.fit(X_train, y_train)
    y_pred_test = final_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    mlflow.log_params(best_params)
    mlflow.log_metrics({"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})
    mlflow.sklearn.log_model(final_pipeline, "model")

    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    pipeline_path = os.path.join(models_dir, "GradientBoosting_optuna_pipeline.pkl")
    joblib.dump(final_pipeline, pipeline_path)
    mlflow.log_artifact(pipeline_path)

    metadata = {
        "model_name": "GradientBoosting_Optuna",
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": numeric_features + categorical_features,
        "hyperparameters": best_params,
        "metrics": {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2},
        "run_id": run.info.run_id
    }

    metadata_list.append(metadata)
    with open(os.path.join(cfg["paths"]["models_dir"], "all_models_metadata.json"), "w") as f:
        json.dump(metadata_list, f, indent=4)

    # Update "best model" if Optuna-tuned is better
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = "GradientBoosting_Optuna"
        best_run_id = run.info.run_id

# ============================================================
# 8️⃣ Save summary of best model for registration
# ============================================================
best_metadata = {
    "best_model": best_model_name,
    "best_run_id": best_run_id,
    "best_r2": best_r2,
    "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(cfg["paths"]["models_dir"], "best_model.json"), "w") as f:
    json.dump(best_metadata, f, indent=4)

print(f"\n🏆 Best model: {best_model_name} (R²={best_r2:.3f})")
print(f"🔖 Run ID: {best_run_id}")
print("🎯 All models trained, tuned, and logged successfully to MLflow.")