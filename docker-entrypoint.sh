#!/bin/bash
set -e

echo "🚀 Starting ML pipeline initialization..."

# Change to app directory
cd /app

# Step 1: Train models
echo "📊 Step 1/4: Training models..."
python src/train_model.py

# Step 2: Register model
echo "📝 Step 2/4: Registering model..."
python src/register_model.py

# Step 3: Promote model
echo "⭐ Step 3/4: Promoting model to Production..."
python src/promote_model.py

# Step 4: Start API server
echo "🌐 Step 4/4: Starting API server..."
exec uvicorn src.serve_model:app --host 0.0.0.0 --port 8000

