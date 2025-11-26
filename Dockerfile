FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (for building some ML libs like lightgbm)
RUN apt-get update && \
	apt-get install -y --no-install-recommends build-essential && \
	rm -rf /var/lib/apt/lists/*

# Create and use a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000
ENV CONFIG_PATH=/app/config.yaml

# Use entrypoint script that runs training pipeline then starts API
ENTRYPOINT ["/app/docker-entrypoint.sh"]