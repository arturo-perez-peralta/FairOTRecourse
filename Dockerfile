FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY fairopt/ ./fairopt/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e ".[viz]"

# Run all fairness experiments (F1, F2, F3) sequentially.
CMD ["python", "-m", "fairopt.experiments.run_all"]
