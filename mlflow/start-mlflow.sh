#!/usr/bin/env bash
set -euo pipefail

# Simple local MLflow runner (no Docker). Creates venv and runs mlflow server.
VENV_DIR=".venv"

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install --no-cache-dir "mlflow[extras]"

mkdir -p mlflow/artifacts
DB_FILE=mlflow/mlflow.db
if [ ! -f "$DB_FILE" ]; then
  mkdir -p "$(dirname "$DB_FILE")"
  python - <<PY
import sqlite3
conn = sqlite3.connect('$DB_FILE')
conn.execute('PRAGMA user_version = 1')
conn.commit()
conn.close()
PY
fi

echo "Starting MLflow UI at http://127.0.0.1:5000"
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts
