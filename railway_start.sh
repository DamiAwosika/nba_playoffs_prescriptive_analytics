#!/usr/bin/env bash
# Railway entrypoint: always refreshes DB + models from seed/ on every deploy,
# then starts gunicorn.

set -e

DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"

mkdir -p "$DATA_DIR/models"

if [ -d "seed/models" ]; then
    echo "Refreshing models from seed/models/ -> $DATA_DIR/models/"
    cp seed/models/*.joblib "$DATA_DIR/models/"
fi

if [ -f "seed/nba.db" ]; then
    echo "Refreshing database from seed/nba.db -> $DATA_DIR/nba.db"
    cp seed/nba.db "$DATA_DIR/nba.db"
fi

echo "Starting gunicorn on port $PORT..."
exec gunicorn webapp.app:app \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 2 \
    --timeout 120
