#!/usr/bin/env bash
# Railway entrypoint: ensures DB + models exist on the persistent volume,
# then starts gunicorn.

set -e

DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"

mkdir -p "$DATA_DIR/models"

if [ -d "seed/models" ] && [ -z "$(ls -A "$DATA_DIR/models" 2>/dev/null)" ]; then
    echo "Seeding models from seed/models/ -> $DATA_DIR/models/"
    cp seed/models/*.joblib "$DATA_DIR/models/"
fi

if [ -f "seed/nba.db" ] && [ ! -f "$DATA_DIR/nba.db" ]; then
    echo "Seeding database from seed/nba.db -> $DATA_DIR/nba.db"
    cp seed/nba.db "$DATA_DIR/nba.db"
fi

echo "Starting gunicorn on port $PORT..."
exec gunicorn webapp.app:app \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 2 \
    --timeout 120
