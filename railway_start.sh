#!/usr/bin/env bash
# Railway entrypoint: ensures DB + models exist on the persistent volume,
# then starts gunicorn.
#
# Railway volume is mounted at /data (configured in Railway dashboard).
# Environment variables set in Railway:
#   NBA_DATABASE_URL=sqlite:////data/nba.db
#   NBA_MODELS_DIR=/data/models
#   NBA_BALLDONTLIE_API_KEY=<your key>

set -e

DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"

# Create models dir on persistent volume if it doesn't exist yet.
mkdir -p "$DATA_DIR/models"

# If this is a fresh deploy and seed files were bundled, copy them over.
# (You upload nba.db + models/ once via `railway volume` or SCP.)
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
