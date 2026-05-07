#!/usr/bin/env bash
# Railway entrypoint: uses seed data bundled in the repo, or a persistent
# volume if one is mounted.
#
# Without a volume, the app runs off seed/ data (read-only — new ETL data
# won't persist across deploys, but the dashboard works fine).
#
# With a volume mounted at /data, seed files are copied there on first boot
# and all writes persist.

set -e

DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-}"

if [ -n "$DATA_DIR" ]; then
    echo "Persistent volume detected at $DATA_DIR"
    mkdir -p "$DATA_DIR/models"

    if [ -d "seed/models" ] && [ -z "$(ls -A "$DATA_DIR/models" 2>/dev/null)" ]; then
        echo "Seeding models from seed/models/ -> $DATA_DIR/models/"
        cp seed/models/*.joblib "$DATA_DIR/models/"
    fi

    if [ -f "seed/nba.db" ] && [ ! -f "$DATA_DIR/nba.db" ]; then
        echo "Seeding database from seed/nba.db -> $DATA_DIR/nba.db"
        cp seed/nba.db "$DATA_DIR/nba.db"
    fi
else
    echo "No persistent volume — using seed/ data directly"
    # Point env vars at seed data if not already set
    export NBA_DATABASE_URL="${NBA_DATABASE_URL:-sqlite:///seed/nba.db}"
    export NBA_MODELS_DIR="${NBA_MODELS_DIR:-seed/models}"
fi

echo "Starting gunicorn on port $PORT..."
exec gunicorn webapp.app:app \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 2 \
    --timeout 120
