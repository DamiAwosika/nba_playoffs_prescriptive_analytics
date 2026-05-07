"""Daily scheduled job for Railway cron service.

Runs the ETL pipeline, archives odds + player props, and refreshes
predictions. Intended to run once daily before games tip off.

Railway cron schedule: 0 16 * * *  (4 PM UTC = ~12 PM ET)
"""
from __future__ import annotations
from datetime import date


def main() -> None:
    print(f"=== daily cron: {date.today()} ===")

    from nba_ml.db.base import init_db
    init_db()

    print("\n--- 1/3  ETL: ingest games ---")
    from datetime import timedelta
    from scripts.run_etl import main as run_etl
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    run_etl(start, end)

    print("\n--- 2/3  archive odds ---")
    from scripts.archive_odds import main as archive_odds
    archive_odds(days_ahead=3)

    print("\n--- 3/3  archive player props ---")
    from scripts.archive_player_props import main as archive_props
    archive_props(days_ahead=5)

    print(f"\n=== daily cron complete: {date.today()} ===")


if __name__ == "__main__":
    main()
