# NBA Predictive ML Backend

## 1. Project Context

This is a production-ready MLOps prototype designed to predict NBA game outcomes. The primary focus is on robust software engineering, modular
backend design, and defensive statistical learning practices. It is not just a Jupyter notebook; it is a full pipeline handling ingestion, feature engineering, and inference.

## 2. Tech Stack & Formatting

- **Language:** Python 3.10+
- **Database:** SQLAlchemy (SQLite for local dev, structured for easy PostgreSQL migration)
- **API:** FastAPI
- **ML Framework:** scikit-learn (default) / XGBoost (advanced)
- **Data Manipulation:** pandas, numpy
- **Style:** Use PEP 8 conventions, type hinting (`def get_team(team_id: int) -> dict:`), and descriptive variable names.

## 3. Critical Guardrails (NEVER VIOLATE)

- **Temporal Integrity (No Data Leakage):** NBA data is highly temporal. When engineering features (e.g., rolling averages, win streaks), you MUST ensure the calculation only includes data available _prior_ to the tip-off of the target game. Never include the target game's stats in its own predictive features. Use strictly shifted windows.
- **Idempotent ETL:** All data ingestion scripts must be idempotent. If an ingestion script is run twice on the same day, it must update or ignore existing records without creating duplicates. Rely on strict database constraints (e.g., unique constraints on `game_id`).
- **Modular Feature Store:** Maintain a strict architectural separation between Raw Data and Engineered Features.
  - `Game` models/tables should map 1:1 with raw box score/play-by-play data.
  - `TeamGameFeature` models/tables should contain the derived, time-lagged metrics used for actual model training. Do not mix them.

## 4. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?"
If yes, simplify.

## 5. Common Commands

_(Note: Update these as the project evolves)_

- **Run API:** `uvicorn app.main:app --reload`
- **Run ETL Pipeline:** `python scripts/ingest_games.py`
- **Run Tests:** `pytest tests/`
