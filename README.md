# NBA Playoffs Predictor

A full-stack ML platform that predicts NBA playoff outcomes and individual player stats. Features an interactive bracket dashboard with real-time model predictions, head-to-head analysis, and roster breakdowns.

## What It Does

- **Game outcome predictions** — Logistic regression, random forest, XGBoost, stacked ensemble, and voting ensemble models predict win probability for each playoff matchup
- **Player stat predictions** — Per-player PTS, REB, AST, STL, BLK, and TO predictions using Ridge, random forest, and XGBoost regression models
- **Vegas integration** — Displays Vegas odds as a reference line and supports vegas-augmented model variants trained on archived pre-game prop lines
- **Interactive bracket** — Click any series to see model predictions, stat comparisons, head-to-head regular season history, full rosters with advanced stats (TS%, Game Score), and player projections

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Flask + Jinja2, vanilla JS, CSS |
| Backend | Python 3.10+, Flask |
| Database | SQLAlchemy + SQLite |
| ML | scikit-learn, XGBoost |
| Data Source | [balldontlie API](https://www.balldontlie.io/) |

## Local Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/nba-predict.git
cd nba-predict
pip install -e .

# Set your API key
echo NBA_BALLDONTLIE_API_KEY=your-key-here > .env

# Ingest data
python scripts/run_etl.py

# Train models
python scripts/train_vegas.py
python scripts/train_player_models.py --tune

# Launch dashboard
python -m webapp.app
```

Open [http://localhost:5000](http://localhost:5000)

## Daily Operations

```bash
# Ingest recent game results
python scripts/run_etl.py

# Archive pre-game odds and player props (run before tipoff)
python scripts/archive_odds.py
python scripts/archive_player_props.py

# Refresh playoff predictions
python scripts/predict_playoffs.py
```

## Project Structure

```
nba-predict/
├── nba_ml/                  # Core ML package
│   ├── db/                  # SQLAlchemy models + session
│   ├── ingest/              # ETL: balldontlie API client + loaders
│   ├── features/            # Feature engineering (team + player)
│   ├── training/            # Model training scripts
│   └── inference/           # Prediction: matchup + player models
├── webapp/                  # Flask dashboard
│   ├── app.py               # Routes
│   ├── data.py              # Data layer (queries + predictions)
│   ├── templates/           # Jinja2 HTML
│   └── static/              # CSS + JS
├── scripts/                 # CLI scripts (ETL, training, archival)
├── models/                  # Trained model bundles (.joblib)
└── DEPLOY.md                # Railway deployment guide
```

## Deployment

See [DEPLOY.md](DEPLOY.md) for step-by-step Railway deployment instructions.

## License

Private project. All rights reserved.
