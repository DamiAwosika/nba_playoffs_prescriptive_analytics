"""Flask dashboard for the NBA playoff predictor.

Renders an interactive bracket. Click a series -> modal showing model
predictions, Vegas reference, and a side-by-side stat comparison.

Run with:
    python -m webapp.app
"""
from __future__ import annotations
from flask import Flask, jsonify, render_template

from webapp.data import (
    get_bracket, get_head_to_head, get_player_predictions,
    get_series_predictions, get_team_detail, get_team_roster,
)

app = Flask(__name__)


@app.route("/")
def index():
    bracket = get_bracket()
    return render_template("index.html", bracket=bracket)


@app.route("/api/series/<int:team1_id>/<int:team2_id>")
def api_series(team1_id: int, team2_id: int):
    return jsonify(get_series_predictions(team1_id, team2_id))


@app.route("/api/team-detail/<int:team1_id>/<int:team2_id>")
def api_team_detail(team1_id: int, team2_id: int):
    return jsonify(get_team_detail(team1_id, team2_id))


@app.route("/api/team-roster/<int:team_id>")
def api_team_roster(team_id: int):
    return jsonify(get_team_roster(team_id))


@app.route("/api/head-to-head/<int:team1_id>/<int:team2_id>")
def api_head_to_head(team1_id: int, team2_id: int):
    return jsonify(get_head_to_head(team1_id, team2_id))


@app.route("/api/player-predictions/<int:team1_id>/<int:team2_id>")
def api_player_predictions(team1_id: int, team2_id: int):
    return jsonify(get_player_predictions(team1_id, team2_id))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
