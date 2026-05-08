// ===== Bracket interactivity + connecting lines =====
//
// On load:
//   1. Wire up click handlers for series cards (open modal).
//   2. Draw SVG connecting lines between series boxes.
//   3. Re-draw lines on window resize.

const modal = document.getElementById("modal");
const modalBody = document.getElementById("modal-body");
const svg = document.getElementById("bracket-svg");

function jsonOrError(r) {
    if (!r.ok) throw new Error(`Server error (${r.status})`);
    return r.json();
}
const SVG_NS = "http://www.w3.org/2000/svg";

// ----- Click handlers (skip TBD placeholders — no team to look up) -----
document.querySelectorAll(".series-card:not(.placeholder)").forEach((card) => {
    card.addEventListener("click", () => openSeries(card));
});

document.querySelector(".close-btn").addEventListener("click", closeModal);
modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
});
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
});

// Re-draw bracket lines after layout settles + on every resize.
window.addEventListener("load", drawBracketLines);
window.addEventListener("resize", drawBracketLines);

// ===== Modal lifecycle =====

function openSeries(card) {
    const t1 = card.dataset.team1Id;
    const t2 = card.dataset.team2Id;
    const t1abbr = card.dataset.team1Abbr;
    const t2abbr = card.dataset.team2Abbr;

    modalBody.innerHTML = `<p class="loading">Loading ${t1abbr} vs ${t2abbr}…</p>`;
    modal.classList.remove("hidden");

    fetch(`/api/team-detail/${t1}/${t2}`)
        .then(jsonOrError)
        .then((data) => renderModal(data))
        .catch((err) => {
            modalBody.innerHTML = `<p class="loading">Error loading data: ${err}</p>`;
        });
}

function closeModal() { modal.classList.add("hidden"); }

// ===== Modal rendering =====

function renderModal(data) {
    const { team1, team2, team1_playoff_stats, team2_playoff_stats,
            team1_features, team2_features, preds,
            prediction_context, prediction_date, series_status } = data;

    const subtitle = prediction_context === "deciding_game"
        ? `Predictions for the deciding game of this completed series (using stats as of ${prediction_date})`
        : (series_status === "in_progress"
            ? `Predictions for the next game in this ongoing series (using current stats)`
            : `Predictions for Game 1 of this matchup (using current stats)`);

    modalBody.innerHTML = `
        <h2 class="modal-title">${team1.full_name} vs ${team2.full_name}</h2>
        <p class="modal-subtitle">${subtitle}</p>

        ${renderPredictions(team1, team2, preds)}

        <h3 class="section-title">Playoff averages (this postseason)</h3>
        ${renderComparison(team1, team2, [
            ["PTS",         team1_playoff_stats.pts,         team2_playoff_stats.pts],
            ["REB",         team1_playoff_stats.reb,         team2_playoff_stats.reb],
            ["AST",         team1_playoff_stats.ast,         team2_playoff_stats.ast],
            ["PTS Allowed", team1_playoff_stats.pts_allowed, team2_playoff_stats.pts_allowed],
        ])}

        <h3 class="section-title">Model features (latest snapshot)</h3>
        ${renderComparison(team1, team2, [
            ["Off Rating (5g)",  team1_features.off_rating_roll5,  team2_features.off_rating_roll5],
            ["Def Rating (5g)",  team1_features.def_rating_roll5,  team2_features.def_rating_roll5],
            ["Net Rating (10g)", team1_features.net_rating_roll10, team2_features.net_rating_roll10],
            ["Win % (10g)",      team1_features.win_pct_roll10,    team2_features.win_pct_roll10],
            ["Pace (5g)",        team1_features.pace_roll5,        team2_features.pace_roll5],
            ["Elo Rating",       team1_features.elo,               team2_features.elo],
        ])}

        <h3 class="section-title">Rosters</h3>
        <div class="roster-buttons">
            <button class="roster-btn" data-team-id="${team1.team_id}" data-team-abbr="${team1.abbreviation}">
                View ${team1.abbreviation} roster
            </button>
            <button class="roster-btn" data-team-id="${team2.team_id}" data-team-abbr="${team2.abbreviation}">
                View ${team2.abbreviation} roster
            </button>
        </div>
        <div id="roster-panel"></div>

        <h3 class="section-title">Head-to-head (regular season)</h3>
        <div class="roster-buttons">
            <button class="roster-btn" id="h2h-btn"
                    data-t1="${team1.team_id}" data-t2="${team2.team_id}"
                    data-t1abbr="${team1.abbreviation}" data-t2abbr="${team2.abbreviation}">
                View ${team1.abbreviation} vs ${team2.abbreviation} season series
            </button>
        </div>
        <div id="h2h-panel"></div>

        <h3 class="section-title">Player stat predictions (PTS / REB / AST / STL / BLK / TO)</h3>
        <div class="roster-buttons">
            <button class="roster-btn" id="player-preds-btn"
                    data-t1="${team1.team_id}" data-t2="${team2.team_id}"
                    data-t1abbr="${team1.abbreviation}" data-t2abbr="${team2.abbreviation}">
                View player predictions
            </button>
        </div>
        <div id="player-preds-panel"></div>
    `;

    modalBody.querySelectorAll(".roster-btn").forEach((btn) => {
        if (btn.id === "h2h-btn" || btn.id === "player-preds-btn") return;
        btn.addEventListener("click", () => loadRoster(btn.dataset.teamId, btn.dataset.teamAbbr));
    });
    const h2hBtn = modalBody.querySelector("#h2h-btn");
    if (h2hBtn) {
        h2hBtn.addEventListener("click", () => loadHeadToHead(
            h2hBtn.dataset.t1, h2hBtn.dataset.t2,
            h2hBtn.dataset.t1abbr, h2hBtn.dataset.t2abbr,
        ));
    }
    const ppBtn = modalBody.querySelector("#player-preds-btn");
    if (ppBtn) {
        ppBtn.addEventListener("click", () => loadPlayerPredictions(
            ppBtn.dataset.t1, ppBtn.dataset.t2,
            ppBtn.dataset.t1abbr, ppBtn.dataset.t2abbr,
        ));
    }
}

function loadHeadToHead(t1, t2, t1abbr, t2abbr) {
    const panel = document.getElementById("h2h-panel");
    panel.innerHTML = `<p class="loading">Loading ${t1abbr} vs ${t2abbr} regular-season series…</p>`;
    fetch(`/api/head-to-head/${t1}/${t2}`)
        .then(jsonOrError)
        .then((data) => panel.innerHTML = renderHeadToHead(data))
        .catch((err) => panel.innerHTML = `<p class="loading">Error: ${err}</p>`);
}

function renderHeadToHead(data) {
    const { team1, team2, n_games, team1_wins, team2_wins,
            team1_avgs, team2_avgs, team1_players, team2_players } = data;

    if (n_games === 0) {
        return `<p class="loading">No regular-season meetings between ${team1.abbreviation} and ${team2.abbreviation} this season.</p>`;
    }

    const seriesLine = team1_wins === team2_wins
        ? `Season series tied ${team1_wins}-${team2_wins} over ${n_games} game${n_games === 1 ? '' : 's'}`
        : (team1_wins > team2_wins
            ? `${team1.abbreviation} won the season series ${team1_wins}-${team2_wins}`
            : `${team2.abbreviation} won the season series ${team2_wins}-${team1_wins}`);

    const teamCompare = renderComparison(team1, team2, [
        ["PTS",         team1_avgs.pts,         team2_avgs.pts],
        ["PTS Allowed", team1_avgs.pts_allowed, team2_avgs.pts_allowed],
        ["REB",         team1_avgs.reb,         team2_avgs.reb],
        ["AST",         team1_avgs.ast,         team2_avgs.ast],
    ]);

    return `
        <p class="h2h-summary">${seriesLine}</p>
        <h4 class="subsection-title">Team averages in these games</h4>
        ${teamCompare}
        <h4 class="subsection-title">${team1.abbreviation} player averages vs ${team2.abbreviation}</h4>
        ${renderH2HPlayerTable(team1_players, team1.abbreviation)}
        <h4 class="subsection-title">${team2.abbreviation} player averages vs ${team1.abbreviation}</h4>
        ${renderH2HPlayerTable(team2_players, team2.abbreviation)}
    `;
}

function renderH2HPlayerTable(players, abbr) {
    if (!players || players.length === 0) {
        return `<p class="loading">No ${abbr} player data for these games.</p>`;
    }
    const rows = players.map((p) => `
        <tr>
            <td>${p.name}</td>
            <td class="value">${p.games}</td>
            <td class="value">${p.mpg}</td>
            <td class="value">${p.ppg}</td>
            <td class="value">${p.rpg}</td>
            <td class="value">${p.apg}</td>
            <td class="value">${p.spg}</td>
            <td class="value">${p.bpg}</td>
            <td class="value">${p.topg}</td>
            <td class="value ${p.plus_minus >= 0 ? 'pm-pos' : 'pm-neg'}">${p.plus_minus > 0 ? '+' : ''}${p.plus_minus}</td>
            <td class="value ts-gold">${p.ts_pct !== null ? (p.ts_pct * 100).toFixed(1) + '%' : '—'}</td>
            <td class="value game-score">${p.game_score}</td>
        </tr>
    `).join("");
    return `
        <table class="roster-table">
            <thead>
                <tr>
                    <th>Player</th>
                    <th class="value">G</th>
                    <th class="value">MIN</th>
                    <th class="value">PTS</th>
                    <th class="value">REB</th>
                    <th class="value">AST</th>
                    <th class="value">STL</th>
                    <th class="value">BLK</th>
                    <th class="value">TO</th>
                    <th class="value">+/-</th>
                    <th class="value" title="True Shooting %">TS%</th>
                    <th class="value" title="Hollinger Game Score per game">GmSc</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function loadPlayerPredictions(t1, t2, t1abbr, t2abbr) {
    const panel = document.getElementById("player-preds-panel");
    panel.innerHTML = `<p class="loading">Loading player predictions for ${t1abbr} vs ${t2abbr}… (this may take a moment)</p>`;
    fetch(`/api/player-predictions/${t1}/${t2}`)
        .then(jsonOrError)
        .then((data) => panel.innerHTML = renderPlayerPredictions(data))
        .catch((err) => panel.innerHTML = `<p class="loading">Error: ${err}</p>`);
}

function renderPlayerPredictions(data) {
    const { team1, team2, team1_players, team2_players,
            prediction_context, prediction_date } = data;

    const contextLabel = prediction_context === "last_game"
        ? `Predictions for the last game of this series (as of ${prediction_date})`
        : `Predictions for the next game (${prediction_date})`;

    if (!team1_players.length && !team2_players.length) {
        return `<p class="loading">No player models available yet. Train models first with <code>python scripts/train_player_models.py</code></p>`;
    }

    return `
        <p class="player-preds-context">${contextLabel}</p>
        <h4 class="subsection-title">${team1.abbreviation} predicted stats</h4>
        ${renderPlayerPredTable(team1_players, team1.abbreviation)}
        <h4 class="subsection-title">${team2.abbreviation} predicted stats</h4>
        ${renderPlayerPredTable(team2_players, team2.abbreviation)}
    `;
}

function renderPlayerPredTable(players, abbr) {
    if (!players || players.length === 0) {
        return `<p class="loading">No predictions available for ${abbr} players (insufficient history).</p>`;
    }
    const rows = players.map((p) => {
        const props = p.prop_lines || {};
        const usesVegas = Object.values(p.variants || {}).some(v => v === "vegas");
        const out = p.active === false;
        const cls = out ? ' class="player-inactive"' : '';
        const dash = '<span class="inactive-dash">—</span>';
        return `
        <tr${cls}>
            <td class="player-name-cell">
                ${p.name}
                ${out ? '<span class="inactive-tag">OUT</span>' : ''}
                ${!out && usesVegas ? '<span class="vegas-badge" title="Vegas-augmented model used">V</span>' : ''}
            </td>
            <td class="value ${out ? '' : 'pred-value'}">${out ? dash : fmtNum(p.pts)}</td>
            <td class="value ${out ? '' : 'pred-detail'}">${out ? dash : (props.pts != null ? fmtNum(props.pts) : '—')}</td>
            <td class="value ${out ? '' : 'pred-value'}">${out ? dash : fmtNum(p.reb)}</td>
            <td class="value ${out ? '' : 'pred-detail'}">${out ? dash : (props.reb != null ? fmtNum(props.reb) : '—')}</td>
            <td class="value ${out ? '' : 'pred-value'}">${out ? dash : fmtNum(p.ast)}</td>
            <td class="value ${out ? '' : 'pred-detail'}">${out ? dash : (props.ast != null ? fmtNum(props.ast) : '—')}</td>
            <td class="value ${out ? '' : 'pred-value'}">${out ? dash : fmtNum(p.stl)}</td>
            <td class="value ${out ? '' : 'pred-detail'}">${out ? dash : (props.stl != null ? fmtNum(props.stl) : '—')}</td>
            <td class="value ${out ? '' : 'pred-value'}">${out ? dash : fmtNum(p.blk)}</td>
            <td class="value ${out ? '' : 'pred-detail'}">${out ? dash : (props.blk != null ? fmtNum(props.blk) : '—')}</td>
            <td class="value ${out ? '' : 'pred-value'}">${out ? dash : fmtNum(p.tov)}</td>
            <td class="value ${out ? '' : 'pred-detail'}">${out ? dash : (props.tov != null ? fmtNum(props.tov) : '—')}</td>
        </tr>
    `;
    }).join("");
    const hasInactive = players.some(p => p.active === false);
    return `
        <table class="roster-table player-preds-table">
            <thead>
                <tr>
                    <th>${abbr} — Player</th>
                    <th class="value">PTS</th>
                    <th class="value" title="Vegas PTS prop line">PTS Line</th>
                    <th class="value">REB</th>
                    <th class="value" title="Vegas REB prop line">REB Line</th>
                    <th class="value">AST</th>
                    <th class="value" title="Vegas AST prop line">AST Line</th>
                    <th class="value">STL</th>
                    <th class="value" title="Vegas STL prop line">STL Line</th>
                    <th class="value">BLK</th>
                    <th class="value" title="Vegas BLK prop line">BLK Line</th>
                    <th class="value">TO</th>
                    <th class="value" title="Vegas TO prop line">TO Line</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
        <p class="roster-note">Predicted stats = model ensemble. Line = Vegas prop (if available). <span class="vegas-badge">V</span> = vegas-augmented model used.${hasInactive ? ' <span class="inactive-tag">OUT</span> = unavailable for this game (injury report).' : ''}</p>
    `;
}

function loadRoster(teamId, abbr) {
    const panel = document.getElementById("roster-panel");
    panel.innerHTML = `<p class="loading">Loading ${abbr} roster…</p>`;
    fetch(`/api/team-roster/${teamId}`)
        .then(jsonOrError)
        .then((data) => panel.innerHTML = renderRoster(data, abbr))
        .catch((err) => panel.innerHTML = `<p class="loading">Error: ${err}</p>`);
}

function renderRoster(data, abbr) {
    const players = data.players || [];
    if (players.length === 0) {
        return `<p class="loading">No games found for ${abbr} this season.</p>`;
    }
    const rows = players.map((p) => `
        <tr>
            <td>${p.name}</td>
            <td class="value">${p.games}</td>
            <td class="value">${p.mpg}</td>
            <td class="value">${p.ppg}</td>
            <td class="value">${p.rpg}</td>
            <td class="value">${p.apg}</td>
            <td class="value">${p.spg}</td>
            <td class="value">${p.bpg}</td>
            <td class="value">${p.topg}</td>
            <td class="value ${p.plus_minus >= 0 ? 'pm-pos' : 'pm-neg'}">${p.plus_minus > 0 ? '+' : ''}${p.plus_minus}</td>
            <td class="value ts-gold">${p.ts_pct !== null ? (p.ts_pct * 100).toFixed(1) + '%' : '—'}</td>
            <td class="value game-score">${p.game_score}</td>
        </tr>
    `).join("");
    return `
        <table class="roster-table">
            <thead>
                <tr>
                    <th>${abbr} — Player</th>
                    <th class="value">G</th>
                    <th class="value">MIN</th>
                    <th class="value">PTS</th>
                    <th class="value">REB</th>
                    <th class="value">AST</th>
                    <th class="value">STL</th>
                    <th class="value">BLK</th>
                    <th class="value">TO</th>
                    <th class="value">+/-</th>
                    <th class="value" title="True Shooting % = PTS / (2*(FGA + 0.44*FTA))">TS%</th>
                    <th class="value" title="Hollinger Game Score per game (per-game equivalent of PER)">GmSc</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
        <p class="roster-note">GmSc = Hollinger Game Score per game (per-game equivalent of PER, ≥10 is solid, ≥20 is elite).</p>
    `;
}

function renderPredictions(team1, team2, preds) {
    const t1home = preds?.team1_at_home || {predictions: {}, vegas_home_win_prob: null};
    const t2home = preds?.team2_at_home || {predictions: {}, vegas_home_win_prob: null};
    const allNames = Array.from(new Set([
        ...Object.keys(t1home.predictions || {}),
        ...Object.keys(t2home.predictions || {}),
    ]));
    const HIGHLIGHTED = new Set(["ensemble", "stack"]);
    const modelNames = [
        ...allNames.filter(n => !HIGHLIGHTED.has(n)),
        ...allNames.filter(n => HIGHLIGHTED.has(n)),
    ];

    const rows = modelNames.map((name) => {
        const cls = HIGHLIGHTED.has(name) ? " class=\"model-highlight\"" : "";
        return `
        <tr${cls}>
            <td>${name}</td>
            <td class="value">${fmtPct(t1home.predictions[name])}</td>
            <td class="value">${fmtPct(t2home.predictions[name])}</td>
        </tr>
    `;
    }).join("");

    const vegasRow = `
        <tr>
            <td class="highlight"><strong>VEGAS_REF</strong></td>
            <td class="value highlight">${fmtPct(t1home.vegas_home_win_prob)}</td>
            <td class="value highlight">${fmtPct(t2home.vegas_home_win_prob)}</td>
        </tr>
    `;

    return `
        <table class="preds-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th style="text-align:right">P(${team1.abbreviation} wins) — ${team1.abbreviation} home</th>
                    <th style="text-align:right">P(${team2.abbreviation} wins) — ${team2.abbreviation} home</th>
                </tr>
            </thead>
            <tbody>${rows}${vegasRow}</tbody>
        </table>
    `;
}

// Side-by-side comparison: numbers on the OUTSIDE, bars meet at the centre label.
// Bar length is proportional to value/max(value1, value2); higher value → longer bar.
function renderComparison(team1, team2, rows) {
    const headers = `
        <div class="team-headers">
            <span class="name left">${team1.abbreviation}</span>
            <span></span>
            <span class="name right">${team2.abbreviation}</span>
        </div>
    `;

    const body = rows.map(([label, v1, v2]) => {
        const max = Math.max(numOrZero(v1), numOrZero(v2)) || 1;
        const w1 = (numOrZero(v1) / max) * 100;
        const w2 = (numOrZero(v2) / max) * 100;
        return `
            <div class="comp-row">
                <span class="comp-value left">${fmtNum(v1)}</span>
                <div class="comp-bar-track left">
                    <div class="comp-bar-fill" style="width:${w1}%"></div>
                </div>
                <div class="comp-label">${label}</div>
                <div class="comp-bar-track right">
                    <div class="comp-bar-fill" style="width:${w2}%"></div>
                </div>
                <span class="comp-value right">${fmtNum(v2)}</span>
            </div>
        `;
    }).join("");

    return `<div class="comparison">${headers}${body}</div>`;
}

// ===== Bracket connecting lines (SVG overlay) =====

function drawBracketLines() {
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const bracket = document.querySelector(".bracket");
    const bracketRect = bracket.getBoundingClientRect();
    // Account for the SVG's own offset (it's `inset` from the bracket's padding).
    const svgRect = svg.getBoundingClientRect();
    const offX = svgRect.left - bracketRect.left;
    const offY = svgRect.top - bracketRect.top;

    const sel = (key) => Array.from(
        document.querySelectorAll(`.round-col[data-round="${key}"] .series-card`),
    );

    const westR1 = sel("first_round_west");
    const westR2 = sel("conf_semis_west");
    const westR3 = sel("conf_finals_west");
    const eastR1 = sel("first_round_east");
    const eastR2 = sel("conf_semis_east");
    const eastR3 = sel("conf_finals_east");
    const finals = sel("finals");

    // West side: lines run LEFT-TO-RIGHT (outside-in).
    // Pair R1 cards into R2: R1[0,1] → R2[0]; R1[2,3] → R2[1].
    westR1.forEach((card, i) => {
        const target = westR2[Math.floor(i / 2)];
        if (target) drawL(rightCenter(card), leftCenter(target));
    });
    westR2.forEach((card) => {
        if (westR3[0]) drawL(rightCenter(card), leftCenter(westR3[0]));
    });
    if (westR3[0] && finals[0]) drawL(rightCenter(westR3[0]), leftCenter(finals[0]));

    // East side: mirrored — lines run RIGHT-TO-LEFT (outside-in from the right).
    eastR1.forEach((card, i) => {
        const target = eastR2[Math.floor(i / 2)];
        if (target) drawL(leftCenter(card), rightCenter(target));
    });
    eastR2.forEach((card) => {
        if (eastR3[0]) drawL(leftCenter(card), rightCenter(eastR3[0]));
    });
    if (eastR3[0] && finals[0]) drawL(leftCenter(eastR3[0]), rightCenter(finals[0]));

    // ----- helpers -----
    function rightCenter(el) {
        const r = el.getBoundingClientRect();
        return {
            x: r.right - bracketRect.left - offX,
            y: r.top + r.height / 2 - bracketRect.top - offY,
        };
    }
    function leftCenter(el) {
        const r = el.getBoundingClientRect();
        return {
            x: r.left - bracketRect.left - offX,
            y: r.top + r.height / 2 - bracketRect.top - offY,
        };
    }
    function drawL(from, to) {
        // Step-shape: out from source, vertical to target row, in to target.
        const midX = (from.x + to.x) / 2;
        const d = `M${from.x},${from.y} H${midX} V${to.y} H${to.x}`;
        const p = document.createElementNS(SVG_NS, "path");
        p.setAttribute("d", d);
        p.setAttribute("stroke", "rgba(139,148,158,0.45)");
        p.setAttribute("stroke-width", "1.5");
        p.setAttribute("fill", "none");
        svg.appendChild(p);
    }
}

// ===== formatters =====
function fmtPct(p) {
    if (p === null || p === undefined) return "—";
    return (p * 100).toFixed(1) + "%";
}
function fmtNum(x) {
    if (x === null || x === undefined) return "—";
    if (typeof x !== "number") return String(x);
    return Number.isInteger(x) ? x.toString() : x.toFixed(1);
}
function numOrZero(x) {
    return typeof x === "number" && !isNaN(x) ? Math.abs(x) : 0;
}
