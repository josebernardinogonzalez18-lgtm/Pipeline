"""STAGE 5 - Inferencia diaria, filtrado por confianza y generacion de logs.

Salidas:
  reports/predictions.csv : game_id, equipos, prob modelo, prob implicita, edge,
                            Kelly stake, confidence, drift_status, last_retrain
  reports/output.json     : metadatos + bandera de drift + timestamp de retrain
Odds: lee config/odds_today.json (commit manual opcional):
  {"games": [{"game_pk": 123, "home_odds": 1.85, "away_odds": 2.10}]}
Si no hay odds, usa placeholder 1.90/1.90 y marca odds_source="placeholder".
"""
from __future__ import annotations
import datetime as dt
import json
import os
import joblib
import numpy as np
import pandas as pd

from src.config_loader import load_config, ensure_parent
from src.kelly import fractional_kelly_stake
from src.stage2_features import build_team_game_log, add_rolling_features, compute_matchup_features

PLACEHOLDER_ODDS = 1.90
BANKROLL_UNITS = 100.0


def _load_odds(cfg: dict) -> tuple[dict[int, dict], str]:
    path = cfg.get("odds_file", "config/odds_today.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {int(g["game_pk"]): g for g in data.get("games", [])}, "config_file"
    return {}, "placeholder_1.90"


def run(cfg: dict) -> pd.DataFrame:
    bundle = joblib.load(cfg["paths"]["model"])
    games = pd.read_csv(cfg["paths"]["raw_games"])
    games["date"] = pd.to_datetime(games["date"])
    today = pd.Timestamp.now().normalize()
    today_games = games[games["date"] >= today]           # juegos de hoy (cualquier estado)
    if today_games.empty:
        today_games = games[games["date"] == games["date"].max()]

    log = add_rolling_features(build_team_game_log(games), cfg["features"]["rolling_windows"])
    feats = compute_matchup_features(today_games, log, cfg)
    if feats.empty:
        print("[stage5] sin juegos para hoy"); return pd.DataFrame()
    feats = feats.sort_values("date")

    X = feats[bundle["features"]]
    p_home = bundle["calibrator"].predict(bundle["model"].predict_proba(X)[:, 1])
    ece = bundle.get("ece_calibrated", 0.02)
    odds_map, odds_source = _load_odds(cfg)
    t_cfg = cfg["training"]

    rows = []
    for (_, f), p in zip(feats.iterrows(), p_home):
        od = odds_map.get(int(f["game_pk"]), {})
        h_odd = float(od.get("home_odds", PLACEHOLDER_ODDS))
        a_odd = float(od.get("away_odds", PLACEHOLDER_ODDS))
        h = fractional_kelly_stake(p, h_odd, BANKROLL_UNITS,
                                   t_cfg["kelly_fraction"], t_cfg["max_stake_fraction"])
        a = fractional_kelly_stake(1 - p, a_odd, BANKROLL_UNITS,
                                   t_cfg["kelly_fraction"], t_cfg["max_stake_fraction"])
        confidence = round(float(abs(p - 0.5) * 2 * max(0.0, 1 - ece * 10)), 4)
        bet = "HOME" if h["edge"] >= max(a["edge"], t_cfg["min_edge_threshold"]) else               "AWAY" if a["edge"] >= t_cfg["min_edge_threshold"] else "NO_BET"
        rows.append({
            "game_pk": int(f["game_pk"]), "date": str(f["date"].date()),
            "home_team": f["home_team"], "away_team": f["away_team"],
            "model_prob_home": round(float(p), 4),
            "model_prob_away": round(float(1 - p), 4),
            "home_odds": h_odd, "away_odds": a_odd, "odds_source": odds_source,
            "implied_prob_home": h["implied_prob"], "implied_prob_away": a["implied_prob"],
            "edge_home": h["edge"], "edge_away": a["edge"],
            "ev_home": h["ev_per_unit"], "ev_away": a["ev_per_unit"],
            "kelly_stake_home": h["stake_units"], "kelly_stake_away": a["stake_units"],
            "recommended_bet": bet, "confidence": confidence,
            "drift_status": _drift_status(cfg), "last_retrain_utc": bundle["trained_at_utc"],
        })
    preds = pd.DataFrame(rows)
    ensure_parent(cfg["paths"]["predictions_csv"])
    preds.to_csv(cfg["paths"]["predictions_csv"], index=False)

    out = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": bundle["model_name"], "model_trained_at_utc": bundle["trained_at_utc"],
        "ece_20bins_at_train": bundle.get("ece_calibrated"),
        "drift_status": _drift_status(cfg),
        "n_games": len(preds),
        "bets": preds[preds["recommended_bet"] != "NO_BET"][
            ["game_pk", "home_team", "away_team", "recommended_bet",
             "edge_home", "edge_away", "kelly_stake_home", "kelly_stake_away", "confidence"]
        ].to_dict(orient="records"),
        "all_games": rows,
    }
    ensure_parent(cfg["paths"]["output_json"])
    with open(cfg["paths"]["output_json"], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[stage5] OK -> {cfg['paths']['predictions_csv']} y output.json "
          f"({(preds['recommended_bet'] != 'NO_BET').sum()} apuestas de {len(preds)})")
    return preds


def _drift_status(cfg: dict) -> str:
    if os.path.exists(cfg["paths"]["drift_report"]):
        with open(cfg["paths"]["drift_report"], encoding="utf-8") as fh:
            return json.load(fh).get("drift", {}).get("status", "UNKNOWN")
    return "UNKNOWN"


if __name__ == "__main__":
    run(load_config())
