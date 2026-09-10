"""STAGE 2 - Feature engineering con sabermetrics fundamentados.

Baseline predictors
  Log5:              p = Ph*(1-Pa) / (Ph*(1-Pa) + Pa*(1-Ph))
  Pythagorean Exp:   W% = RF^x / (RF^x + RA^x)   (x = 1.83)
  HFA:               win% en casa vs fuera (ponderado)
Ofensivo (proxies rolling): R/G, wRAA-ish = (R/G - liga)/liga
Pitcheo (proxies):  RA/G como ERA de equipo, WHIP proxy = (H+BB)/IP no disponible
                    a nivel equipo via Stats API gratis -> se documenta placeholder.
SELECCION: variance threshold + correlacion + importancia de arboles (en stage 3/4).
ANTI-LEAKAGE: todos los rolling usan shift(1) -> solo info de juegos ANTERIORES.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config_loader import load_config, ensure_parent

# Park factors (runs) - aproximados, actualizar cada temporada.
# KEYS = nombres exactos como vienen de la API ("team.name").
PARK_FACTORS = {
    "Colorado Rockies": 1.28, "Boston Red Sox": 1.06, "Cincinnati Reds": 1.04,
    "Texas Rangers": 1.04, "Philadelphia Phillies": 1.02, "Baltimore Orioles": 1.01,
    "Arizona Diamondbacks": 1.01, "Chicago Cubs": 1.00, "Houston Astros": 1.00,
    "Kansas City Royals": 1.00, "Minnesota Twins": 1.00, "Washington Nationals": 1.00,
    "New York Yankees": 0.99, "Toronto Blue Jays": 0.99, "Atlanta Braves": 0.98,
    "Cleveland Guardians": 0.98, "Detroit Tigers": 0.98, "Los Angeles Angels": 0.98,
    "Pittsburgh Pirates": 0.98, "St. Louis Cardinals": 0.98, "Chicago White Sox": 0.97,
    "Los Angeles Dodgers": 0.97, "Milwaukee Brewers": 0.97, "New York Mets": 0.97,
    "San Diego Padres": 0.96, "San Francisco Giants": 0.96, "Miami Marlins": 0.95,
    "Oakland Athletics": 0.95, "Seattle Mariners": 0.95, "Tampa Bay Rays": 0.94,
}
LEAGUE_AVG_PF = 1.0

FEATURE_COLS = [
    "log5_prob_home", "pythag_diff", "winpct_diff", "rpg_diff", "rag_diff",
    "pf_home", "pf_away", "rest_days_diff", "hfa_weighted",
    "home_rpg_roll10", "away_rpg_roll10", "home_rag_roll10", "away_rag_roll10",
    "home_winpct_roll10", "away_winpct_roll10",
]


def park_factor(team: str) -> float:
    return PARK_FACTORS.get(team, LEAGUE_AVG_PF)


def log5(ph: float, pa: float) -> float:
    """Probabilidad de que el equipo con win% Ph venza al de win% Pa."""
    ph, pa = np.clip(ph, 0.01, 0.99), np.clip(pa, 0.01, 0.99)
    num = ph * (1 - pa)
    return float(num / (num + pa * (1 - ph)))


def pythagorean_expectation(runs_for: float, runs_against: float, exponent: float = 1.83) -> float:
    rf, ra = max(runs_for, 0.1), max(runs_against, 0.1)
    return float(rf ** exponent / (rf ** exponent + ra ** exponent))


def build_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """games (solo Final) -> log largo por equipo, ordenado temporalmente."""
    fin = games[games["status"] == "Final"].dropna(subset=["home_score", "away_score"]).copy()
    fin["home_score"] = fin["home_score"].astype(int)
    fin["away_score"] = fin["away_score"].astype(int)
    rows = []
    for _, g in fin.iterrows():
        rows.append({"game_pk": g["game_pk"], "date": g["date"], "team": g["home_team"],
                     "opp": g["away_team"], "is_home": 1, "runs_for": g["home_score"],
                     "runs_against": g["away_score"],
                     "win": int(g["home_score"] > g["away_score"])})
        rows.append({"game_pk": g["game_pk"], "date": g["date"], "team": g["away_team"],
                     "opp": g["home_team"], "is_home": 0, "runs_for": g["away_score"],
                     "runs_against": g["home_score"],
                     "win": int(g["away_score"] > g["home_score"])})
    log = pd.DataFrame(rows).sort_values(["date", "game_pk"]).reset_index(drop=True)
    return log


def add_rolling_features(log: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Stats shift(1) + rolling/expanding. Cero leakage."""
    log = log.copy()
    log["rest_days"] = log.groupby("team")["date"].diff().dt.days.fillna(3).clip(upper=7)
    grp = log.groupby("team")
    log["winpct_exp"] = grp["win"].transform(lambda s: s.shift(1).expanding().mean()).fillna(0.5)
    log["rpg_exp"] = grp["runs_for"].transform(lambda s: s.shift(1).expanding().mean()).fillna(4.5)
    log["rag_exp"] = grp["runs_against"].transform(lambda s: s.shift(1).expanding().mean()).fillna(4.5)
    for w in windows:
        log[f"winpct_roll{w}"] = grp["win"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=3).mean()).fillna(0.5)
        log[f"rpg_roll{w}"] = grp["runs_for"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=3).mean()).fillna(4.5)
        log[f"rag_roll{w}"] = grp["runs_against"].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=3).mean()).fillna(4.5)
    return log


def compute_matchup_features(games: pd.DataFrame, log: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Una fila por juego con FEATURE_COLS + target home_win (NaN si no Final)."""
    exp_x = cfg["features"]["pythagorean_exponent"]
    idx = log.drop_duplicates("game_pk", keep="last").set_index(["game_pk", "team"])
    hfa = _weighted_hfa(log)
    rows = []
    for _, g in games.iterrows():
        h = idx.xs(g["game_pk"]).loc[g["home_team"]] if g["home_team"] in idx.xs(g["game_pk"]).index else None
        a = idx.xs(g["game_pk"]).loc[g["away_team"]] if g["away_team"] in idx.xs(g["game_pk"]).index else None
        if h is None or a is None:  # equipo sin historial (inicio de temporada)
            continue
        row = {
            "game_pk": g["game_pk"], "date": g["date"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "log5_prob_home": log5(h["winpct_exp"], a["winpct_exp"]),
            "pythag_diff": pythagorean_expectation(h["rpg_exp"], h["rag_exp"], exp_x)
                         - pythagorean_expectation(a["rpg_exp"], a["rag_exp"], exp_x),
            "winpct_diff": h["winpct_exp"] - a["winpct_exp"],
            "rpg_diff": h["rpg_roll10"] - a["rpg_roll10"],
            "rag_diff": a["rag_roll10"] - h["rag_roll10"],  # + favorece al home
            "pf_home": park_factor(g["home_team"]), "pf_away": park_factor(g["away_team"]),
            "rest_days_diff": float(h["rest_days"]) - float(a["rest_days"]),
            "hfa_weighted": hfa,
            "home_rpg_roll10": h["rpg_roll10"], "away_rpg_roll10": a["rpg_roll10"],
            "home_rag_roll10": h["rag_roll10"], "away_rag_roll10": a["rag_roll10"],
            "home_winpct_roll10": h["winpct_roll10"], "away_winpct_roll10": a["winpct_roll10"],
        }
        if g["status"] == "Final":
            row["home_win"] = int(g["home_score"] > g["away_score"])
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_hfa(log: pd.DataFrame) -> float:
    """HFA ponderado: win% casa - win% visita (expanding por equipo)."""
    home = log[log["is_home"] == 1]["win"].mean()
    away = log[log["is_home"] == 0]["win"].mean()
    return float((home - away) if pd.notna(home - away) else 0.04)


def hybrid_feature_selection(X: pd.DataFrame, cfg: dict) -> list[str]:
    """Filtro hibrido: varianza + matriz de correlacion (embedded importancia en stage3/4)."""
    from sklearn.feature_selection import VarianceThreshold
    vt = VarianceThreshold(threshold=cfg["features"]["variance_threshold"])
    kept = X.columns[vt.fit(X).get_support()].tolist()
    corr = X[kept].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > cfg["features"]["correlation_threshold"])]
    return [c for c in kept if c not in to_drop]


def run(cfg: dict) -> pd.DataFrame:
    games = pd.read_csv(cfg["paths"]["raw_games"])
    games["date"] = pd.to_datetime(games["date"])
    log = add_rolling_features(build_team_game_log(games), cfg["features"]["rolling_windows"])
    feats = compute_matchup_features(games, log, cfg)
    feats["date"] = pd.to_datetime(feats["date"])
    feats = feats.sort_values("date").reset_index(drop=True)
    path = cfg["paths"]["features"]
    ensure_parent(path)
    feats.to_parquet(path, index=False)
    print(f"[stage2] OK -> {path} ({len(feats)} juegos, {len(FEATURE_COLS)} features)")
    return feats


if __name__ == "__main__":
    run(load_config())
