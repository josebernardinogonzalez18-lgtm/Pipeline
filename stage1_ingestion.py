"""STAGE 1 - Ingestion de datos desde MLB Stats API (sin API key).

Endpoints oficiales (gratuitos):
  - /api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=team,linescore,probablePitcher
Salida: data/games_raw.csv  (append incremental, idempotente por game_pk)
"""
from __future__ import annotations
import datetime as dt
import os
import time
import pandas as pd
import requests

from src.config_loader import load_config, ensure_parent

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
REQUEST_TIMEOUT = 30


def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_day(date_str: str, sport_id: int = 1, hydrate: str = "team,linescore,probablePitcher") -> list[dict]:
    """Un juego = una fila con marcador final si esta COMPLETED."""
    data = _get(SCHEDULE_URL, {"sportId": sport_id, "date": date_str, "hydrate": hydrate})
    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            row = {
                "game_pk": g.get("gamePk"),
                "date": date_str,
                "game_type": g.get("gameType"),
                "status": status,
                "home_team": g.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                "away_team": g.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                "home_probable_pitcher": (g.get("teams", {}).get("home", {}).get("probablePitcher") or {}).get("fullName"),
                "away_probable_pitcher": (g.get("teams", {}).get("away", {}).get("probablePitcher") or {}).get("fullName"),
                "venue": (g.get("venue") or {}).get("name"),
            }
            if status == "Final":
                row["home_score"] = g.get("teams", {}).get("home", {}).get("score")
                row["away_score"] = g.get("teams", {}).get("away", {}).get("score")
            rows.append(row)
    return rows


def fetch_range(start: str, end: str, cfg: dict) -> pd.DataFrame:
    sleep_s = cfg["ingestion"]["request_sleep_seconds"]
    sport_id = cfg["ingestion"]["sport_id"]
    hydrate = cfg["ingestion"]["hydrate"]
    start_d, end_d = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    rows, cur = [], start_d
    while cur <= end_d:
        ds = cur.isoformat()
        try:
            rows.extend(fetch_day(ds, sport_id, hydrate))
        except requests.RequestException as exc:
            print(f"[stage1] WARN {ds}: {exc}")
        time.sleep(sleep_s)
        cur += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def run(cfg: dict, today: str | None = None) -> pd.DataFrame:
    """Descarga historico (solo dias faltantes) + hoy. Idempotente."""
    path = cfg["paths"]["raw_games"]
    ensure_parent(path)
    today = today or dt.date.today().isoformat()

    existing = pd.DataFrame()
    if os.path.exists(path):
        existing = pd.read_csv(path, dtype={"game_pk": "Int64"})

    # 1) Relleno historico: desde historical_start hasta ayer
    start = cfg["ingestion"]["historical_start"]
    done_dates = set(existing["date"]) if not existing.empty else set()
    need = [d for d in pd.date_range(start, today)[:-1]]
    to_fetch = [d.date().isoformat() for d in need if d.date().isoformat() not in done_dates]

    new_parts = []
    BATCH = 15
    for i in range(0, len(to_fetch), BATCH):
        chunk = to_fetch[i:i + BATCH]
        df_chunk = pd.concat([fetch_day(d, cfg["ingestion"]["sport_id"],
                                        cfg["ingestion"]["hydrate"])
                              for d in chunk], ignore_index=True)
        new_parts.append(df_chunk)
        time.sleep(cfg["ingestion"]["request_sleep_seconds"])
        print(f"[stage1] historico {min(chunk)}..{max(chunk)} ({len(df_chunk)} juegos)")

    # 2) Hoy (incluye juegos aun no finalizados -> status != Final)
    today_rows = fetch_day(today, cfg["ingestion"]["sport_id"], cfg["ingestion"]["hydrate"])
    print(f"[stage1] hoy {today}: {len(today_rows)} juegos")

    new = pd.concat(new_parts + [pd.DataFrame(today_rows)], ignore_index=True)         if new_parts else pd.DataFrame(today_rows)
    if existing.empty:
        out = new
    else:
        merged = pd.concat([existing, new], ignore_index=True)
        # upsert por game_pk: conservar la fila mas reciente (trae scores si ya hay)
        merged["date"] = pd.to_datetime(merged["date"]).dt.date.astype(str)
        out = (merged.sort_values("status")  # 'Final' queda al final -> gana
                    .drop_duplicates("game_pk", keep="last")
                    .sort_values("date"))
    out.to_csv(path, index=False)
    print(f"[stage1] OK -> {path} ({len(out)} filas)")
    return out


if __name__ == "__main__":
    run(load_config())
