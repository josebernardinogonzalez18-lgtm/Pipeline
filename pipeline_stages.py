"""Orquestador de los 5 stages secuenciales.

Uso (GitHub Actions):
  python -m src.pipeline_stages all          # flujo completo (retrain)
  python -m src.pipeline_stages daily        # ingest->features->eval->predict
  python -m src.pipeline_stages --from-stage 3
Stages:
  1 ingestion | 2 features | 3 train | 4 evaluate | 5 predict
"""
from __future__ import annotations
import argparse
import json
import sys

from src.config_loader import load_config

STAGES = ["ingest", "features", "train", "evaluate", "predict"]


def run_stages(cfg: dict, names: list[str]) -> dict:
    results: dict = {}
    for name in names:
        if name == "ingest":
            from src.stage1_ingestion import run as r
            r(cfg)
        elif name == "features":
            from src.stage2_features import run as r
            r(cfg)
        elif name == "train":
            from src.stage3_training import run as r
            results["train"] = r(cfg)
        elif name == "evaluate":
            from src.stage4_evaluation import run as r
            results["evaluate"] = r(cfg)
        elif name == "predict":
            from src.stage5_inference import run as r
            r(cfg)
        else:
            raise ValueError(f"stage desconocido: {name}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "daily"] + STAGES)
    ap.add_argument("--from-stage", type=int, default=None, choices=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    cfg = load_config()

    if args.mode == "all":
        seq = STAGES
    elif args.mode == "daily":
        seq = ["ingest", "features", "evaluate", "predict"]  # sin retrain
    else:
        seq = [args.mode]
    if args.from_stage:
        seq = STAGES[args.from_stage - 1:]

    print(f"[pipeline] ejecutando stages: {seq}")
    results = run_stages(cfg, seq)
    if results:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
