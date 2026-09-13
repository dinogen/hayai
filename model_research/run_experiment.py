"""
Entry point for running experiments.

Usage:
    python run_experiment.py --exp EXP-001
    python run_experiment.py --exp EXP-002 --dry-run
    python run_experiment.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from settings import REGISTRY_FILE, RESULTS_DIR


def load_registry() -> list[dict]:
    with open(REGISTRY_FILE) as f:
        return yaml.safe_load(f)["experiments"]


def list_experiments(registry: list[dict]) -> None:
    print(f"\n{'ID':<12} {'Status':<10} {'Name'}")
    print("-" * 60)
    for exp in registry:
        print(f"{exp['id']:<12} {exp['status']:<10} {exp['name']}")
    print()


def get_experiment(registry: list[dict], exp_id: str) -> dict:
    for exp in registry:
        if exp["id"] == exp_id:
            return exp
    raise ValueError(f"Experiment {exp_id!r} not found in registry.")


def run(exp: dict, dry_run: bool = False) -> None:
    exp_id = exp["id"]
    out_dir = RESULTS_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {exp_id} — {exp['name']}")
    print(f"{'='*60}")
    print(f"  Model   : {exp.get('model', 'tbd')}")
    print(f"  Window  : {exp.get('window', '-')}")
    print(f"  Features: {exp.get('features', '-')}")
    print(f"  Targets : {exp.get('targets', '-')}")
    print(f"  Split   : {exp.get('split', '-')}")
    print(f"  Out dir : {out_dir}")
    print()

    if dry_run:
        print("  [dry-run] Skipping execution.")
        return

    model_type = exp.get("model", "")

    if model_type == "baseline":
        from models.baseline import run_baseline
        run_baseline(exp, out_dir)
    elif model_type == "dnn":
        from models.dnn import run_dnn
        run_dnn(exp, out_dir)
    elif model_type == "cnn":
        from models.cnn import run_cnn
        run_cnn(exp, out_dir)
    elif model_type == "gru":
        from models.gru import run_gru
        run_gru(exp, out_dir)
    else:
        print(f"  Model type '{model_type}' not yet implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Hayai V2 research experiment")
    parser.add_argument("--exp",     type=str, help="Experiment ID, e.g. EXP-001")
    parser.add_argument("--list",    action="store_true", help="List all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")
    args = parser.parse_args()

    registry = load_registry()

    if args.list:
        list_experiments(registry)
        sys.exit(0)

    if not args.exp:
        parser.print_help()
        sys.exit(1)

    exp = get_experiment(registry, args.exp)
    run(exp, dry_run=args.dry_run)
