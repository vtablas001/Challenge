"""Run the complete, reproducible MMM sequence from the repository root."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmm.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final MMM model sequence")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-sensitivity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        root=args.root,
        n_trials=args.n_trials,
        seed=args.seed,
        run_sensitivity=not args.skip_sensitivity,
    )
    print(result.model_table.round(5).to_string(index=False))
    print("\nModelo recomendado:", result.recommendation["selected_model"])


if __name__ == "__main__":
    main()
