"""Run one pipeline from a YAML config.

Usage: python scripts/run_experiment.py configs/base.yaml [--seed N]
"""

import argparse

from thunder.config import load_config
from thunder.experiments.runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a YAML config")
    parser.add_argument("--seed", type=int, help="override the config seed")
    args = parser.parse_args()

    overrides = {"seed": args.seed} if args.seed is not None else None
    cfg = load_config(args.config, overrides)
    run_dir = run_pipeline(cfg)
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
