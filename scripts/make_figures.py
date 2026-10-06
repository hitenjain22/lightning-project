"""Regenerate figures and summary.md for a Monte Carlo run folder.

Usage: python scripts/make_figures.py results/<experiment>/<run_id> [--docs-prefix e1_]

With --docs-prefix, the PNGs are also copied to docs/figures/<prefix><name>.png.
"""

import argparse
from pathlib import Path

from thunder.experiments.reports import make_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--docs-prefix", default=None)
    args = parser.parse_args()
    docs = Path("docs/figures") if args.docs_prefix is not None else None
    out = make_report(args.run_dir, docs, args.docs_prefix or "")
    print(out.read_text())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
