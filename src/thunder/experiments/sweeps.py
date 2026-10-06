"""Helpers shared by the Phase 7 experiment scripts (E2-E5).

An experiment is a Monte Carlo over variants (overrides of one base config, all run on the
same bolts; see pipeline.run_bolt_variants). `run_variants` executes it into a run folder;
the analysis helpers read a run folder back, so figures and summaries can be regenerated
from the folder alone (SPEC.md reproducibility rule).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from thunder.config import RunConfig
from thunder.eval.aggregate import bootstrap_ci
from thunder.experiments.runner import run_experiment


def run_variants(base: RunConfig, variants: dict[str, dict], meta: dict[str, Any] | None = None) -> Path:
    """Run every variant on the same bolts; write `variants_meta.json` (per-variant labels) too."""
    if base.monte_carlo is None:
        raise ValueError("the base config needs a monte_carlo section")
    cfg = base.model_copy(update={"monte_carlo": base.monte_carlo.model_copy(update={"variants": variants})})
    cfg = RunConfig.model_validate(cfg.model_dump(mode="json"))  # validate the variants as a whole
    run_dir = run_experiment(cfg)
    (run_dir / "variants_meta.json").write_text(json.dumps(meta or {}, indent=2, default=float))
    return run_dir


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict, dict, dict]:
    """(bolts table, summary, resolved config, variants meta) of a run folder."""
    run_dir = Path(run_dir)
    table = pd.read_csv(run_dir / "bolts.csv")
    summary = json.loads((run_dir / "metrics.json").read_text())
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    meta_path = run_dir / "variants_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return table, summary, cfg, meta


def _median(v: np.ndarray) -> float:
    return float(np.median(v))


def _mean(v: np.ndarray) -> float:
    return float(np.mean(v))


def per_variant(
    table: pd.DataFrame,
    column: str,
    stat: Callable[[np.ndarray], float] = _median,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Statistic of a per-bolt column for each variant, with a 95% bootstrap CI over bolts."""
    rng = np.random.default_rng(seed)
    rows = []
    for v, t in table.groupby("variant", sort=False):
        est, lo, hi = bootstrap_ci(t[column].to_numpy(dtype=float), stat, rng, n_boot)
        rows.append({"variant": v, "value": est, "lo": lo, "hi": hi, "n": int(np.isfinite(t[column]).sum())})
    return pd.DataFrame(rows).set_index("variant")


def crossing(levels: np.ndarray, values: np.ndarray, target: float, log_x: bool = True) -> float | None:
    """Largest factor level at which `values` (increasing with level) stays <= target.

    Linear interpolation in log(level) (or level) between the bracketing levels; None if the
    target is never met, +inf if it is met at every level. A missing value (NaN: e.g. no bolt
    produced any point) counts as failing; if the first failure is missing, the last passing
    level is returned (conservative: nothing is known in between).
    """
    lv = np.asarray(levels, dtype=float)
    vv = np.asarray(values, dtype=float)
    vv = np.where(np.isfinite(vv), vv, np.inf)
    order = np.argsort(lv)
    lv, vv = lv[order], vv[order]
    ok = vv <= target
    if not ok[0]:
        return None
    if ok.all():
        return float("inf")
    i = int(np.argmin(ok)) - 1  # last level meeting the target, before the first failure
    x0, x1 = lv[i], lv[i + 1]
    y0, y1 = vv[i], vv[i + 1]
    if not np.isfinite(y1):
        return float(x0)
    f = (target - y0) / (y1 - y0) if y1 != y0 else 0.0
    if log_x and x0 > 0:
        return float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
    return float(x0 + f * (x1 - x0))


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    """Minimal markdown table (no external dependency)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + " --- |" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, floatfmt) if isinstance(v, float | np.floating) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def paired_ratio(
    table: pd.DataFrame, column: str, baseline: str, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Median over bolts of column[variant] / column[baseline], with a 95% bootstrap CI.

    Every variant ran on the same bolts (common random numbers), so the per-bolt ratio removes
    bolt-to-bolt variation: a much sharper comparison than two independent medians. Bolts where
    either value is missing are skipped (`n` counts the pairs used).
    """
    rng = np.random.default_rng(seed)
    piv = table.pivot(index="bolt", columns="variant", values=column)
    base = piv[baseline].to_numpy(dtype=float)
    rows = []
    for v in piv.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = piv[v].to_numpy(dtype=float) / base
        r = r[np.isfinite(r)]
        est, lo, hi = bootstrap_ci(r, np.median, rng, n_boot)
        rows.append({"variant": v, "ratio": est, "lo": lo, "hi": hi, "n": int(r.size)})
    return pd.DataFrame(rows).set_index("variant")


def variant_table(table: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Standard per-variant summary: medians/means over bolts with 95% bootstrap CIs."""
    cols = {
        "angular_error_median_deg": _median,
        "point_error_median_m": _median,
        "point_error_p90_m": _median,
        "coverage_50m": _mean,
        "coverage_main_50m": _mean,
        "coverage_branch_50m": _mean,
        "strike_error_m": _median,
        "time_reconstruct_s": _median,
        "time_reconstruct_cpu_s": _median,
    }
    out = pd.DataFrame(index=pd.Index(table["variant"].unique(), name="variant"))
    for col, stat in cols.items():
        if col not in table:
            continue
        pv = per_variant(table, col, stat, n_boot, seed)
        out[col] = pv["value"]
        out[col + "_lo"] = pv["lo"]
        out[col + "_hi"] = pv["hi"]
    g = table.groupby("variant", sort=False)
    out["bolts"] = g.size()
    out["bolts_with_points"] = g["n_points"].apply(lambda s: int((s > 0).sum()))
    out["points_per_bolt"] = g["n_points"].mean()
    if {"n_windows_passed", "n_windows_active"} <= set(table):
        out["windows_passed_frac"] = g.apply(
            lambda t: float(t["n_windows_passed"].sum() / max(t["n_windows_active"].sum(), 1))
        )
    return out


def ci_text(row: pd.Series, col: str, fmt: str = ".2f", scale: float = 1.0, unit: str = "") -> str:
    """'v [lo, hi]unit' from a variant_table row."""
    v, lo, hi = (row[col] * scale, row[col + "_lo"] * scale, row[col + "_hi"] * scale)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]{unit}"
