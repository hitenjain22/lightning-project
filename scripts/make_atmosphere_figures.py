"""M5 deliverable: ray fans (still air, lapse rate, strong wind) and straight-line error maps.

Usage: python scripts/make_atmosphere_figures.py configs/experiments/atmosphere_figures.yaml

Ray fans use the independent reference tracer (solve_ivp on the Hamiltonian ray equations).
Error maps compare the straight-line travel time R / c(ground) and direction with the traced
eigenrays over a grid of source heights and ranges, in a vertical plane along the wind.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from thunder.atmosphere import raytrace as rt  # noqa: E402
from thunder.atmosphere.profiles import build_atmosphere  # noqa: E402
from thunder.config import AtmosphereConfig, load_config  # noqa: E402
from thunder.experiments.runner import make_run_dir, write_run_files  # noqa: E402

SOURCE_HEIGHT_M = 500.0
LAUNCH_DEG = np.arange(-60.0, 10.1, 5.0)  # elevation of the launch direction


def cases(cfg):
    lapse = AtmosphereConfig(
        model="stratified", temperature_c=cfg.atmosphere.temperature_c, relative_humidity=0.0
    )
    windy = AtmosphereConfig.model_validate(
        {**lapse.model_dump(), "wind": {"speed_mps": 10.0, "direction_from_deg": 270.0}}
    )
    still = AtmosphereConfig(model="stratified", lapse_rate_k_per_km=0.0, relative_humidity=0.0)
    return [
        ("still air (no gradient)", still),
        ("standard lapse rate (6.5 K/km)", lapse),
        ("lapse rate + 10 m/s wind from the west", windy),
    ]


def ray_fans(path, cfg):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for ax, (title, acfg) in zip(axes, cases(cfg), strict=True):
        atm = build_atmosphere(acfg)
        for sign in (-1.0, 1.0):  # rays toward -x (upwind with a west wind) and +x (downwind)
            for el in LAUNCH_DEG:
                d = np.array([sign * np.cos(np.radians(el)), 0.0, np.sin(np.radians(el))])
                _, pos, _ = rt.trace_ray(atm, np.array([0.0, 0.0, SOURCE_HEIGHT_M]), d, 40.0, max_step=0.01)
                ax.plot(pos[:, 0] / 1000, pos[:, 2], lw=0.7, color="#1f3a93" if sign > 0 else "#c0392b")
        ax.plot(0, SOURCE_HEIGHT_M, "k*", ms=12)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_ylim(-20, 1500)
        ax.set_xlim(-12, 12)
        ax.set_ylabel("height (m)")
        ax.set_title(title, loc="left", fontsize=11)
    axes[-1].set_xlabel("x (km): blue rays head east (downwind in the last panel), red rays west")
    fig.suptitle(f"Ray fans from a source at {SOURCE_HEIGHT_M:.0f} m (launch elevations -60 to +10 deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def error_maps(path, cfg):
    acfg = cases(cfg)[2][1]  # lapse + wind
    atm = build_atmosphere(acfg)
    mic = np.array([0.0, 0.0, 1.5])
    c0 = float(atm.sound_speed(np.array([mic[2]]))[0])
    x = np.linspace(-12000, 12000, 121)
    z = np.linspace(50, 7000, 70)
    X, Z = np.meshgrid(x, z)
    src = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    p = atm.propagate(src, mic)
    R = np.linalg.norm(src - mic, axis=1)
    dt = (p.travel_time - R / c0).reshape(X.shape)  # traced minus straight-line, s
    straight_dir = (mic - src) / R[:, None]
    ang = np.degrees(np.arccos(np.clip(np.sum(p.arrival_direction * straight_dir, axis=1), -1, 1))).reshape(
        X.shape
    )
    ang[~p.valid.reshape(X.shape)] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    m0 = axes[0].pcolormesh(
        x / 1000,
        z / 1000,
        1000 * dt,
        cmap="RdBu_r",
        shading="auto",
        vmin=-np.nanmax(np.abs(1000 * dt)),
        vmax=np.nanmax(np.abs(1000 * dt)),
    )
    fig.colorbar(m0, ax=axes[0], label="traced - straight-line travel time (ms)")
    m1 = axes[1].pcolormesh(x / 1000, z / 1000, ang, cmap="magma", shading="auto")
    fig.colorbar(m1, ax=axes[1], label="arrival direction error of a straight line (deg)")
    for ax in axes:
        shadow = (~p.valid).reshape(X.shape)
        ax.contourf(x / 1000, z / 1000, shadow, levels=[0.5, 1.5], colors=["0.85"])
        ax.set_xlabel("source x (km) (wind blows toward +x)")
        ax.set_ylabel("source height (km)")
    axes[0].set_title("Travel-time error of straight rays at c(ground); gray: acoustic shadow")
    axes[1].set_title("Direction error of straight rays (gray: shadow)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    valid = p.valid
    return {
        "max_abs_travel_time_error_ms": float(np.nanmax(np.abs(1000 * dt))),
        "median_direction_error_deg": float(np.nanmedian(ang)),
        "shadow_fraction": float(1 - valid.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg)
    ray_fans(run_dir / "figures" / "ray_fans.png", cfg)
    summary = error_maps(run_dir / "figures" / "straight_line_error_maps.png", cfg)
    write_run_files(cfg, run_dir, summary, {})
    print(summary)
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
