"""M3 deliverable: array preset plots, before/after waveforms and spectra per corruption type.

Usage: python scripts/make_sensor_demo.py configs/experiments/sensor_demo.yaml
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from thunder.acoustics.synth import synthesize  # noqa: E402
from thunder.atmosphere.profiles import build_atmosphere  # noqa: E402
from thunder.channel.generator import generate_channel  # noqa: E402
from thunder.config import ArrayConfig, SensorsConfig, load_config  # noqa: E402
from thunder.experiments.runner import make_run_dir, write_run_files  # noqa: E402
from thunder.sensors.arrays import build_mic_array, horizontal_aperture, nominal_positions  # noqa: E402
from thunder.sensors.corruption import corrupt  # noqa: E402

ARRAY_GALLERY = [
    ("triangle", {}),
    ("square", {}),
    ("square_center", {}),
    ("circle", {"n_mics": 8}),
    ("l_shape", {"n_mics": 7}),
    ("cross", {"n_mics": 9}),
    ("mast", {}),
    ("random_disk", {"n_mics": 8}),
    ("distributed", {"aperture_m": 20.0, "subarray_separation_m": 300.0}),
]

# (title, sensors config, needs resynthesis because the error is physical)
CORRUPTIONS = [
    ("background pink noise, SNR 10 dB", {"noise": {"background_snr_db": 10.0}}, False),
    ("wind noise, 8 m/s", {"noise": {"wind_speed_mps": 8.0}}, False),
    ("rain noise, 60 dB SPL", {"noise": {"rain_db_spl": 60.0}}, False),
    ("measurement mic (2 Hz-2 kHz)", {"mic": {"preset": "measurement"}}, False),
    ("audio mic (20 Hz high-pass)", {"mic": {"preset": "audio"}}, False),
    ("phone mic (100 Hz HP, 120 dB clip, 16-bit)", {"mic": {"preset": "phone"}}, False),
    ("clipping at 105 dB SPL + 8-bit ADC", {"mic": {"clip_db_spl": 105.0, "adc_bits": 8}}, False),
    ("sample jitter 20 us", {"timing": {"jitter_std_s": 20e-6}}, False),
    ("hand-synced clocks (ms offsets, 20 ppm)", {"timing": {"preset": "hand_synced"}}, True),
    ("consumer-GPS mic positions (3-5 m)", {"position": {"preset": "consumer_gps"}}, True),
]


def plot_arrays(path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for ax, (layout, extra) in zip(axes.flat, ARRAY_GALLERY, strict=True):
        cfg = ArrayConfig(layout=layout, **{"aperture_m": 50.0, **extra})
        p = nominal_positions(cfg, np.random.default_rng(0))
        raised = p[:, 2] > cfg.mic_height_m + 1e-9
        ax.scatter(p[~raised, 0], p[~raised, 1], s=60, color="#1f3a93", zorder=3)
        if raised.any():
            ax.scatter(
                p[raised, 0],
                p[raised, 1],
                s=140,
                marker="^",
                color="#c0392b",
                zorder=3,
                label=f"mast, z = {p[raised, 2][0]:.0f} m",
            )
            ax.legend(loc="center right", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        note = (
            f"sub-array aperture {cfg.aperture_m:g} m"
            if layout == "distributed"
            else (f"aperture {horizontal_aperture(p):.1f} m")
        )
        ax.set_title(f"{layout} ({len(p)} mics)\n{note}", fontsize=11)
        ax.set_xlabel("east (m)")
        ax.set_ylabel("north (m)")
    fig.suptitle("Array presets (top view, horizontal centroid at origin)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg.channel is not None and cfg.synthesis is not None
    run_dir = make_run_dir(cfg)
    plot_arrays(run_dir / "figures" / "array_presets.png")

    channel_rng, array_rng, synth_rng, sensor_rng = np.random.default_rng(cfg.seed).spawn(4)
    ch = generate_channel(cfg.channel, channel_rng, seed=cfg.seed)
    atm = build_atmosphere(cfg.atmosphere)
    syn = cfg.synthesis
    fs = cfg.sample_rate_hz

    def make(sensors: SensorsConfig | None):
        arr = build_mic_array(cfg.array, sensors, np.random.default_rng(cfg.seed + 1))
        rec = synthesize(
            ch,
            atm,
            arr,
            fs,
            cfg.oversample,
            syn.emitter_spacing_m,
            syn.acoustic_efficiency,
            np.random.default_rng(cfg.seed + 2),
            syn.micro_turn_mean,
            syn.micro_scale_m,
        )
        return arr, rec

    ideal_arr, clean = make(None)
    x0 = clean.signals[0]
    peak = int(np.argmax(np.abs(x0)))
    win = slice(max(0, peak - int(0.15 * fs)), peak + int(0.35 * fs))
    t = np.arange(len(x0)) / fs

    fig, axes = plt.subplots(len(CORRUPTIONS), 1, figsize=(13, 2.1 * len(CORRUPTIONS)), sharex=True)
    spectra = {}
    summary = {}
    for ax, (title, overrides, physical) in zip(axes, CORRUPTIONS, strict=True):
        sensors = SensorsConfig(**overrides)
        arr, base = make(sensors) if physical else (ideal_arr, clean)
        out = corrupt(base, arr, sensors, np.random.default_rng(cfg.seed + 3))
        y = out.signals[0]
        n = min(len(y), len(x0))
        ax.plot(t[win], x0[win], color="0.65", lw=1.0, label="clean")
        ax.plot(t[win][: n - win.start], y[win][: n - win.start], color="#1f3a93", lw=0.7, label="recorded")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("Pa")
        assert out.truth is not None
        ex = out.truth.extra
        detail = {}
        if physical and "hand" in title:
            detail = {
                "offset_ms": (1e3 * ex["clock_offset"]).round(2).tolist(),
                "drift_ppm": ex["clock_drift_ppm"].round(1).tolist(),
            }
            ax.text(
                0.99,
                0.85,
                f"mic 0 offset {1e3 * ex['clock_offset'][0]:+.2f} ms",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
            )
        if physical and "GPS" in title:
            err = np.linalg.norm(arr.true_positions - arr.nominal_positions, axis=1)
            detail = {"position_error_m": err.round(2).tolist()}
            ax.text(
                0.99, 0.85, f"mic 0 displaced {err[0]:.1f} m", transform=ax.transAxes, ha="right", fontsize=9
            )
        summary[title] = {**ex["corruption_info"], **detail}
        f, p = signal.welch(y, fs, nperseg=4096)
        spectra[title] = (f, p)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time since flash (s)")
    fig.suptitle("Before (gray) and after (blue) each corruption, mic 0, around the loudest clap")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "corruption_waveforms.png", dpi=110)

    f, p_clean = signal.welch(x0, fs, nperseg=4096)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    groups = (
        ("noise", [k for k in spectra if "noise" in k]),
        ("mic chain", [k for k in spectra if "mic" in k or "clipping" in k or "jitter" in k]),
    )
    for ax, (name, keys) in zip(axes, groups, strict=True):
        ax.loglog(f[1:], p_clean[1:], color="k", lw=1.8, label="clean thunder")
        for k in keys:
            ax.loglog(spectra[k][0][1:], spectra[k][1][1:], lw=1.0, label=k)
        ax.axvspan(10, 300, color="0.92", zorder=0)
        ax.set_xlim(1, fs / 2)
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(f"Recorded spectra: {name} (shaded: 10-300 Hz analysis band)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("PSD (Pa²/Hz), mic 0, whole recording")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "corruption_spectra.png", dpi=110)

    # Flash-time error: what each preset does to the range of every reconstructed point.
    c = float(atm.sound_speed(np.array([1.5]))[0])
    t0 = {}
    for preset in ("photodiode", "lightning_network", "video_30fps"):
        s = SensorsConfig(flash_time={"preset": preset})
        scale = s.flash_time.scale_s
        std = scale if s.flash_time.distribution == "normal" else scale / math.sqrt(3)
        t0[preset] = {"t0_error_std_ms": 1e3 * std, "range_error_std_m": c * std}
    summary["flash_time"] = t0
    write_run_files(cfg, run_dir, summary, {})
    print(json.dumps(t0, indent=1))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
