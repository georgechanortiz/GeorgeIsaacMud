"""Damping classification for the E sweep: overshoot of the sphere's depth trace after contact.

    python analyze_esweep.py <report dir> [<report dir> ...] --out <summary dir>
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "/home/george/Documents/research/GeorgeIsaacMud/genesis")
from scripts.unitree_policy.plots import INK_MUTED, SURFACE, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

VZ_DEADBAND = 0.002  # m/s; velocity reversals smaller than this are not counted as bobbing
REBOUND_TOL = 0.5e-3  # m; springback below this (~4% of a 13 mm particle) counts as none


def analyze(run_dir):
    d = np.load(run_dir / "drop_data.npz")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    t, depth, vz = d["t"], d["depth"], d["vz"]
    c = int(np.argmax(depth > 0))
    # The first stop (vz first reaches >= 0 after contact), not the global max: the sphere keeps
    # creeping down afterwards, so the global max is often the last sample.
    stops = np.nonzero(vz[c:] >= 0.0)[0]
    i_max = c + int(stops[0]) if len(stops) else len(depth) - 1
    tail = t >= t[-1] - 0.1
    final = float(depth[tail].mean())
    overshoot = float(depth[i_max] - final)
    rebound = float(depth[i_max] - depth[i_max:].min())  # largest springback after the deepest point
    # direction reversals after contact, ignoring sub-deadband jitter
    v = vz[c:]
    sign = np.where(v > VZ_DEADBAND, 1, np.where(v < -VZ_DEADBAND, -1, 0))
    nz = sign[sign != 0]
    reversals = int(np.count_nonzero(np.diff(nz) != 0))
    os_ratio = rebound / final if final > 0 else 0.0  # zeta from the springback, as a step-response overshoot
    if os_ratio > 1e-4:
        ln = math.log(os_ratio)
        zeta = -ln / math.sqrt(math.pi**2 + ln**2)
    else:
        zeta = None
    # time to settle within 2% of the final depth (and stay there)
    outside = np.abs(depth[c:] - final) > 0.02 * final
    settle_t = float(t[c + int(np.nonzero(outside)[0][-1]) + 1] - t[c]) if outside.any() else 0.0
    # creep: depth change over the last 0.5 s (positive = still sinking)
    late = t >= t[-1] - 0.5
    creep = float(depth[late][-1] - depth[late][0])
    return {
        "run": run_dir.name,
        "E": float(cfg["mud"]["E"]),
        "substeps": int(cfg["sim"]["substeps"]),
        "t_contact": float(t[c]),
        "max_depth_mm": depth[i_max] * 1e3,
        "final_depth_mm": final * 1e3,
        "overshoot_mm": overshoot * 1e3,
        "overshoot_pct": os_ratio * 100,
        "rebound_mm": rebound * 1e3,
        "max_rebound_vz_m_s": float(v.max()),
        "reversals": reversals,
        "zeta_est": zeta,
        "settle_time_s": settle_t,
        "late_creep_mm": creep * 1e3,
        "peak_fz_n": float(d["fz"].max()),
        "verdict": "underdamped" if rebound > REBOUND_TOL or reversals > 1 else "critically/overdamped",
        "_t": t - t[c], "_depth": depth,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = sorted((analyze(r) for r in args.runs), key=lambda r: (r["substeps"], r["E"]))
    args.out.mkdir(parents=True, exist_ok=True)

    groups = sorted({r["substeps"] for r in res})
    fig, axes = plt.subplots(len(groups), 2, figsize=(13, 4.6 * len(groups)), facecolor=SURFACE, squeeze=False)
    cmap = plt.get_cmap("viridis")
    all_E = sorted({r["E"] for r in res})
    for (ax, ax_z), n in zip(axes, groups):
        rows = [r for r in res if r["substeps"] == n]
        for r in rows:
            color = cmap(all_E.index(r["E"]) / max(1, len(all_E) - 1))
            for a in (ax, ax_z):
                a.set_facecolor(SURFACE)
                a.plot(r["_t"], r["_depth"] * 1e3, color=color, linewidth=1.6, label=f"E = {r['E']:.0e} Pa")
        ax.invert_yaxis()
        style_axes(ax, f"Sphere depth after contact, {n} substeps (yield 80 Pa)", "time since contact (s)", "depth (mm)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)
        ax_z.set_xlim(0.0, 0.6)
        win = [r["_depth"][(r["_t"] >= 0.08) & (r["_t"] <= 0.6)] * 1e3 for r in rows]
        lo = min(w.min() for w in win) - 1
        hi = max(w.max() for w in win) + 1
        ax_z.set_ylim(hi, lo)
        style_axes(ax_z, f"Zoom: peak and springback, {n} substeps", "time since contact (s)", "depth (mm)")
    fig.tight_layout()
    fig.savefig(args.out / "depth_vs_time_by_E.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    table = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]
    (args.out / "summary.json").write_text(json.dumps(table, indent=2))
    hdr = f"{'sub':>3} {'E (Pa)':>8} {'max mm':>7} {'final mm':>8} {'springback mm':>13} {'rebound v':>9} {'revs':>4} {'zeta':>6} {'settle s':>8} {'creep mm':>8} {'peak Fz':>7}  verdict"
    lines = [hdr]
    for r in table:
        z = f"{r['zeta_est']:.2f}" if r["zeta_est"] is not None else ">=1"
        lines.append(f"{r['substeps']:3d} {r['E']:8.0e} {r['max_depth_mm']:7.1f} {r['final_depth_mm']:8.1f} {r['rebound_mm']:13.2f} "
                     f"{r['max_rebound_vz_m_s']:9.3f} {r['reversals']:4d} {z:>6} "
                     f"{r['settle_time_s']:8.3f} {r['late_creep_mm']:8.2f} {r['peak_fz_n']:7.1f}  {r['verdict']}  [{r['run']}]")
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
