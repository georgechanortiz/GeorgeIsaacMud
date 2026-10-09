#!/usr/bin/env python3
"""Yield-stress sweep over the three policies, for the workshop table.

Runs eval_mpm.py once per (seed, yield stress, policy) and collects the metrics into a CSV.

Seeds are the OUTER loop on purpose: repeat 1 fills every cell of the table before repeat 2
starts, so an interrupted sweep still yields a complete table rather than a complete column.
Already-finished cells are skipped, so the sweep is resumable.
"""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "Workshop sweep"
PY = "/home/rml/anaconda3/envs/genesis/bin/python"

POLICIES = [("Baseline", "logs/Baseline/model_4999.pt"),
            ("Policy_A", "logs/Policy_A/model_4999.pt"),
            ("Policy_B", "logs/Policy_B/model_1499.pt")]
# (label, E [Pa], sigma_y [Pa], rho [kg/m3], n_seeds)
#
# Mud is parameterised from Chen, Yi & Shan 2024, Table I: for each water content the measured
# bulk stiffness alpha is used as the Young's modulus and sigma_y from the same row as the yield
# stress. alpha and sigma_y are both functions of water content, so they are swept as paired rows
# rather than independently. nu is not measured in that work and is held at the value used across
# the 24 validated runs in this repo.
#
# ANCHOR reproduces the archive's validated operating point (24 prior runs, best 2.71 m); it exists
# to confirm the new sweep is commensurable with existing results before any of it is interpreted.
CASES = [("ANCHOR", 2.0e4,  2500.0, 1500.0, 1),   # validated point, not a Table I row
         ("W30",    1.0e4,  2000.0, 1840.0, 3),   # alpha 0.01 MPa, sigma_y  2 kPa (W=35% identical)
         ("W25",    4.0e4,  6000.0, 1840.0, 3),   # alpha 0.04 MPa, sigma_y  6 kPa
         ("W15",    1.7e5, 17000.0, 1840.0, 3)]   # alpha 0.17 MPa, sigma_y 17 kPa
SOFT_NU = 0.35
SOFT_Z  = ("0.06", "0.11")   # flush on the stiff base; the foot must reach it to push off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)      # 500 steps @ dt 0.02 = 10 s
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    # Anchor first (validation gate), then seeds outermost so repeat 1 fills every cell
    # before repeat 2 starts -- an interrupted sweep yields a complete table, not a complete column.
    jobs  = [(1, l, E, y, r, n, c) for l, E, y, r, ns in CASES if ns == 1 for n, c in POLICIES]
    jobs += [(sd, l, E, y, r, n, c) for sd in range(1, a.seeds + 1)
             for l, E, y, r, ns in CASES if ns > 1 for n, c in POLICIES]
    print(f"{len(jobs)} runs ({len(CASES)} columns x {len(POLICIES)} policies) -> {OUT}", flush=True)

    t0 = time.time()
    for i, (seed, lbl, E, sy, rho, name, ckpt) in enumerate(jobs, 1):
        tag = f"{lbl}/{name}/seed{seed}"
        dest = OUT / tag
        if list(dest.glob("*/params.json")):
            print(f"[{i}/{len(jobs)}] SKIP (done) {tag}", flush=True)
            continue
        cmd = [PY, "scripts/eval_mpm.py", "--policy", ckpt, "--reporting", "--headless",
               "--steps", str(a.steps), "--seed", str(seed),
               "--yield-stress", str(sy), "--soft-E", str(E),
               "--soft-nu", str(SOFT_NU), "--soft-rho", str(rho),
               "--soft-z", SOFT_Z[0], SOFT_Z[1],
               "--out-root", str(OUT), "--tag", tag]
        print(f"[{i}/{len(jobs)}] RUN {tag}  (elapsed {(time.time()-t0)/60:.0f} min)", flush=True)
        if a.dry_run:
            print("   " + " ".join(cmd), flush=True); continue
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   FAILED rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}", flush=True)
        else:
            print(f"   ok ({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
        collect()
    collect()
    print(f"SWEEP COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


def collect():
    """Re-scrape every finished run into results.csv (cheap; keeps partial results usable)."""
    rows = []
    for pj in sorted(OUT.glob("**/params.json")):
        if any(part.startswith("_discarded") for part in pj.parts):
            continue   # superseded runs kept for the record, not results
        d = json.loads(pj.read_text())
        sw, m = d.get("soft_mud_sweep", {}), d.get("metrics", {})
        rows.append({
            "policy": d.get("run"), "yield_stress_pa": sw.get("yield_stress"), "seed": sw.get("seed"),
            "E": sw.get("E"), "nu": sw.get("nu"), "rho": sw.get("rho"),
            "particles": d.get("mpm_particles"), "wall_per_sim": d.get("wall_time_to_sim_time_ratio"),
            "mech_cost_of_transport": m.get("mech_cost_of_transport"),
            "mean_speed_xy_m_s": m.get("mean_speed_xy_m_s"),
            "distance_travelled_m": m.get("distance_travelled_m"),
            "rms_lin_vel_error_m_s": m.get("rms_lin_vel_error_m_s"),
            "mean_mech_power_w": m.get("mean_mech_power_w"),
            "mean_height_m": m.get("mean_height_m"),
            "path": str(pj.parent.relative_to(OUT)),
        })
    if not rows:
        return
    rows.sort(key=lambda r: (r["policy"] or "", -(r["yield_stress_pa"] or 0), r["seed"] or 0))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
