"""Rank mud configs by how well (sim_mud - sim_flat) matches (hw_mud - hw_grass), in units of the hw grass value."""
import json
from pathlib import Path
S = Path(__file__).resolve().parent
H = json.load(open(S / "results/hw_targets.json")); hg, hm = H["grass"], H["mud"]
flat = json.load(open(S / "results/flat_s60.json"))
KEYS = ["speed", "power_w", "front_thigh_mean", "front_calf_mean", "front_calf_p95", "front_hip_mean",
        "rear_thigh_mean", "rear_calf_mean", "rear_hip_mean"]
short = dict(zip(KEYS, ["spd", "pow", "fTh", "fCa", "fCa95", "fHp", "rTh", "rCa", "rHp"]))
def delta(m, base, ref): return {k: (m[k] - base[k]) / ref[k] for k in KEYS}
target = delta(hm, hg, hg)
rows = []
for f in sorted((S / "results").glob("*.json")):
    r = json.load(open(f))
    if r.get("cfg", {}).get("mud") is None: continue
    if "speed" not in r: rows.append((9, r, None)); continue
    if r["cfg"]["mud"].get("rho", 1500) < 500: continue  # near-massless control run, not a mud candidate
    d = delta(r, flat, hg)
    # Ranked on the terms that carry the mud's load: calf torque (the joint that does the work in
    # both sim and hardware) and speed. Left out: front thigh and power, which come from suction
    # on pull-out that Genesis' MPM coupling cannot produce, and the hips / rear thigh, whose
    # change in mud is a gait adaptation of opposite sign in sim and on hardware.
    FIT = ["speed", "front_calf_mean", "front_calf_p95", "rear_calf_mean"]
    err = sum(abs(d[k] - target[k]) for k in FIT) / len(FIT)
    rows.append((err, r, d))
rows.sort(key=lambda x: x[0])
print("change vs flat, as a fraction of the hardware grass value")
print(f"{'config':<22}{'err':>6} " + "".join(f"{short[k]:>7}" for k in KEYS) + "   mudW  footz fell")
print(f"{'HARDWARE':<22}{'':>6} " + "".join(f"{target[k]:+7.2f}" for k in KEYS))
for err, r, d in rows:
    if d is None: print(f"{r['name']:<22}  no steady window (fell={r.get('fell')})"); continue
    print(f"{r['name']:<22}{err:6.3f} " + "".join(f"{d[k]:+7.2f}" for k in KEYS) + f"  {r['mud_weight_frac']:5.2f} {r['stance_foot_z_p10']:6.3f} {r['fell']}")
