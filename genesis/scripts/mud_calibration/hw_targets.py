"""Grass vs mud joint-torque/power targets from the hardware bags (walking segments only)."""
import sys, json, numpy as np
from pathlib import Path
GROUPS = {"front_hip": [0, 3], "front_thigh": [1, 4], "front_calf": [2, 5],
          "rear_hip": [6, 9], "rear_thigh": [7, 10], "rear_calf": [8, 11]}

def metrics(tau, dq, speed):
    out = {"speed": float(np.mean(speed)), "power_w": float(np.clip(tau * dq, 0, None).sum(1).mean())}
    for g, idx in GROUPS.items():
        a = np.abs(tau[:, idx])
        out[f"{g}_mean"] = float(a.mean()); out[f"{g}_p95"] = float(np.percentile(a, 95))
    return out

def walking(d):
    t0 = d["ls_t"][0]; st = d["sm_t"] - t0
    sp = np.linalg.norm(d["vel"][:, :2], axis=1); w = sp > 0.15
    lo, hi = st[w].min() + 1.0, st[w].max() - 0.5   # drop start-up and stop transients
    lt = d["ls_t"] - t0; m = (lt > lo) & (lt < hi)
    ms = (st > lo) & (st < hi)
    return d["tau"][m], d["dq"][m], sp[ms]

src = Path(sys.argv[1]); res = {}
grass = walking(np.load(src / "grasswalk.npz"))
muds = [walking(np.load(src / f"mudwalk{i}.npz")) for i in (2, 3, 4)]
res["grass"] = metrics(*grass)
res["mud"] = metrics(*(np.concatenate(x) for x in zip(*muds)))
res["mud_runs"] = [metrics(*m) for m in muds]
res["ratio"] = {k: res["mud"][k] / res["grass"][k] for k in res["grass"]}
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(f"{'metric':<18}{'grass':>8}{'mud':>8}{'ratio':>8}   per-run mud")
for k in res["grass"]:
    print(f"{k:<18}{res['grass'][k]:8.2f}{res['mud'][k]:8.2f}{res['ratio'][k]:8.2f}   " + " ".join(f"{r[k]:6.2f}" for r in res["mud_runs"]))
