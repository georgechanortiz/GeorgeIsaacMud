"""Grass vs mud joint-torque/power targets from the hardware bags (walking segments only).

usage: python scripts/mud_calibration/hw_targets.py <npz_dir> <out.json>
    <npz_dir> is extract_bags.py's output.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.mud_calibration.metrics import torque_metrics, walking_span  # noqa: E402


def walking(d):
    """(tau, dq, speed) over the steady part of the walk: start-up and stop transients dropped."""
    start, end = walking_span(d)
    lo, hi = start + 1.0, end - 0.5
    t0 = d["ls_t"][0]
    lt, st = d["ls_t"] - t0, d["sm_t"] - t0
    m = (lt > lo) & (lt < hi)
    ms = (st > lo) & (st < hi)
    speed = np.linalg.norm(d["vel"][:, :2], axis=1)
    return d["tau"][m], d["dq"][m], speed[ms]


def main():
    src, out_path = Path(sys.argv[1]), sys.argv[2]
    grass = walking(np.load(src / "grasswalk.npz"))
    muds = [walking(np.load(src / f"mudwalk{i}.npz")) for i in (2, 3, 4)]
    res = {
        "grass": torque_metrics(*grass),
        "mud": torque_metrics(*(np.concatenate(x) for x in zip(*muds))),
        "mud_runs": [torque_metrics(*m) for m in muds],
    }
    res["ratio"] = {k: res["mud"][k] / res["grass"][k] for k in res["grass"]}
    json.dump(res, open(out_path, "w"), indent=1)

    print(f"{'metric':<18}{'grass':>8}{'mud':>8}{'ratio':>8}   per-run mud")
    for k in res["grass"]:
        print(f"{k:<18}{res['grass'][k]:8.2f}{res['mud'][k]:8.2f}{res['ratio'][k]:8.2f}   "
              + " ".join(f"{r[k]:6.2f}" for r in res["mud_runs"]))


if __name__ == "__main__":
    main()
