"""Decode each hardware bag into <out_dir>/<run>.npz for the calibration scripts.

usage: python scripts/mud_calibration/extract_bags.py <bag_dir> <out_dir>
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.hardware.bags import find_bags, read_bag, run_name  # noqa: E402

# What hw_targets.py and plot_hw_report.py read (see read_bag for the meaning of each).
NPZ_KEYS = ("ls_t", "q", "dq", "tau", "quat", "sm_t", "vel", "pos", "ff", "fpb", "lc_t", "q_cmd", "kp")


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    for bag in find_bags(src):
        d = read_bag(bag)
        arr = {k: d[k] for k in NPZ_KEYS}
        name = run_name(bag)
        np.savez(out / f"{name}.npz", **arr)
        print(name, {k: v.shape for k, v in arr.items()})


if __name__ == "__main__":
    main()
