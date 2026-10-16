"""Walking metrics shared by the hardware targets (hw_targets.py) and the sim rollouts (run_one.py)."""

import numpy as np

# Joint columns of each group, in the A2's motor order (FR, FL, RR, RL; hip, thigh, calf each).
GROUPS = {"front_hip": [0, 3], "front_thigh": [1, 4], "front_calf": [2, 5],
          "rear_hip": [6, 9], "rear_thigh": [7, 10], "rear_calf": [8, 11]}

WALK_SPEED = 0.15  # m/s; /sportmodestate planar speed above this counts as walking


def torque_metrics(tau, dq, speed):
    """Mean speed, mean positive mechanical power, and mean / 95th-percentile |tau| per joint group."""
    out = {"speed": float(np.mean(speed)), "power_w": float(np.clip(tau * dq, 0, None).sum(1).mean())}
    for g, idx in GROUPS.items():
        a = np.abs(tau[:, idx])
        out[f"{g}_mean"] = float(a.mean())
        out[f"{g}_p95"] = float(np.percentile(a, 95))
    return out


def walking_span(d):
    """(start, end) of a hardware walk, in seconds since its first /lowstate message.

    The first and last /sportmodestate samples whose planar speed exceeds WALK_SPEED.
    `d` is a decoded bag (scripts/hardware/bags.py) or its extract_bags.py npz.
    """
    t0 = d["ls_t"][0]
    speed = np.linalg.norm(d["vel"][:, :2], axis=1)
    t = d["sm_t"][speed > WALK_SPEED] - t0
    return t.min(), t.max()
