"""Plot every A2 hardware bag the way quad-sdk's quad_logger plots a run.

quad-sdk (quad_logger/scripts/plot_state.py) draws joint positions, velocities and efforts as
three figures, each with an Ab/Ad, Hip and Knee panel and one line per leg, and process_log.py
overlays the reference on the same axes with a dashed line. Its UnitreeInterface fills those
joint states from /lowstate: position = q, velocity = dq, effort = tau_est, re-indexed from the
Unitree motor order (FR, FL, RR, RL) into quad-sdk's leg order (FL, RL, FR, RR) with kLegMap.
This script does the same from the recorded bags, and draws the commanded joint angle from
/lowcmd as the dashed reference. /lowcmd's velocity target and feed-forward torque are zero
throughout (pure PD on position), so velocity and effort have no reference line.

Joint names differ between the two conventions: quad-sdk's Ab/Ad, Hip and Knee are the A2's
hip, thigh and calf, so the panels carry both.

Body states follow quad-sdk's linear_states and angular_states figures, from /sportmodestate
(odometry position in the world frame, velocity in the body frame: X is forward speed) and the
/lowstate IMU (roll, pitch, yaw and gyro). Foot position and velocity plots are left out:
/sportmodestate's foot fields are zero in these bags.

Time is seconds since the first /lowstate message, on the bag's log clock.

usage: python scripts/hardware/plot_hw_bags.py [bag_dir] [out_dir]
    bag_dir defaults to hardware_bags/, out_dir to HW_bag_plots/. Each bag's figures go to
    <out_dir>/<bag name>/. Needs `pip install mcap mcap-ros2-support` (ROS is not required).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.hardware.bags import find_bags, read_bag  # noqa: E402
from scripts.unitree_policy.plots import GRID, INK, INK_MUTED, LEG_SERIES as SERIES  # noqa: E402

# quad-sdk robot_driver UnitreeInterface::kLegMap, flattened: column i of the re-indexed array
# is Unitree motor QUAD_SDK_ORDER[i]. After it, quad-sdk's plot_state indices apply unchanged.
QUAD_SDK_ORDER = [3, 4, 5, 9, 10, 11, 0, 1, 2, 6, 7, 8]
LEG_NAMES = ["FL", "RL", "FR", "RR"]
JOINT_INDEX = {"abad": [0, 3, 6, 9], "hip": [1, 4, 7, 10], "knee": [2, 5, 8, 11]}
JOINT_LABEL = {"abad": "Ab/Ad (hip)", "hip": "Hip (thigh)", "knee": "Knee (calf)"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.size": 10,
    "axes.titlesize": 12,
    "legend.frameon": False,
})


def load_bag(path):
    """A decoded bag (scripts/hardware/bags.py) re-indexed into quad-sdk's joint order.

    Times become seconds since the first /lowstate message; odometry position becomes
    displacement from the start.
    """
    raw = read_bag(path)
    joints = lambda a: a[:, QUAD_SDK_ORDER]
    d = {
        "t": raw["ls_t"], "q": joints(raw["q"]), "dq": joints(raw["dq"]), "tau": joints(raw["tau"]),
        "rpy": raw["rpy"], "gyro": raw["gyro"],
        "cmd_t": raw["lc_t"], "q_cmd": joints(raw["q_cmd"]), "kp": joints(raw["kp"]),
        "sm_t": raw["sm_t"], "pos": raw["pos"], "vel": raw["vel"],
    }
    t0 = d["t"][0]
    for k in ("t", "cmd_t", "sm_t"):
        d[k] = d[k] - t0
    # A motor with kp = 0 is not position-controlled, and Unitree's "no target" value (PosStopF,
    # 2.146e9) is not an angle: neither is a reference worth drawing.
    d["q_cmd"][(d["kp"] == 0) | (np.abs(d["q_cmd"]) > 10)] = np.nan
    d["pos"] = d["pos"] - d["pos"][0]
    return d


def _finish(fig, axes, title, handles=()):
    for ax in axes:
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.margins(y=0.05)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    if handles:
        fig.legend(handles=handles, loc="upper right", ncol=len(handles), bbox_to_anchor=(0.995, 0.995),
                   labelcolor=INK)
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def _leg_handles(extra=()):
    return [Line2D([], [], color=c, lw=2, label=n) for c, n in zip(SERIES, LEG_NAMES)] + list(extra)


def plot_joints(t, x, unit, title, ref_t=None, ref=None):
    """quad-sdk plot_state's joint figure: Ab/Ad, Hip, Knee panels, one line per leg."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (joint, idx) in zip(axes, JOINT_INDEX.items()):
        for leg, i in enumerate(idx):
            ax.plot(t, x[:, i], color=SERIES[leg], lw=1.0)
            if ref is not None:
                ax.plot(ref_t, ref[:, i], color=SERIES[leg], lw=1.0, ls="--")
        ax.set_ylabel(f"{JOINT_LABEL[joint]} ({unit})")
    extra = []
    if ref is not None:
        extra = [Line2D([], [], color=INK_MUTED, lw=1.0, label="measured"),
                 Line2D([], [], color=INK_MUTED, lw=1.0, ls="--", label="commanded")]
    _finish(fig, axes, title, _leg_handles(extra))
    return fig


def plot_series(t, rows, title):
    """Body-state figure: rows is a list of (data, labels, ylabel), one panel each."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 3 * len(rows)), sharex=True)
    for ax, (x, labels, ylabel) in zip(axes, rows):
        for k, label in enumerate(labels):
            ax.plot(t, x[:, k], color=SERIES[k], lw=1.0)
        ax.legend([Line2D([], [], color=SERIES[k], lw=2) for k in range(len(labels))], labels,
                  loc="upper left", ncol=len(labels), labelcolor=INK)
        ax.set_ylabel(ylabel)
    _finish(fig, axes, title)
    return fig


def plot_bag(path, out_dir):
    name = path.parent.name
    d = load_bag(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "joint_positions": plot_joints(d["t"], d["q"], "rad", f"{name} · Joint Angles",
                                       d["cmd_t"], d["q_cmd"]),
        "joint_velocities": plot_joints(d["t"], d["dq"], "rad/s", f"{name} · Joint Velocities"),
        "joint_effort": plot_joints(d["t"], d["tau"], "N·m", f"{name} · Joint Effort (tau_est)"),
        "linear_states": plot_series(d["sm_t"], [
            (d["pos"][:, :2], ["X", "Y"], "World position from start (m)"),
            (d["pos"][:, 2:], ["Z"], "Height change (m)"),
            (d["vel"], ["X", "Y", "Z"], "Body velocity (m/s)"),
        ], f"{name} · Linear States (/sportmodestate)"),
        "angular_states": plot_series(d["t"], [
            (d["rpy"][:, :2], ["Roll", "Pitch"], "Roll, pitch (rad)"),
            (d["rpy"][:, 2:], ["Yaw"], "Yaw (rad)"),
            (d["gyro"], ["X", "Y", "Z"], "Ang. Vel. (rad/s)"),
        ], f"{name} · Angular States (IMU)"),
    }
    for stem, fig in figs.items():
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        plt.close(fig)
    print(f"{name}: {d['t'][-1]:.1f} s, {len(d['t'])} lowstate / {len(d['cmd_t'])} lowcmd msgs, "
          f"peak |tau_est| {np.abs(d['tau']).max():.0f} N·m -> {out_dir}")


def main():
    bag_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "hardware_bags"
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "HW_bag_plots"
    bags = find_bags(bag_dir)
    if not bags:
        sys.exit(f"no */*.mcap bags under {bag_dir}")
    for bag in bags:
        plot_bag(bag, out_root / bag.parent.name)


if __name__ == "__main__":
    main()
