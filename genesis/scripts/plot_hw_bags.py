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

usage: python scripts/plot_hw_bags.py [bag_dir] [out_dir]
    bag_dir defaults to hardware_bags/, out_dir to HW_bag_plots/. Each bag's figures go to
    <out_dir>/<bag name>/. Needs `pip install mcap mcap-ros2-support` (ROS is not required).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

REPO = Path(__file__).resolve().parents[1]

# quad-sdk robot_driver UnitreeInterface::kLegMap, flattened: column i of the re-indexed array
# is Unitree motor QUAD_SDK_ORDER[i]. After it, quad-sdk's plot_state indices apply unchanged.
QUAD_SDK_ORDER = [3, 4, 5, 9, 10, 11, 0, 1, 2, 6, 7, 8]
LEG_NAMES = ["FL", "RL", "FR", "RR"]
JOINT_INDEX = {"abad": [0, 3, 6, 9], "hip": [1, 4, 7, 10], "knee": [2, 5, 8, 11]}
JOINT_LABEL = {"abad": "Ab/Ad (hip)", "hip": "Hip (thigh)", "knee": "Knee (calf)"}

# Categorical slots 1-4 of the dataviz reference palette, in fixed leg order. quad-sdk's own CMU
# colours put FL red next to FR green, which colour-blind readers cannot separate.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

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


def read_bag(path):
    """Decode /lowstate, /lowcmd and /sportmodestate into arrays, joints in quad-sdk order."""
    d = {k: [] for k in ("t", "q", "dq", "tau", "rpy", "gyro", "cmd_t", "q_cmd", "kp", "sm_t", "pos", "vel")}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, msg, m in reader.iter_decoded_messages(topics=["/lowstate", "/lowcmd", "/sportmodestate"]):
            t = msg.log_time * 1e-9
            if ch.topic == "/lowstate":
                ms = [m.motor_state[i] for i in QUAD_SDK_ORDER]
                d["t"].append(t)
                d["q"].append([x.q for x in ms])
                d["dq"].append([x.dq for x in ms])
                d["tau"].append([x.tau_est for x in ms])
                d["rpy"].append(list(m.imu_state.rpy))
                d["gyro"].append(list(m.imu_state.gyroscope))
            elif ch.topic == "/lowcmd":
                mc = [m.motor_cmd[i] for i in QUAD_SDK_ORDER]
                d["cmd_t"].append(t)
                d["q_cmd"].append([x.q for x in mc])
                d["kp"].append([x.kp for x in mc])
            else:
                d["sm_t"].append(t)
                d["pos"].append(list(m.position))
                d["vel"].append(list(m.velocity))
    d = {k: np.asarray(v, dtype=float) for k, v in d.items()}
    t0 = d["t"][0]
    for k in ("t", "cmd_t", "sm_t"):
        d[k] -= t0
    # A motor with kp = 0 is not position-controlled, and Unitree's "no target" value (PosStopF,
    # 2.146e9) is not an angle: neither is a reference worth drawing.
    d["q_cmd"][(d["kp"] == 0) | (np.abs(d["q_cmd"]) > 10)] = np.nan
    d["pos"] -= d["pos"][0]
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
    d = read_bag(path)
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
    bags = sorted(bag_dir.glob("*/*.mcap"))
    if not bags:
        sys.exit(f"no */*.mcap bags under {bag_dir}")
    for bag in bags:
        plot_bag(bag, out_root / bag.parent.name)


if __name__ == "__main__":
    main()
