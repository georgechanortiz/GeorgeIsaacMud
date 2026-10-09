"""Plot hardware runs in the same format as the sim reports' foot_forces.png and torques_by_leg.png.

torques_by_leg.png is /lowstate tau_est as recorded, averaged into the same 20 ms bins.

foot_forces.png needs more work.
The A2 bags carry no measured foot forces (/sportmodestate foot_force is zero throughout), so
they are estimated from the joint torques the bags do record. For each leg, the static balance
tau = -J(q)^T F gives the force F the ground (or mud) applies to the foot:

    F = -(J(q)^T)^-1 tau_est

with J the foot-position Jacobian from the A2 URDF (assets/a2/urdf/a2.urdf). That balance holds
only while the foot is on the ground. In swing there is no contact force, and the torque that
lifts and swings the leg is read as a phantom force instead: the short negative F_z dips of
-100 to -200 N at each lift-off. They appear on grass as much as in mud, so they are this
artifact, not mud suction. Read the plot in stance only. The result is rotated from the body frame into a gravity-aligned frame with the
IMU's roll and pitch. The heading is removed, so F_x is forward along the robot's heading, which
is how the sim walks (straight along +x). Samples are averaged into 20 ms bins, the sim's 50 Hz.

The estimated vertical forces sum to about 520 N on every run, grass and mud alike, against
the 393 N the URDF's 40.1 kg would weigh. The bias is the same in every run, so grass-vs-mud
comparisons hold, but absolute forces may read ~30% high. Likely causes are tau_est including
gearbox friction, or the robot carrying payload the URDF does not model.

usage: python scripts/mud_calibration/plot_hw_report.py <npz_dir> [out_dir]
    <npz_dir> is extract_bags.py's output. Plots go to <out_dir>/<run>/, default
    reports/hardware_bags/.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.unitree_policy.reporting import Reporter  # noqa: E402

# A2 URDF geometry. Hardware motor order is FR, FL, RR, RL, each (hip, thigh, calf), the same
# order as configs/a2_flat.py's joint_names.
HIP_X, HIP_Y, THIGH_Y, L_THIGH, L_CALF = 0.25944, 0.075113, 0.12779, 0.275, 0.275
LEGS = {"FR": (+1, -1), "FL": (+1, +1), "RR": (-1, -1), "RL": (-1, +1)}  # (front/rear, left/right) signs
MOTOR_SLICE = {"FR": slice(0, 3), "FL": slice(3, 6), "RR": slice(6, 9), "RL": slice(9, 12)}
JOINT_NAMES = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]
PANEL_ORDER = ("FL", "RL", "FR", "RR")  # same panel layout as the sim reports (Genesis link order)
CONTROL_DT = 0.02


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def foot_pos(q, side):
    """Foot position in the body frame for joint angles q = (hip, thigh, calf)."""
    hip = np.array([0.0, 0.0, 0.0])
    thigh_off = np.array([0.0, side * THIGH_Y, 0.0])
    down = np.array([0.0, 0.0, -L_THIGH])
    foot = np.array([0.0, 0.0, -L_CALF])
    return hip + _rx(q[0]) @ (thigh_off + _ry(q[1]) @ (down + _ry(q[2]) @ foot))


def jacobian(q, side, eps=1e-6):
    J = np.zeros((3, 3))
    for k in range(3):
        dq = np.zeros(3)
        dq[k] = eps
        J[:, k] = (foot_pos(q + dq, side) - foot_pos(q - dq, side)) / (2 * eps)
    return J


def level_rotation(quat):
    """Body-to-gravity-aligned rotation with the yaw removed. quat is (w, x, y, z)."""
    w, x, y, z = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R


def foot_forces(d):
    """(samples, 4 legs in PANEL_ORDER, 3) reaction force on each foot, gravity-aligned frame."""
    out = np.zeros((len(d["q"]), 4, 3))
    for i, (q, tau, quat) in enumerate(zip(d["q"], d["tau"], d["quat"])):
        R = level_rotation(quat)
        for j, leg in enumerate(PANEL_ORDER):
            sl = MOTOR_SLICE[leg]
            J = jacobian(q[sl], LEGS[leg][1])
            out[i, j] = R @ -np.linalg.solve(J.T, tau[sl])
    return out


def walking_window(d, lead_s=0.5):
    """Time span from just before the walk starts to when it stops, on the lowstate clock."""
    t0 = d["ls_t"][0]
    speed = np.linalg.norm(d["vel"][:, :2], axis=1)
    st = d["sm_t"][speed > 0.15] - t0
    return st.min() - lead_s, st.max()


def bin_50hz(t, x):
    edges = np.arange(t[0], t[-1], CONTROL_DT)
    idx = np.digitize(t, edges) - 1
    keep = np.bincount(idx, minlength=len(edges)) > 0
    sums = np.zeros((len(edges),) + x.shape[1:])
    np.add.at(sums, idx, x)
    counts = np.bincount(idx, minlength=len(edges)).reshape((-1,) + (1,) * (x.ndim - 1))
    return edges[keep] - edges[0], (sums / np.maximum(counts, 1))[keep]


def main():
    npz_dir = Path(sys.argv[1])
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "reports" / "hardware_bags"
    names = [f"{leg} foot" for leg in PANEL_ORDER]
    for f in sorted(npz_dir.glob("*.npz")):
        d = np.load(f)
        lo, hi = walking_window(d)
        t = d["ls_t"] - d["ls_t"][0]
        m = (t >= lo) & (t <= hi)
        sub = {k: d[k][m] for k in ("q", "tau", "quat")}
        tb, ff = bin_50hz(t[m] - lo, foot_forces(sub))
        _, tau = bin_50hz(t[m] - lo, sub["tau"])
        out_dir = out_root / f.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        plotter = SimpleNamespace(out_dir=out_dir)
        Reporter._plot_foot_forces(plotter, tb, ff, names)
        Reporter._plot_torques(plotter, tb, tau, JOINT_NAMES)
        walk = tb > 1.0  # skip the lead-in and first strides
        print(f"{f.stem}: {tb[-1]:.1f} s, mean total F_z {ff[walk, :, 2].sum(1).mean():.0f} N, "
              f"peak foot F_z {ff[walk, :, 2].max():.0f} N, mean F_z per foot "
              + " ".join(f"{n.split()[0]}={ff[walk, j, 2].mean():.0f}" for j, n in enumerate(names))
              + f", peak |tau| {np.abs(tau).max():.0f} N*m -> {out_dir}")


if __name__ == "__main__":
    main()
