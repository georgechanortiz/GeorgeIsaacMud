"""Scripted single-leg tests in MPM mud: is dragging a foot cheaper than pulling it out?

The A2's base is bolted in the air above a box of mud, and its front-right leg tracks a scripted
foot path with joint PID control (the trained kp/kd plus an integral term). The other legs are
tucked out of the mud. Each test starts from the same mud (scene state restored) and the foot
moves along straight 1D segments with a minimum-jerk profile:

    drag            foot at drag height, moved forward by --stroke              (horizontal, 1D)
    intrude_retract foot pushed down from above the surface to drag height,
                    held, then pulled straight back up                          (vertical, 1D)
    stepover        the alternative to drag: lift out, carry forward --stroke
                    above the surface, plunge back down to drag height          (three 1D segments)

drag and stepover start and end at the same foot positions, so their energies compare directly.

    python scripts/eval_a2_PD_controlled_tests.py                     # calibrated 4 in mud
    python scripts/eval_a2_PD_controlled_tests.py --mud-E 4e5         # stiffer mud
    python scripts/eval_a2_PD_controlled_tests.py --foot-speed 1.0 --tests drag stepover
    python scripts/eval_a2_PD_controlled_tests.py --no-mud            # same paths in air (leg-only cost)

Energy is reported three ways per phase, all over the leg's three joints at 200 Hz:
    joint_energy_abs_j  sum |tau * qdot| dt   (motors don't regenerate; the metric behind CoT)
    joint_energy_pos_j  sum max(tau * qdot, 0) dt
    mud_work_j          sum -F_mud . v_foot dt (work the leg does on the mud; F_mud is the
                        coupling force on the calf link, which carries the foot)

Writes reports/pd_tests/<NNN>_pd_tests/ with summary.csv/json, per-test plots and videos, and a
row in reports/index.csv.
"""

import argparse
import csv
import inspect
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import transform_by_quat

from envs import mud
from envs.unitree_env import dc_motor_torque_limits, resolve_urdf
from scripts.unitree_policy import cli
from scripts.unitree_policy.reporting import INDEX_FIELDS, _claim_report_dir

DEFAULT_POLICY = "Policies/v7_rom/model_best.pt"  # only its env.yaml is used: URDF, gains, limits
LEG = "FR"
THIGH_LEN = CALF_LEN = 0.275  # m, from the URDF
FOOT_RADIUS = 0.032
TUCK = {"thigh": 1.4, "calf": -2.6}  # the other three legs, feet well above the mud
CONTROL_DT = 0.02
TESTS = ("drag", "intrude_retract", "stepover")
# reference categorical palette (dataviz skill), fixed order
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_policy_arg(p, DEFAULT_POLICY)
    cli.add_cpu_arg(p, "MPM")
    p.add_argument("--tests", nargs="+", choices=TESTS, default=list(TESTS))
    p.add_argument("--foot-speed", type=float, default=0.5, metavar="M_S", help="average foot speed along each segment")
    p.add_argument("--stroke", type=float, default=0.20, metavar="M", help="horizontal foot travel for drag and stepover")
    p.add_argument("--drag-height", type=float, default=0.045, metavar="M",
                   help="foot-centre height while in the mud (foot radius is 0.032, so 0.045 clears the floor by 13 mm)")
    p.add_argument("--clearance", type=float, default=0.04, metavar="M", help="foot-centre height above the mud surface when out")
    p.add_argument("--hold", type=float, default=0.15, metavar="S", help="pause at the bottom of intrude_retract")
    p.add_argument("--base-height", type=float, default=0.40, metavar="M", help="height of the fixed base")
    p.add_argument("--gain-scale", type=float, default=1.0, help="scale the trained kp and kd")
    p.add_argument("--ki-ratio", type=float, default=5.0, metavar="PER_S",
                   help="integral gain as ki = ratio * kp; 0 turns it into plain PD")
    p.add_argument("--mud-depth", type=float, default=0.1016, metavar="M")
    p.add_argument("--mud-E", type=float, default=1e5, metavar="PA")
    p.add_argument("--mud-nu", type=float, default=0.35)
    p.add_argument("--mud-rho", type=float, default=1500.0)
    p.add_argument("--coup-friction", type=float, default=0.4)
    p.add_argument("--substeps", type=int, default=60, help="MPM substeps per 0.02 s (multiple of 4)")
    p.add_argument("--grid_density", type=int, default=48)
    p.add_argument("--no-mud", action="store_true", help="run the same paths in air")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--desc", type=str, default=None)
    p.add_argument("--out-root", type=str, default=None, metavar="DIR")
    return p.parse_args()


# ---- kinematics (sagittal plane of the leg, relative to the thigh joint) -------------------------


def leg_ik(x, z):
    """(thigh, calf) angles putting the foot centre at (x, z) from the thigh joint, knee bent backwards."""
    r2 = x * x + z * z
    c2 = (r2 - THIGH_LEN**2 - CALF_LEN**2) / (2 * THIGH_LEN * CALF_LEN)
    if not -1.0 <= c2 <= 1.0:
        raise ValueError(f"foot target ({x:.3f}, {z:.3f}) is out of the leg's reach")
    q2 = -math.acos(c2)
    q1 = math.atan2(-x, -z) - math.atan2(CALF_LEN * math.sin(q2), THIGH_LEN + CALF_LEN * math.cos(q2))
    return q1, q2


def min_jerk(p0, p1, n):
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    return np.asarray(p0) + np.outer(blend, np.asarray(p1) - np.asarray(p0))


def hold(p, n):
    return np.repeat(np.asarray(p, dtype=float)[None], n, axis=0)


# ---- scene --------------------------------------------------------------------------------------


class LegRig:
    def __init__(self, args, env_cfg):
        self.args = args
        self.sim_dt = CONTROL_DT / 4
        self.mud_on = not args.no_mud

        hip = (0.25944, -0.075113)
        self.thigh_xy = (hip[0], hip[1] - 0.12779)  # thigh joint, base at the origin
        self.x0 = self.thigh_xy[0] - args.stroke / 2  # foot start
        self.surface = args.mud_depth if self.mud_on else args.mud_depth  # same paths with and without mud
        self.z_mud = args.drag_height
        self.z_air = args.mud_depth + args.clearance

        fx, fy = self.thigh_xy
        dx = 1.0 / args.grid_density
        lower = (self.x0 - 0.25, fy - 0.18, -0.1)
        upper = (self.x0 + args.stroke + 0.30, fy + 0.18, args.mud_depth + 0.22)
        inset = 3.5 * dx
        self.mpm_cfg = {
            "grid_density": args.grid_density, "coup_friction": args.coup_friction, "coup_softness": 0.01,
            "wall_height": 0.0, "wall_thickness": 0.0, "bounds_lower": lower, "bounds_upper": upper,
            "patches": [{
                "name": "mud", "material": "liquid", "E": args.mud_E, "nu": args.mud_nu, "rho": args.mud_rho,
                "viscous": True,
                "patch_lower": (lower[0] + inset, lower[1] + inset, 0.0),
                "patch_upper": (upper[0] - inset, upper[1] - inset, args.mud_depth),
                "color": (0.55, 0.42, 0.30),
            }],
        }

        rigid_options = gs.options.RigidOptions(enable_self_collision=False, tolerance=1e-5, batch_dofs_info=True)
        kwargs = dict(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=args.substeps // 4),
            rigid_options=rigid_options,
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=False,
        )
        material = mud.coupling_material(self.mpm_cfg) if self.mud_on else None
        if self.mud_on:
            kwargs["mpm_options"] = mud.mpm_options(self.mpm_cfg)
        self.scene = gs.Scene(**kwargs)
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True), material=material)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=resolve_urdf(env_cfg.get("urdf")), pos=(0.0, 0.0, args.base_height), fixed=True),
            material=material,
        )
        if self.mud_on:
            mud.add_mud_patches(self.scene, self.mpm_cfg)
        self.cam = None
        if not args.no_video:
            cx = self.x0 + args.stroke / 2
            self.cam = self.scene.add_camera(res=(1280, 720), pos=(cx + 0.15, fy - 1.05, 0.32),
                                             lookat=(cx, fy, 0.10), fov=40, GUI=False)
        self.scene.build(n_envs=1)

        names = env_cfg["joint_names"]
        self.dofs = [self.robot.get_joint(n).dofs_idx_local[0] for n in names]
        self.leg = [self.robot.get_joint(f"{LEG}_{j}_joint").dofs_idx_local[0] for j in ("hip", "thigh", "calf")]
        scale = args.gain_scale
        self.kp = np.array([env_cfg["kp"][f"{LEG}_{j}_joint"] for j in ("hip", "thigh", "calf")]) * scale
        self.kd = np.array([env_cfg["kd"][f"{LEG}_{j}_joint"] for j in ("hip", "thigh", "calf")]) * scale
        self.robot.set_dofs_kp([env_cfg["kp"][n] * scale for n in names], self.dofs)
        self.robot.set_dofs_kv([env_cfg["kd"][n] * scale for n in names], self.dofs)
        h = self.sim_dt / (args.substeps // 4)
        arm = float(env_cfg.get("armature", 0.0))
        self.robot.set_dofs_armature([max(arm - env_cfg["kd"][n] * scale * h, 0.0) for n in names], self.dofs)
        as_t = lambda v: torch.tensor(v, dtype=gs.tc_float, device=gs.device)
        self.effort = as_t([env_cfg["effort_limit"][n] for n in names])
        self.saturation = as_t([env_cfg.get("saturation_effort", env_cfg["effort_limit"])[n] for n in names])
        self.vel_limit = as_t([env_cfg["velocity_limit"][n] for n in names])
        self.leg_t = torch.tensor(self.leg, dtype=gs.tc_int, device=gs.device)
        self.dofs_t = torch.tensor(self.dofs, dtype=gs.tc_int, device=gs.device)

        self.calf = self.robot.get_link(f"{LEG}_calf")
        self.tuck_targets = {}
        for n, d in zip(names, self.dofs):
            if not n.startswith(LEG):
                self.tuck_targets[d] = 0.0 if "hip" in n else TUCK["thigh" if "thigh" in n else "calf"]

        # start every test from this state: leg above the start point, mud undisturbed
        q_start = self.leg_q(self.x0, self.z_air)
        self.pose(q_start)
        self.check_fk()
        self.init_state = self.scene.get_state()

    # -- helpers --
    def leg_q(self, x, z):
        q1, q2 = leg_ik(x - self.thigh_xy[0], z - self.args.base_height)
        return np.array([0.0, q1, q2])

    def pose(self, q_leg):
        targets = dict(self.tuck_targets)
        targets.update(zip(self.leg, q_leg))
        dofs = list(targets)
        self.robot.set_dofs_position(torch.tensor([[targets[d] for d in dofs]], dtype=gs.tc_float, device=gs.device), dofs)

    def foot_pos(self):
        pos = self.calf.get_pos()[0]
        quat = self.calf.get_quat()[0]
        offset = torch.tensor([0.0, 0.0, -CALF_LEN], dtype=gs.tc_float, device=gs.device)
        return (pos + transform_by_quat(offset, quat)).cpu().numpy()

    def check_fk(self):
        foot = self.foot_pos()
        want = np.array([self.x0, self.thigh_xy[1], self.z_air])
        err = np.linalg.norm(foot - want)
        if err > 2e-3:
            raise RuntimeError(f"leg IK/FK mismatch: foot at {foot}, wanted {want} ({err * 1000:.1f} mm)")

    def mud_force(self):
        if not self.mud_on:
            return np.zeros(3)
        return mud.coupling_force(self.scene, [self.calf.idx])[0, 0]

    # -- one test --
    def run(self, name, segments, video_path=None):
        """segments: list of (phase, foot path (n, 2) in x/z, measured). Returns per-step arrays."""
        self.scene.reset(state=self.init_state)
        q_ref = []
        phases = []
        measured = []
        for phase, path, meas in segments:
            for x, z in path:
                q_ref.append(self.leg_q(x, z))
                phases.append(phase)
                measured.append(meas)
        q_ref = np.asarray(q_ref)
        qd_ref = np.gradient(q_ref, self.sim_dt, axis=0)
        path_xz = np.concatenate([seg[1] for seg in segments])

        streaming = False
        if self.cam is not None and video_path is not None:
            streaming = "save_to_filename" in inspect.signature(self.cam.start_recording).parameters
            if streaming:
                self.cam.start_recording(save_to_filename=str(video_path), fps=round(1 / self.sim_dt))
            else:
                self.cam.start_recording()

        ki = self.args.ki_ratio * self.kp
        integral = np.zeros(3)
        rows = {k: [] for k in ("t", "foot", "foot_ref", "tau", "qd", "q", "q_ref", "force")}
        prev_foot = self.foot_pos()
        for i in range(len(q_ref)):
            q = self.robot.get_dofs_position(self.leg_t)[0].cpu().numpy()
            err = q_ref[i] - q
            integral = np.clip(integral + err * self.sim_dt, -0.3 * self.kp / np.maximum(ki, 1e-9), 0.3 * self.kp / np.maximum(ki, 1e-9))
            # implicit PD toward an offset target gives tau = kp e + kd (qd_ref - qd) + ki * integral
            target_leg = q_ref[i] + (self.kd / self.kp) * qd_ref[i] + (ki / self.kp) * integral
            targets = dict(self.tuck_targets)
            targets.update(zip(self.leg, target_leg))
            dofs = list(targets)
            self.robot.control_dofs_position(
                torch.tensor([[targets[d] for d in dofs]], dtype=gs.tc_float, device=gs.device), dofs)
            lo, hi = dc_motor_torque_limits(self.robot.get_dofs_velocity(self.dofs_t), self.saturation, self.effort, self.vel_limit)
            self.robot.set_dofs_force_range(lo, hi, self.dofs_t)
            self.scene.step()

            foot = self.foot_pos()
            rows["t"].append(i * self.sim_dt)
            rows["foot"].append(foot)
            rows["foot_ref"].append(path_xz[i])
            rows["tau"].append(self.robot.get_dofs_control_force(self.leg_t)[0].cpu().numpy())
            rows["qd"].append(self.robot.get_dofs_velocity(self.leg_t)[0].cpu().numpy())
            rows["q"].append(self.robot.get_dofs_position(self.leg_t)[0].cpu().numpy())
            rows["q_ref"].append(q_ref[i])
            rows["force"].append(self.mud_force())
            prev_foot = foot
            if self.cam is not None and video_path is not None and not streaming and i % 4 == 0:
                self.cam.render()
            if not np.all(np.isfinite(foot)):
                raise RuntimeError(f"{name}: simulation went unstable at t = {i * self.sim_dt:.2f} s")

        if self.cam is not None and video_path is not None:
            if streaming:
                self.cam.stop_recording()
            else:
                self.cam.stop_recording(save_to_filename=str(video_path), fps=round(1 / CONTROL_DT))

        data = {k: np.asarray(v) for k, v in rows.items()}
        data["phase"] = np.asarray(phases)
        data["measured"] = np.asarray(measured)
        data["foot_vel"] = np.gradient(data["foot"], self.sim_dt, axis=0)
        data["power"] = (data["tau"] * data["qd"]).sum(axis=1)
        data["mud_power"] = -(data["force"] * data["foot_vel"]).sum(axis=1)
        return data

    # -- the three tests --
    def segments(self, name):
        a = self.args
        n = lambda dist: max(int(round(dist / a.foot_speed / self.sim_dt)), 2)
        start_air, start_mud = (self.x0, self.z_air), (self.x0, self.z_mud)
        end_air, end_mud = (self.x0 + a.stroke, self.z_air), (self.x0 + a.stroke, self.z_mud)
        vert = self.z_air - self.z_mud
        settle = ("settle", hold(start_air, int(0.3 / self.sim_dt)), False)
        if name == "drag":
            return [settle,
                    ("pre_intrude", min_jerk(start_air, start_mud, n(vert)), False),
                    ("pre_hold", hold(start_mud, int(0.15 / self.sim_dt)), False),
                    ("drag", min_jerk(start_mud, end_mud, n(a.stroke)), True),
                    ("post_hold", hold(end_mud, int(0.1 / self.sim_dt)), False)]
        if name == "intrude_retract":
            return [settle,
                    ("intrude", min_jerk(start_air, start_mud, n(vert)), True),
                    ("hold", hold(start_mud, max(int(a.hold / self.sim_dt), 1)), True),
                    ("retract", min_jerk(start_mud, start_air, n(vert)), True),
                    ("post_hold", hold(start_air, int(0.1 / self.sim_dt)), False)]
        if name == "stepover":
            return [settle,
                    ("pre_intrude", min_jerk(start_air, start_mud, n(vert)), False),
                    ("pre_hold", hold(start_mud, int(0.15 / self.sim_dt)), False),
                    ("lift", min_jerk(start_mud, start_air, n(vert)), True),
                    ("carry", min_jerk(start_air, end_air, n(a.stroke)), True),
                    ("plunge", min_jerk(end_air, end_mud, n(vert)), True),
                    ("post_hold", hold(end_mud, int(0.1 / self.sim_dt)), False)]
        raise ValueError(name)


# ---- metrics and plots --------------------------------------------------------------------------


def phase_metrics(test, data, dt, mask, phase):
    p = data["power"][mask]
    f = data["force"][mask]
    disp = data["foot"][mask][-1] - data["foot"][mask][0] if mask.sum() > 1 else np.zeros(3)
    track = np.linalg.norm(data["foot"][mask][:, [0, 2]] - data["foot_ref"][mask], axis=1)
    return {
        "test": test,
        "phase": phase,
        "duration_s": float(mask.sum() * dt),
        "foot_dx_m": float(disp[0]),
        "foot_dz_m": float(disp[2]),
        "joint_energy_abs_j": float(np.abs(p).sum() * dt),
        "joint_energy_pos_j": float(np.clip(p, 0, None).sum() * dt),
        "joint_energy_net_j": float(p.sum() * dt),
        "mud_work_j": float(data["mud_power"][mask].sum() * dt),
        "mean_abs_fx_n": float(np.abs(f[:, 0]).mean()),
        "max_fz_n": float(f[:, 2].max()),
        "min_fz_n": float(f[:, 2].min()),
        "peak_abs_torque_nm": float(np.abs(data["tau"][mask]).max()),
        "rms_tracking_error_mm": float(np.sqrt((track**2).mean()) * 1000),
    }


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_test(name, data, out_dir, surface):
    meas = data["measured"]
    t = data["t"] - data["t"][meas][0]
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(f"{name}: foot path, mud force, joint power (t = 0 at start of measured motion)", color=INK, fontsize=12)

    ax = axes[0]
    ax.plot(t, data["foot_ref"][:, 0], color=C_BLUE, linewidth=1.2, linestyle="--", label="x reference")
    ax.plot(t, data["foot"][:, 0], color=C_BLUE, linewidth=2, label="x")
    ax.plot(t, data["foot_ref"][:, 1], color=C_ORANGE, linewidth=1.2, linestyle="--", label="z reference")
    ax.plot(t, data["foot"][:, 2], color=C_ORANGE, linewidth=2, label="z")
    ax.axhline(surface, color=INK2, linewidth=1, linestyle=":")
    ax.text(t[0], surface, " mud surface", color=INK2, fontsize=8, va="bottom")
    ax.set_ylabel("foot centre (m)", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=4)

    ax = axes[1]
    ax.plot(t, data["force"][:, 0], color=C_BLUE, linewidth=2, label="Fx")
    ax.plot(t, data["force"][:, 2], color=C_ORANGE, linewidth=2, label="Fz")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylabel("mud force on\ncalf + foot (N)", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[2]
    ax.plot(t, data["power"], color=C_BLUE, linewidth=2, label="joint power tau.qdot")
    ax.plot(t, data["mud_power"], color=C_AQUA, linewidth=2, label="power into mud -F.v")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylabel("power (W)", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[3]
    dt = data["t"][1] - data["t"][0]
    e_abs = np.cumsum(np.abs(data["power"]) * meas) * dt
    e_mud = np.cumsum(data["mud_power"] * meas) * dt
    ax.plot(t, e_abs, color=C_BLUE, linewidth=2, label="joint energy sum|tau.qdot|")
    ax.plot(t, e_mud, color=C_AQUA, linewidth=2, label="work on mud")
    ax.set_ylabel("cumulative, measured\nphases only (J)", color=INK2)
    ax.set_xlabel("time (s)", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    # shade measured phases and label them
    for a in axes:
        style(a)
        for phase in dict.fromkeys(data["phase"][meas]):
            m = (data["phase"] == phase) & meas
            a.axvspan(t[m][0], t[m][-1], color=GRID, alpha=0.5, linewidth=0)
    for phase in dict.fromkeys(data["phase"][meas]):
        m = (data["phase"] == phase) & meas
        axes[0].text((t[m][0] + t[m][-1]) / 2, axes[0].get_ylim()[1], phase, ha="center", va="top", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def plot_comparison(rows, out_dir, stroke):
    """Drag vs stepover over the same stroke: joint energy split by phase, and work on the mud."""
    by_test = {}
    for r in rows:
        if r["test"] in ("drag", "stepover") and r["phase"] != "total":
            by_test.setdefault(r["test"], []).append(r)
    if len(by_test) < 2:
        return
    colors = {"drag": C_BLUE, "lift": C_ORANGE, "carry": C_AQUA, "plunge": C_YELLOW}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle(f"Moving the foot {stroke * 100:.0f} cm forward: drag through the mud vs lift over it", color=INK, fontsize=12)
    for ax, key, label in ((axes[0], "joint_energy_abs_j", "joint energy sum|tau.qdot| (J)"),
                           (axes[1], "mud_work_j", "work done on the mud (J)")):
        for i, test in enumerate(("drag", "stepover")):
            bottom = 0.0
            for r in by_test[test]:
                v = r[key]
                ax.bar(i, v, bottom=bottom, width=0.55, color=colors[r["phase"]], edgecolor="white", linewidth=2,
                       label=r["phase"] if ax is axes[0] else None)
                bottom += v
            ax.text(i, bottom, f"{bottom:.1f} J", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks([0, 1], ["drag", "lift - carry - plunge"])
        ax.set_ylabel(label, color=INK2)
        style(ax)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_dir / "drag_vs_stepover.png", dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    if args.substeps % 4:
        sys.exit("--substeps must be a multiple of 4")
    _, env_cfg, *_ = cli.load_run(args.policy)
    cli.init_genesis(args.cpu)

    root = Path(args.out_root) if args.out_root else REPO_ROOT / "reports"
    rig = LegRig(args, env_cfg)
    out_dir = _claim_report_dir(root, "pd_tests", "pd_tests")
    print(f"writing to {out_dir}")

    rows, wall0, sim_s = [], time.perf_counter(), 0.0
    for name in args.tests:
        t0 = time.perf_counter()
        data = rig.run(name, rig.segments(name), None if args.no_video else out_dir / f"{name}.mp4")
        sim_s += data["t"][-1]
        np.savez(out_dir / f"{name}.npz", **data)
        meas = data["measured"]
        phases = list(dict.fromkeys(data["phase"][meas]))
        for phase in phases:
            rows.append(phase_metrics(name, data, rig.sim_dt, (data["phase"] == phase) & meas, phase))
        if len(phases) > 1:
            rows.append(phase_metrics(name, data, rig.sim_dt, meas, "total"))
        plot_test(name, data, out_dir, args.mud_depth)
        print(f"{name}: {data['t'][-1]:.2f} s sim in {time.perf_counter() - t0:.0f} s wall")

    plot_comparison(rows, out_dir, args.stroke)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    mud_txt = "air (no mud)" if args.no_mud else f"liquid mud E {args.mud_E:.0e}, {args.mud_depth * 100:.1f} cm deep"
    settings = ("air" if args.no_mud else f"mud-liquid_d{args.mud_depth:g}_E{args.mud_E:.0e}_rho{args.mud_rho:g}_ss{args.substeps}".replace("e+0", "e")) + \
        f"_v{args.foot_speed:g}_stroke{args.stroke:g}_h{args.drag_height:g}_ki{args.ki_ratio:g}"
    description = args.desc or f"FR leg PID tests ({', '.join(args.tests)}) in {mud_txt}, foot {args.foot_speed:g} m/s"
    params = {
        "report": out_dir.name.split("_")[0], "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "description": description, "settings": settings, "args": vars(args),
        "leg": LEG, "kp": rig.kp.tolist(), "kd": rig.kd.tolist(), "ki": (args.ki_ratio * rig.kp).tolist(),
        "mud_surface_z": args.mud_depth, "foot_z_in_mud": rig.z_mud, "foot_z_out": rig.z_air,
        "mpm_cfg": rig.mpm_cfg if not args.no_mud else None,
        "wall_time_to_sim_time_ratio": (time.perf_counter() - wall0) / max(sim_s, 1e-9),
        "results": rows,
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))

    index = root / "index.csv"
    new = not index.exists()
    with index.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: "" for k in INDEX_FIELDS} | {
            "report": params["report"], "date": params["date"], "policy": "pd_tests",
            "description": description, "settings": settings,
            "folder": out_dir.relative_to(root).as_posix()})

    print()
    cols = ("test", "phase", "duration_s", "foot_dx_m", "foot_dz_m", "joint_energy_abs_j", "joint_energy_pos_j",
            "mud_work_j", "mean_abs_fx_n", "max_fz_n", "min_fz_n", "peak_abs_torque_nm", "rms_tracking_error_mm")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    print(f"\nreport written to {out_dir}")


if __name__ == "__main__":
    main()
