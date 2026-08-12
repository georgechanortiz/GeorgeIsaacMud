"""Finite-horizon rollouts with plots and a parameter dump.

Enabled by --reporting on any eval entry point. Writes to

    reports/<experiment>/<YYYY-MM-DD_HH-MM-SS>/

where <experiment> is the checkpoint's parent-of-parent directory, mirroring how
logs/ is organised (go2_mud_flat, unitree_go2_flat_rma, ...), so reports for
different policies never mix and repeated runs never collide.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # report files only; never open a GUI alongside the genesis viewer

import matplotlib.pyplot as plt
import numpy as np
import torch

from genesis.utils.geom import quat_to_xyz

# Categorical slots 1-3 of the validated reference palette, used in fixed order.
# Three series max per axis here (hip/thigh/calf), so slots 4+ are never reached.
SERIES = ("#2a78d6", "#008300", "#e87ba4")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e0e0dc"


def _style_axes(ax, title, xlabel, ylabel):
    """Recessive grid and axes; all text in ink tokens, never a series colour."""
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


class Reporter:
    """Collects per-step state from env 0 and writes plots + params at the end."""

    def __init__(self, experiment, params, out_root=None, reports_dir="reports"):
        self.experiment = experiment
        self.params = params
        root = Path(out_root) if out_root else Path(reports_dir)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Every eval entry point sets params["run"] to the checkpoint's run directory name
        # (e.g. "Policy_B") -- suffix it onto the timestamp so report dirs are identifiable
        # without opening params.json.
        run = params.get("run")
        if run:
            stamp = f"{stamp}_{run}"
        self.out_dir = root / experiment / stamp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rows = {
            k: []
            for k in ("lin_vel", "ang_vel", "command", "torque", "dof_vel", "height", "pos", "yaw")
        }
        # Foot forces are only available on envs that expose get_foot_forces (the MPM sim);
        # kept out of _rows so other eval scripts (rigid, sph) record exactly as before.
        self._foot_force = []
        self._foot_names = None
        # phi (the adaptation module) is only available on RMA/privileged-obs actors that
        # went through stabilise_adaptation in eval_cfg.py; kept out of _rows for the same
        # reason as foot forces -- most policies never set last_e_t.
        self._phi_values = []
        self._phi_fresh = []
        self._recording = False
        self._wall_sim_s = None  # (wall_s, sim_s), set by run_rollout via note_rollout_timing

    def record(self, env, policy=None):
        """Append one step. Reads env 0 only -- reports describe a single robot.

        If the env was built with a recording camera (record=True), also capture a
        video frame here. start_recording is lazy so the very first frame is grabbed.
        """
        self._rows["lin_vel"].append(env.base_lin_vel[0].detach().cpu().numpy().copy())
        self._rows["ang_vel"].append(env.base_ang_vel[0].detach().cpu().numpy().copy())
        self._rows["command"].append(env.commands[0].detach().cpu().numpy().copy())
        torque = env.robot.get_dofs_control_force(env.motors_dof_idx)
        self._rows["torque"].append(torque[0].detach().cpu().numpy().copy())
        # paired with torque for joint power; env.dof_vel is refreshed in env.step
        self._rows["dof_vel"].append(env.dof_vel[0].detach().cpu().numpy().copy())
        self._rows["height"].append(float(env.base_pos[0, 2]))
        # world XY and yaw, so the report can say whether the walk actually held a line
        self._rows["pos"].append(env.base_pos[0, :2].detach().cpu().numpy().copy())
        self._rows["yaw"].append(float(quat_to_xyz(env.base_quat, rpy=True, degrees=False)[0, 2]))

        # Per-foot reaction force (world frame), when the env can supply it. Env 0 only.
        if hasattr(env, "get_foot_forces"):
            ff = np.asarray(env.get_foot_forces())  # (num_envs, num_feet, 3)
            self._foot_force.append(ff[0].copy())   # -> (num_feet, 3)
            if self._foot_names is None:
                self._foot_names = list(
                    getattr(env, "foot_names", None) or [f"foot {i}" for i in range(ff.shape[1])]
                )

        # phi's current estimate of the privileged latent e_t, and whether this step
        # recomputed it or reused a held value (see stabilise_adaptation in eval_cfg.py).
        if policy is not None and getattr(policy, "last_e_t", None) is not None:
            self._phi_values.append(policy.last_e_t[0].detach().cpu().numpy().copy())
            self._phi_fresh.append(bool(policy.last_e_t_fresh))

        if getattr(env, "cam", None) is not None:
            if not self._recording:
                env.cam.start_recording()
                self._recording = True
            env.update_camera()  # follow the robot
            env.cam.render()

    def note_rollout_timing(self, wall_s, sim_s):
        """Record how long the rollout took in wall-clock time vs. simulated time.

        Called by run_rollout once the loop finishes; finalize() writes the ratio into
        params.json so a report can be read back later to gauge how slow (or, with
        --headless, how much faster) that run's config was.
        """
        self._wall_sim_s = (wall_s, sim_s)

    def finalize(self, env):
        data = {k: np.asarray(v) for k, v in self._rows.items()}
        if len(data["lin_vel"]) == 0:
            raise RuntimeError("Reporter.finalize called with no recorded steps.")
        t = np.arange(len(data["lin_vel"])) * env.dt

        lin, ang, cmd = data["lin_vel"], data["ang_vel"], data["command"]
        err_x = lin[:, 0] - cmd[:, 0]
        err_y = lin[:, 1] - cmd[:, 1]
        err_lin = np.linalg.norm(lin[:, :2] - cmd[:, :2], axis=1)
        err_ang = ang[:, 2] - cmd[:, 2]

        # Straightness, measured against the ray leaving the first recorded pose along its own
        # heading -- the same reference the env's cross_track_error term uses. A mid-rollout
        # reset teleports the robot back to spawn, so it shows up here as a discontinuity.
        pos, yaw = data["pos"], data["yaw"]
        yaw0 = yaw[0]
        normal = np.array([-np.sin(yaw0), np.cos(yaw0)])
        cross_track = (pos - pos[0]) @ normal
        heading_err = np.degrees(np.arctan2(np.sin(yaw - yaw0), np.cos(yaw - yaw0)))

        # Mechanical joint power, sum|tau * omega|, matching the mech_power convention in
        # go2_mud/scripts/lib/metrics.py so these numbers are comparable to training-side
        # reports. Mechanical, not electrical: it excludes resistive heating in the motor
        # windings, which is the dominant cost when a leg pushes hard and moves slowly.
        p_mech = np.abs(data["torque"] * data["dof_vel"]).sum(axis=1)
        energy = float((p_mech * env.dt).sum())
        # Path length, not start-to-end displacement -- energy is spent along the path
        # travelled. Accumulated from speed the same way the training metric is.
        speed = np.linalg.norm(lin[:, :2], axis=1)
        path_len = float((speed * env.dt).sum())
        mass = np.asarray(env.robot.get_mass(), dtype=float).reshape(-1)[0]
        # E / (m g d), dimensionless. Undefined for a robot that never moved.
        weight_distance = float(mass) * 9.81 * path_len
        cot = energy / weight_distance if weight_distance > 0 else float("nan")

        self._plot_speed(t, lin, cmd, err_x)
        self._plot_power(t, p_mech, energy, cot)
        self._plot_torques(t, data["torque"], env.env_cfg["joint_names"])
        self._plot_path(pos, yaw0)
        self._plot_error(t, err_lin, "Linear velocity tracking error", "|v_xy - cmd| (m/s)", "lin_vel_error")
        self._plot_error(t, err_x, "X linear velocity error", "v_x - cmd_x (m/s)", "lin_vel_error_x", signed=True)
        self._plot_error(t, err_y, "Y linear velocity error", "v_y - cmd_y (m/s)", "lin_vel_error_y", signed=True)
        self._plot_error(t, err_ang, "Angular velocity error (yaw)", "w_z - cmd_wz (rad/s)", "ang_vel_error", signed=True)
        self._plot_error(t, cross_track, "Cross-track error", "offset from path (m)", "cross_track_error", signed=True)
        self._plot_error(t, heading_err, "Heading error", "yaw - yaw_0 (deg)", "heading_error", signed=True)

        foot = np.asarray(self._foot_force) if self._foot_force else None  # (steps, num_feet, 3)
        if foot is not None:
            self._plot_foot_forces(t, foot, self._foot_names)
            self._plot_total_foot_forces(t, foot)

        if self._phi_values:
            phi = np.asarray(self._phi_values)                    # (steps, latent_dim)
            phi_fresh = np.asarray(self._phi_fresh, dtype=float)   # (steps,)
            self._plot_phi(t, phi, phi_fresh)

        metrics = {
            "steps": int(len(t)),
            "duration_s": float(t[-1] + env.dt),
            "mean_speed_xy_m_s": float(np.linalg.norm(lin[:, :2], axis=1).mean()),
            "mean_vx_m_s": float(lin[:, 0].mean()),
            "mean_vy_m_s": float(lin[:, 1].mean()),
            "mean_height_m": float(data["height"].mean()),
            "rms_lin_vel_error_m_s": float(np.sqrt((err_lin**2).mean())),
            "rms_lin_vel_error_x_m_s": float(np.sqrt((err_x**2).mean())),
            "rms_lin_vel_error_y_m_s": float(np.sqrt((err_y**2).mean())),
            "rms_ang_vel_error_rad_s": float(np.sqrt((err_ang**2).mean())),
            "distance_travelled_m": float(np.linalg.norm(pos[-1] - pos[0])),
            "final_cross_track_m": float(cross_track[-1]),
            "max_abs_cross_track_m": float(np.abs(cross_track).max()),
            "rms_cross_track_m": float(np.sqrt((cross_track**2).mean())),
            "final_heading_error_deg": float(heading_err[-1]),
            "max_abs_heading_error_deg": float(np.abs(heading_err).max()),
            "peak_abs_torque_nm": float(np.abs(data["torque"]).max()),
            "mean_abs_torque_nm": float(np.abs(data["torque"]).mean()),
            "robot_mass_kg": float(mass),
            "path_length_m": path_len,
            "mean_mech_power_w": float(p_mech.mean()),
            "peak_mech_power_w": float(p_mech.max()),
            "mech_energy_j": energy,
            "mech_cost_of_transport": cot,
        }
        if foot is not None:
            metrics["peak_abs_foot_force_n"] = float(np.abs(foot).max())
            total_fz = foot[:, :, 2].sum(axis=1)  # (steps,) -- same series total_foot_forces.png plots
            metrics["mean_total_foot_fz_n"] = float(total_fz.mean())
            metrics["peak_total_fz_n"] = float(total_fz.max())
            for i, name in enumerate(self._foot_names):
                metrics[f"mean_fz_{name.replace(' ', '_')}_n"] = float(foot[:, i, 2].mean())
        if self._recording:
            # fps = control rate so the video plays at wall-clock speed
            fps = round(1.0 / env.dt)
            env.cam.stop_recording(save_to_filename=str(self.out_dir / "rollout.mp4"), fps=fps)
            metrics["video_fps"] = fps

        payload = {"experiment": self.experiment}
        if self._wall_sim_s is not None:
            wall_s, sim_s = self._wall_sim_s
            payload["wall_time_to_sim_time_ratio"] = wall_s / sim_s if sim_s > 0 else None
        payload.update(self.params)
        payload["metrics"] = metrics
        (self.out_dir / "params.json").write_text(json.dumps(payload, indent=2, default=str))
        return self.out_dir, metrics

    # ---- individual figures ----------------------------------------------------
    def _finish(self, fig, name):
        fig.tight_layout()
        fig.savefig(self.out_dir / f"{name}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    def _plot_speed(self, t, lin, cmd, err_x):
        """Forward speed against its command.

        v_y is deliberately absent: the command holds it at zero, so it is tracking
        error rather than speed and lives in the lin_vel_error_y figure. err_x is the
        same series the X error figure plots, so the RMS quoted here matches it.
        """
        fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        # Command is a reference, not a peer series: muted and dashed, so the measured
        # series keeps the categorical slot.
        ax.plot(t, cmd[:, 0], color=INK_MUTED, linewidth=1.2, linestyle="--", label="commanded v_x")
        ax.plot(t, lin[:, 0], color=SERIES[0], linewidth=1.6, label="measured v_x")
        _style_axes(ax, "Base linear velocity", "time (s)", "velocity (m/s)")
        # Headroom so the legend never sits on the trace.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.22 * (hi - lo))
        # Sits on the right end of the command line, labelling it the way the error
        # figures label their zero line.
        rms = float(np.sqrt((err_x**2).mean()))
        ax.annotate(
            f"RMS {rms:.3f}",
            xy=(t[-1], cmd[-1, 0]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=INK,
        )
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=2, loc="upper right")
        self._finish(fig, "speed")

    def _plot_power(self, t, p_mech, energy, cot):
        """Mechanical joint power over the rollout, with the run totals annotated.

        The dashed line is mean power -- the level whose area under it equals the total
        energy, so the annotated joules and the trace read as the same quantity.
        """
        fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        mean_p = float(p_mech.mean())
        ax.axhline(mean_p, color=INK_MUTED, linewidth=1.2, linestyle="--", label="mean power")
        ax.plot(t, p_mech, color=SERIES[0], linewidth=1.4, label="sum |tau * omega|")
        ax.set_ylim(0.0, float(p_mech.max()) * 1.30 or 1.0)  # power is non-negative; headroom for the totals
        _style_axes(ax, "Mechanical joint power", "time (s)", "power (W)")
        ax.annotate(
            f"total {energy:.1f} J    CoT {cot:.3f}    mean {mean_p:.1f} W",
            xy=(0.995, 0.94),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            color=INK,
        )
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=2, loc="upper left")
        self._finish(fig, "mech_power")

    def _plot_torques(self, t, torque, joint_names):
        """Small multiples: one panel per leg, three joints each, shared y-scale."""
        legs = [(joint_names[i][:2], slice(i, i + 3)) for i in range(0, 12, 3)]
        lim = float(np.abs(torque).max()) * 1.08 or 1.0
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), facecolor=SURFACE, sharex=True, sharey=True)
        flat = axes.ravel()
        for idx, (ax, (leg, sl)) in enumerate(zip(flat, legs)):
            ax.set_facecolor(SURFACE)
            for k, joint in enumerate(("hip", "thigh", "calf")):
                ax.plot(t, torque[:, sl][:, k], color=SERIES[k], linewidth=1.4, label=joint)
            ax.set_ylim(-lim, lim)
            # Axes are shared, so label only the outer edges.
            _style_axes(
                ax,
                f"{leg} leg",
                "time (s)" if idx >= 2 else "",
                "torque (N m)" if idx % 2 == 0 else "",
            )
        # One figure-level legend above the panels: identity is shared across all four,
        # and this keeps it off the data.
        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=9,
            labelcolor=INK_MUTED,
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(0.995, 1.0),
        )
        fig.suptitle("Joint torque by leg", color=INK, fontsize=12, x=0.008, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.out_dir / "torques_by_leg.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    def _plot_foot_forces(self, t, ff, names):
        """Small multiples: one panel per foot, world-frame Fx/Fy/Fz, shared y-scale.

        ff: (steps, num_feet, 3) -- the MPM mud reaction on each foot (see
        Go2MPMEnv.get_foot_forces). One panel per foot answers "each foot as a separate
        plot"; the three series answer "break down the xyz forces".
        """
        n = ff.shape[1]
        ncol = 2
        nrow = int(np.ceil(n / ncol))
        lim = float(np.abs(ff).max()) * 1.08 or 1.0
        comps = ("F_x", "F_y", "F_z")
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(11, 3.2 * nrow), facecolor=SURFACE, sharex=True, sharey=True
        )
        flat = np.atleast_1d(axes).ravel()
        for idx, ax in enumerate(flat):
            ax.set_facecolor(SURFACE)
            if idx >= n:
                ax.set_visible(False)  # odd foot count: hide the empty panel
                continue
            # Zero line: Fx/Fy swing about it, Fz (support) rides above it.
            ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
            for k in range(3):
                ax.plot(t, ff[:, idx, k], color=SERIES[k], linewidth=1.4, label=comps[k])
            ax.set_ylim(-lim, lim)
            # Shared axes: label only the outer edges.
            _style_axes(
                ax,
                names[idx],
                "time (s)" if idx >= n - ncol else "",
                "force (N)" if idx % ncol == 0 else "",
            )
        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=9,
            labelcolor=INK_MUTED,
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(0.995, 1.0),
        )
        fig.suptitle("Foot reaction force (world frame)", color=INK, fontsize=12, x=0.008, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.out_dir / "foot_forces.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    def _plot_total_foot_forces(self, t, ff):
        """Whole-robot reaction force: all four feet summed, one line per world-frame axis.

        ff: (steps, num_feet, 3), same data as foot_forces.png but collapsed across feet --
        this answers "what's the net push on the robot" rather than "what's each foot doing".
        Summed F_z should hover near the robot's weight (mg), since that's what holds it up;
        see mean_total_foot_fz_n in metrics for the same quantity as a scalar.
        """
        total = ff.sum(axis=1)  # (steps, 3)
        comps = ("F_x", "F_y", "F_z")
        fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
        for k in range(3):
            ax.plot(t, total[:, k], color=SERIES[k], linewidth=1.6, label=comps[k])
        _style_axes(ax, "Total foot reaction force (world frame, summed across feet)", "time (s)", "force (N)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3, loc="upper right")
        self._finish(fig, "total_foot_forces")

    def _plot_phi(self, t, phi, fresh):
        """Small multiples: phi's live estimate of each privileged latent dimension,
        one line per panel, plus a usage panel showing which steps were freshly
        recomputed vs held over (see stabilise_adaptation in eval_cfg.py, --phi-rate).

        phi: (steps, latent_dim). Dimensions mix unrelated quantities (mass, friction,
        mud coefficients, ...) on unrelated scales, so panels do NOT share a y-axis --
        unlike torques_by_leg / foot_forces, which are homogeneous units. Dimensions are
        labelled by index only: their composition (which privileged term is where) is
        training-run-specific and not reliably recoverable from the checkpoint.
        """
        n_dims = phi.shape[1]
        n = n_dims + 1  # + the usage panel
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.4 * nrow), facecolor=SURFACE, sharex=True)
        flat = np.atleast_1d(axes).ravel()

        panels = [("phi usage", fresh, True)] + [(f"e_t[{i}]", phi[:, i], False) for i in range(n_dims)]
        for idx, ax in enumerate(flat):
            ax.set_facecolor(SURFACE)
            if idx >= n:
                ax.set_visible(False)
                continue
            title, series, is_usage = panels[idx]
            if is_usage:
                ax.plot(t, series, color=SERIES[0], linewidth=1.4, drawstyle="steps-post")
                ax.set_ylim(-0.15, 1.15)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["held", "fresh"])
            else:
                ax.plot(t, series, color=SERIES[0], linewidth=1.4)
            _style_axes(ax, title, "time (s)" if idx >= n - ncol else "", "")

        fig.suptitle("Adaptation estimate (phi)", color=INK, fontsize=12, x=0.008, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.out_dir / "phi_estimate.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    def _plot_path(self, pos, yaw0):
        """Top-down trace against the commanded ray, on equal axes.

        Equal aspect on purpose: it shows the path's true shape rather than magnifying a few
        centimetres of drift into a dramatic curve. The cross_track_error figure is the zoomed
        view of the same quantity.
        """
        fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        rel = pos - pos[0]
        # reference ray, drawn to the along-track extent actually reached
        direction = np.array([np.cos(yaw0), np.sin(yaw0)])
        reach = float((rel @ direction).max())
        ax.plot([0.0, reach * direction[0]], [0.0, reach * direction[1]],
                color=INK_MUTED, linewidth=1.2, linestyle="--", label="commanded path")
        ax.plot(rel[:, 0], rel[:, 1], color=SERIES[0], linewidth=1.6, label="actual path")
        ax.plot(rel[0, 0], rel[0, 1], marker="o", markersize=5, color=INK, linestyle="none", label="start")
        ax.set_aspect("equal", adjustable="datalim")
        _style_axes(ax, "Ground track", "along start heading (m)", "lateral (m)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="best")
        self._finish(fig, "ground_track")

    def _plot_error(self, t, err, title, ylabel, name, signed=False):
        fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        if signed:
            # Zero line is the target; make it readable without competing with the data.
            ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
        # Single series: no legend box, the title names it.
        ax.plot(t, err, color=SERIES[0], linewidth=1.6)
        rms = float(np.sqrt((err**2).mean()))
        ax.annotate(
            f"RMS {rms:.3f}",
            xy=(0.995, 0.94),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            color=INK,
        )
        _style_axes(ax, title, "time (s)", ylabel)
        self._finish(fig, name)


def run_rollout(env, policy, steps=None, reporter=None):
    """Step the env. Runs forever when `steps` is None, matching the viewer scripts."""
    obs_dict = env.reset()
    wall_start = time.perf_counter()
    with torch.no_grad():
        i = 0
        while steps is None or i < steps:
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)
            if reporter is not None:
                reporter.record(env, policy)
            i += 1
    wall_s = time.perf_counter() - wall_start
    sim_s = i * env.dt
    if sim_s > 0:
        print(f"rollout: {sim_s:.1f}s sim in {wall_s:.1f}s wall -> {wall_s / sim_s:.2f}s wall per sim-second ({sim_s / wall_s:.2f}x real time)")
        if reporter is not None:
            reporter.note_rollout_timing(wall_s, sim_s)
    return obs_dict
