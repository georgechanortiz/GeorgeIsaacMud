"""Finite-horizon rollouts with plots and a parameter dump.

Enabled by --reporting on any eval entry point. Writes to

    reports/<experiment>/<YYYY-MM-DD_HH-MM-SS>/

where <experiment> is the checkpoint's parent-of-parent directory, mirroring how
logs/ is organised (go2_mud_flat, unitree_go2_flat_rma, ...), so reports for
different policies never mix and repeated runs never collide.
"""

import inspect
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from genesis.utils.geom import quat_to_xyz

from scripts.unitree_policy import plots


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
        self._streaming_video = False
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
                # Genesis changed its recording API: newer versions take the filename and
                # framerate here and capture frames by themselves as the scene steps; older
                # ones take them in stop_recording and need an explicit render() per frame.
                self._streaming_video = "save_to_filename" in inspect.signature(env.cam.start_recording).parameters
                if self._streaming_video:
                    # fps = control rate so the video plays at wall-clock speed
                    env.cam.start_recording(save_to_filename=str(self.out_dir / "rollout.mp4"), fps=round(1.0 / env.dt))
                else:
                    env.cam.start_recording()
                self._recording = True
            env.update_camera()  # follow the robot
            if not self._streaming_video:
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
        mass = env.robot.get_mass()
        if hasattr(mass, "detach"):  # torch tensor, possibly on the GPU
            mass = mass.detach().cpu().numpy()
        mass = np.asarray(mass, dtype=float).reshape(-1)[0]
        # E / (m g d), dimensionless. Undefined for a robot that never moved.
        weight_distance = float(mass) * 9.81 * path_len
        cot = energy / weight_distance if weight_distance > 0 else float("nan")

        plots.plot_speed(self.out_dir, t, lin, cmd, err_x)
        plots.plot_power(self.out_dir, t, p_mech, energy, cot)
        plots.plot_torques(self.out_dir, t, data["torque"], env.env_cfg["joint_names"])
        plots.plot_path(self.out_dir, pos, yaw0)
        plots.plot_error(self.out_dir, t, err_lin, "Linear velocity tracking error", "|v_xy - cmd| (m/s)", "lin_vel_error")
        plots.plot_error(self.out_dir, t, err_x, "X linear velocity error", "v_x - cmd_x (m/s)", "lin_vel_error_x", signed=True)
        plots.plot_error(self.out_dir, t, err_y, "Y linear velocity error", "v_y - cmd_y (m/s)", "lin_vel_error_y", signed=True)
        plots.plot_error(self.out_dir, t, err_ang, "Angular velocity error (yaw)", "w_z - cmd_wz (rad/s)", "ang_vel_error", signed=True)
        plots.plot_error(self.out_dir, t, cross_track, "Cross-track error", "offset from path (m)", "cross_track_error", signed=True)
        plots.plot_error(self.out_dir, t, heading_err, "Heading error", "yaw - yaw_0 (deg)", "heading_error", signed=True)

        foot = np.asarray(self._foot_force) if self._foot_force else None  # (steps, num_feet, 3)
        if foot is not None:
            plots.plot_foot_forces(self.out_dir, t, foot, self._foot_names)
            plots.plot_total_foot_forces(self.out_dir, t, foot)

        if self._phi_values:
            phi = np.asarray(self._phi_values)                    # (steps, latent_dim)
            phi_fresh = np.asarray(self._phi_fresh, dtype=float)   # (steps,)
            plots.plot_phi(self.out_dir, t, phi, phi_fresh)

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
            if self._streaming_video:
                env.cam.stop_recording()
            else:
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


def run_rollout(env, policy, steps=None, reporter=None, stop_when=None):
    """Step the env until `steps` control steps have run or `stop_when(env)` returns True,
    whichever comes first. Runs forever when both are None, matching the viewer scripts.

    `stop_when` is checked after each step is recorded, so the step that meets it is in the report.
    """
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
            if stop_when is not None and stop_when(env):
                print(f"rollout: stop condition met after {i} steps")
                break
    wall_s = time.perf_counter() - wall_start
    sim_s = i * env.dt
    if sim_s > 0:
        print(f"rollout: {sim_s:.1f}s sim in {wall_s:.1f}s wall -> {wall_s / sim_s:.2f}s wall per sim-second ({sim_s / wall_s:.2f}x real time)")
        if reporter is not None:
            reporter.note_rollout_timing(wall_s, sim_s)
    return obs_dict
