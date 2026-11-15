"""The rollout loop, and the Reporter that turns a rollout into plots, metrics and a video.

Reports are written to reports/<version>/<NNN>_<version>/, where <version> is the policy version
(the checkpoint's run directory, e.g. v6) and NNN a report number unique across reports/. Each finished report
appends a row describing it to reports/index.csv.
"""

import csv
import inspect
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from genesis.utils.geom import quat_to_xyz

from scripts.unitree_policy import plots


def _claim_report_dir(root, experiment, run=None):
    """Create reports/<experiment>/<NNN>_<run>/ with NNN one past the highest number anywhere under reports/."""
    numbers = (d.name.split("_")[0] for d in root.glob("*/*") if d.is_dir())
    number = max((int(n) for n in numbers if n.isdigit()), default=0) + 1
    while True:
        out_dir = root / experiment / (f"{number:03d}_{run}" if run else f"{number:03d}")
        try:
            out_dir.mkdir(parents=True)
            return out_dir
        except FileExistsError:  # another run claimed it first
            number += 1


STEADY_AFTER_S = 1.0  # steady_vx_m_s ignores the first second of the rollout


class Reporter:
    """Collects per-step state from env 0 and writes plots + params at the end."""

    def __init__(self, experiment, params, out_root=None, reports_dir="reports"):
        self.experiment = experiment
        self.params = params
        root = Path(out_root) if out_root else Path(reports_dir)
        self.index_path = root / "index.csv"
        self.out_dir = _claim_report_dir(root, experiment, params.get("run"))
        self.params = {"report": self.out_dir.name.split("_")[0], "date": datetime.now().strftime("%Y-%m-%d %H:%M"), **params}
        self._rows = {
            k: []
            for k in ("lin_vel", "ang_vel", "command", "torque", "dof_vel", "height", "pos", "yaw")
        }
        # only recorded when the env has get_foot_forces (MPM)
        self._foot_force = []
        self._foot_names = None
        # only recorded for actors with an adaptation module (see eval_cfg.stabilise_adaptation)
        self._phi_values = []
        self._phi_fresh = []
        self._recording = False
        self._streaming_video = False
        self._wall_sim_s = None  # (wall_s, sim_s)

    def record(self, env, policy=None):
        """Record env 0's state for this step, and a video frame if the env has a camera."""
        self._rows["lin_vel"].append(env.base_lin_vel[0].detach().cpu().numpy().copy())
        self._rows["ang_vel"].append(env.base_ang_vel[0].detach().cpu().numpy().copy())
        self._rows["command"].append(env.commands[0].detach().cpu().numpy().copy())
        torque = env.robot.get_dofs_control_force(env.motors_dof_idx)
        self._rows["torque"].append(torque[0].detach().cpu().numpy().copy())
        self._rows["dof_vel"].append(env.dof_vel[0].detach().cpu().numpy().copy())
        self._rows["height"].append(float(env.base_pos[0, 2]))
        self._rows["pos"].append(env.base_pos[0, :2].detach().cpu().numpy().copy())
        self._rows["yaw"].append(float(quat_to_xyz(env.base_quat, rpy=True, degrees=False)[0, 2]))

        if hasattr(env, "get_foot_forces"):
            ff = np.asarray(env.get_foot_forces())  # (num_envs, num_feet, 3)
            self._foot_force.append(ff[0].copy())   # -> (num_feet, 3)
            if self._foot_names is None:
                self._foot_names = list(
                    getattr(env, "foot_names", None) or [f"foot {i}" for i in range(ff.shape[1])]
                )

        if policy is not None and getattr(policy, "last_e_t", None) is not None:
            self._phi_values.append(policy.last_e_t[0].detach().cpu().numpy().copy())
            self._phi_fresh.append(bool(policy.last_e_t_fresh))

        if getattr(env, "cam", None) is not None:
            if not self._recording:
                # Newer Genesis takes the filename here and captures frames itself; older
                # versions take it in stop_recording and need a render() per frame.
                self._streaming_video = "save_to_filename" in inspect.signature(env.cam.start_recording).parameters
                if self._streaming_video:
                    env.cam.start_recording(save_to_filename=str(self.out_dir / "rollout.mp4"), fps=round(1.0 / env.dt))
                else:
                    env.cam.start_recording()
                self._recording = True
            env.update_camera()
            if not self._streaming_video:
                env.cam.render()

    def note_rollout_timing(self, wall_s, sim_s):
        """Store wall-clock vs simulated time; finalize() writes the ratio to params.json."""
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

        # Straightness, relative to the line from the first recorded pose along its heading.
        pos, yaw = data["pos"], data["yaw"]
        yaw0 = yaw[0]
        normal = np.array([-np.sin(yaw0), np.cos(yaw0)])
        cross_track = (pos - pos[0]) @ normal
        heading_err = np.degrees(np.arctan2(np.sin(yaw - yaw0), np.cos(yaw - yaw0)))

        # Mechanical joint power sum|tau * omega| (no motor heating), as in the training-side metrics.
        p_mech = np.abs(data["torque"] * data["dof_vel"]).sum(axis=1)
        energy = float((p_mech * env.dt).sum())
        # path length, not displacement
        speed = np.linalg.norm(lin[:, :2], axis=1)
        path_len = float((speed * env.dt).sum())
        mass = env.robot.get_mass()
        if hasattr(mass, "detach"):  # torch tensor, possibly on the GPU
            mass = mass.detach().cpu().numpy()
        mass = np.asarray(mass, dtype=float).reshape(-1)[0]
        # cost of transport E / (m g d)
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
            # after the spin-up from standing; the whole run if it is shorter than that
            "steady_vx_m_s": float(lin[t >= STEADY_AFTER_S, 0].mean() if (t >= STEADY_AFTER_S).any() else lin[:, 0].mean()),
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
            total_fz = foot[:, :, 2].sum(axis=1)  # (steps,)
            metrics["mean_total_foot_fz_n"] = float(total_fz.mean())
            metrics["peak_total_fz_n"] = float(total_fz.max())
            for i, name in enumerate(self._foot_names):
                metrics[f"mean_fz_{name.replace(' ', '_')}_n"] = float(foot[:, i, 2].mean())
        if self._recording:
            # control rate, so the video plays in real time
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
        append_index(self.index_path, self.out_dir, payload)
        return self.out_dir, metrics


INDEX_METRICS = (
    "steady_vx_m_s",
    "mean_vx_m_s",
    "mean_mech_power_w",
    "mech_energy_j",
    "mech_cost_of_transport",
    "peak_abs_torque_nm",
    "peak_abs_foot_force_n",
    "max_abs_cross_track_m",
)
INDEX_FIELDS = ("report", "date", "policy", "description", "settings", "commanded_vx_m_s", "reached_pit_end", *INDEX_METRICS, "folder")


def append_index(index_path, out_dir, payload):
    """Append one row summarising a report to reports/index.csv, writing the header on first use."""
    metrics = payload.get("metrics", {})
    vx = payload.get("command_cfg", {}).get("lin_vel_x_range", [None])[0]
    row = {
        "report": payload.get("report", out_dir.name.split("_")[0]),
        "date": payload.get("date", ""),
        "policy": payload.get("run", ""),
        "description": payload.get("description", ""),
        "settings": payload.get("settings", ""),
        "commanded_vx_m_s": vx,
        "reached_pit_end": payload.get("reached_pit_end", ""),
        **{k: (round(metrics[k], 3) if isinstance(metrics.get(k), float) else metrics.get(k, "")) for k in INDEX_METRICS},
        "folder": out_dir.relative_to(index_path.parent).as_posix(),
    }
    new = not index_path.exists()
    with index_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def run_rollout(env, policy, steps=None, reporter=None, stop_when=None):
    """Step the env until `steps` steps have run or `stop_when(env)` is True (forever if both are None)."""
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
