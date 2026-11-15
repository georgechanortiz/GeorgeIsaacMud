"""Interactive mud tuner: watch an A2 policy walk through the MPM pit and change the mud while it runs.

    python scripts/mud_tuner_a2.py                       # v6 in the calibrated 4 in mud
    python scripts/mud_tuner_a2.py -p Policies/v2_erft/model_4999.pt --mud-depth 0.1375

Opens the Genesis viewer and a control panel (scripts/mud_tuner_ui.py):
- live sliders: stiffness E, Poisson ratio, density, foot-mud friction, commanded speed. E also sets
  the liquid's viscosity (mu * substep dt). Changes apply on slider release and persist across
  restarts.
- Restart puts the robot and every mud particle back to the start; Pause freezes the simulation.
- depth, substeps and grid density need a new scene, so Rebuild relaunches this script with them
  (and the current slider values).

The mud is the calibrated viscous liquid (envs/live_mud.py). MPM is slow: at the default 60
substeps expect ~15 s of computer time per simulated second, so the viewer plays in slow motion;
lower the grid density or substeps for a faster, coarser preview.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import genesis as gs

from envs import live_mud, mud
from envs.mpm_env import A2MPMEnv
from scripts.unitree_policy import cli
from scripts.unitree_policy.eval_cfg import load_policy

DEFAULT_POLICY = "Policies/v6_erft/model_4999.pt"
CALIBRATED = dict(E=1e5, nu=0.35, rho=1500.0, substeps=60)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_policy_arg(parser, DEFAULT_POLICY)
    parser.add_argument("--vx", type=float, default=0.35, help="initial commanded forward speed (m/s)")
    parser.add_argument("--mud-depth", type=float, default=0.1016, metavar="M")
    parser.add_argument("--substeps", type=int, default=CALIBRATED["substeps"])
    parser.add_argument("--grid_density", type=int, default=48)
    parser.add_argument("--mud-E", type=float, default=CALIBRATED["E"], metavar="PA")
    parser.add_argument("--mud-nu", type=float, default=CALIBRATED["nu"], metavar="NU")
    parser.add_argument("--mud-rho", type=float, default=CALIBRATED["rho"], metavar="KGM3")
    parser.add_argument("--coup-friction", type=float, default=0.4)
    parser.add_argument("--no-ui", action="store_true", help="run without the control panel (for testing)")
    parser.add_argument("--headless", action="store_true", help="no viewer window (for testing)")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (for testing)")
    return parser.parse_args()


class Panel:
    """The control panel subprocess: commands arrive on a queue, status goes out on its stdin."""

    def __init__(self, init):
        self.proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts" / "mud_tuner_ui.py"), json.dumps(init)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self.commands = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.commands.put(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.commands.put({"cmd": "quit"})  # window closed

    def status(self, text):
        try:
            self.proc.stdin.write(json.dumps({"text": text}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


def main():
    args = parse_args()
    if args.substeps % 4:
        sys.exit("--substeps must be a multiple of the A2's decimation (4)")

    ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cli.load_run(args.policy)
    env_cfg["substeps"] = args.substeps
    cli.drive_straight(command_cfg, args.vx)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    mud_material = {"material": "liquid", "E": args.mud_E, "nu": args.mud_nu, "rho": args.mud_rho, "viscous": True,
                    "yield_stress": 80.0, "friction_angle": 15.0}
    mpm_cfg = mud.a2_mud_pit(mud_material, args.mud_depth, grid_density=args.grid_density,
                             coup_friction=args.coup_friction)
    mud.spawn_on_pit(env_cfg, mpm_cfg, offset=0.6)
    end_x = mud.pit_end_x(mpm_cfg, 0.55)

    live = live_mud.LiveLiquid(args.mud_E, args.mud_nu, args.mud_rho)
    mud.mpm_material = lambda patch: live.material  # the pit's single mud layer uses the live material

    env = A2MPMEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
                   show_viewer=not args.headless, mpm_cfg=mpm_cfg)
    policy = load_policy(env, ckpt_path, train_cfg)
    mass = float(np.asarray(env.robot.get_mass()).reshape(-1)[0])
    substep_dt = env.dt / args.substeps
    dx = 1.0 / args.grid_density

    init = dict(E=args.mud_E, nu=args.mud_nu, rho=args.mud_rho, friction=args.coup_friction, vx=args.vx,
                depth=args.mud_depth, substeps=args.substeps, grid_density=args.grid_density,
                calibrated_visc=CALIBRATED["E"] / (2 * (1 + CALIBRATED["nu"])) * env.dt / CALIBRATED["substeps"])
    panel = None if args.no_ui else Panel(init)

    state = dict(paused=False, auto_restart=True, follow=not args.headless, vx=args.vx)
    window = deque(maxlen=50)  # last second: (vx, power, total foot Fz)
    run = dict(energy=0.0, dist=0.0, t=0.0)
    last_note = ""

    def set_vx(vx):
        state["vx"] = vx
        env.commands_limits[0][0] = vx
        env.commands_limits[1][0] = vx
        env.commands[:, 0] = vx

    def restart():
        nonlocal obs
        env.scene.reset()
        obs = env.reset()
        set_vx(state["vx"])
        window.clear()
        run.update(energy=0.0, dist=0.0, t=0.0)

    def handle(msg):
        nonlocal last_note
        cmd = msg.get("cmd")
        if cmd == "set":
            name, value = msg["name"], float(msg["value"])
            if name in ("E", "nu", "rho"):
                live.set(env.scene, **{name: value})
            elif name == "friction":
                live_mud.set_coupling_friction(env.robot, value)
            elif name == "vx":
                set_vx(value)
            last_note = f"set {name} = {value:.4g}"
        elif cmd == "restart":
            restart()
            last_note = "restarted"
        elif cmd == "pause":
            state["paused"] = bool(msg["value"])
        elif cmd == "auto_restart":
            state["auto_restart"] = bool(msg["value"])
        elif cmd == "follow":
            state["follow"] = bool(msg["value"])
        elif cmd == "rebuild":
            argv = [sys.executable, str(Path(__file__).resolve()), "-p", str(ckpt_path),
                    "--mud-depth", str(msg["depth"]), "--substeps", str(msg["substeps"]),
                    "--grid_density", str(msg["grid_density"]), "--mud-E", str(msg["E"]), "--mud-nu", str(msg["nu"]),
                    "--mud-rho", str(msg["rho"]), "--coup-friction", str(msg["friction"]), "--vx", str(msg["vx"])]
            if panel:
                panel.close()
            os.execv(sys.executable, argv)
        elif cmd == "quit":
            return False
        return True

    obs = env.reset()
    set_vx(args.vx)
    step = 0
    wall0, sim0 = time.perf_counter(), 0.0
    with torch.no_grad():
        while args.max_steps is None or step < args.max_steps:
            if panel:
                try:
                    while True:
                        if not handle(panel.commands.get_nowait()):
                            panel.close()
                            return
                except queue.Empty:
                    pass
            if state["paused"]:
                time.sleep(0.05)
                continue

            obs, _, dones, _ = env.step(policy(obs))
            step += 1
            torque = env.robot.get_dofs_control_force(env.motors_dof_idx)[0].cpu().numpy()
            power = float(np.abs(torque * env.dof_vel[0].cpu().numpy()).sum())
            vx = float(env.base_lin_vel[0, 0])
            fz = float(env.get_foot_forces()[0, :, 2].sum())
            window.append((vx, power, fz))
            run["energy"] += power * env.dt
            run["dist"] += abs(vx) * env.dt
            run["t"] += env.dt
            sim0 += env.dt

            x = float(env.base_pos[0, 0])
            unstable = not np.isfinite(x) or bool(dones[0])
            if unstable:
                last_note = "robot fell or the sim went unstable -- restarted"
                restart()
            elif x >= end_x and state["auto_restart"]:
                last_note = f"reached the pit end in {run['t']:.1f} s -- restarted"
                restart()

            if state["follow"] and env.scene.viewer is not None:
                bx, by = float(env.base_pos[0, 0]), float(env.base_pos[0, 1])
                env.scene.viewer.set_camera_pose(pos=np.array([bx + 1.8, by - 2.2, 1.3]), lookat=np.array([bx, by, 0.3]))

            if step % 10 == 0:
                w = np.array(window)
                cot = run["energy"] / (mass * 9.81 * run["dist"]) if run["dist"] > 0.05 else float("nan")
                text = (
                    f"t = {run['t']:5.1f} s    x = {x:5.2f} / {end_x:.2f} m\n"
                    f"speed (1 s) = {w[:, 0].mean():.3f} m/s   (command {state['vx']:.2f})\n"
                    f"joint power (1 s) = {w[:, 1].mean():6.1f} W\n"
                    f"CoT since restart = {cot:.2f}\n"
                    f"mud vertical force on feet (1 s) = {w[:, 2].mean():6.1f} N\n"
                    f"mud now: E={live.E:.3g} Pa  nu={live.nu:.2f}  rho={live.rho:.0f}\n"
                    f"sim speed: {(time.perf_counter() - wall0) / max(sim0, 1e-9):.1f} s wall per sim s\n"
                    f"{last_note}"
                )
                if panel:
                    panel.status(text)
                else:
                    print(text.replace("\n", " | "))
    if panel:
        panel.close()


if __name__ == "__main__":
    main()
