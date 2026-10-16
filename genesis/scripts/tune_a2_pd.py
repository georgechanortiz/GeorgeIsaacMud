"""Check, and if needed tune, the A2's Genesis PD gains against the IsaacLab actuator it trained with.

The a2_flat policies were trained with isaaclab's DCMotor: an *explicit* PD evaluated at every
200 Hz physics step,

    tau = clip(kp * (q* - q) - kd * qd,  DCMotor torque-speed envelope),

applied to joints with 0.03 kg*m^2 armature and no joint friction or damping.

Genesis replays them through a different path (envs/go2_env.py): the PD is Genesis' built-in
position controller running inside the rigid solver's substep loop, and the torque-speed envelope
can only be refreshed once per 50 Hz control step. This script drives the same A2 through both
paths with identical 50 Hz zero-order-hold targets and measures how far the joint trajectories
drift apart:

  reference   dt = 0.005, one physics step per substep, DCMotor torque computed here in Python
              from the instantaneous state and applied with control_dofs_force -- i.e. the
              IsaacLab actuator model, reproduced exactly.
  candidate   what Go2Env does with configs/a2_flat.py: Genesis' built-in PD
              (control_dofs_position) with gains (s_kp * kp, s_kd * kd), armature reduced by the
              implicit-integrator term kd * dt, and the envelope refreshed before each of the 4
              physics steps per control step. Variants switch each of those off.

Each candidate env gets a different (s_kp, s_kd), so one batched rollout sweeps the whole grid.
The gains to put in configs/a2_flat.py are the scale factors that minimise the tracking error,
per actuator group.

Two scenarios:
  swing   base fixed in the air, a trot-like 2 Hz target pattern plus step changes. Isolates the
          actuator response: no contacts, so the two paths cannot diverge chaotically.
  stand   free base on the ground, a slow squat. Gravity-loaded actuation, the regime that sets
          stance height and body sag.

Usage:
    python scripts/tune_a2_pd.py            # CPU
    python scripts/tune_a2_pd.py --gpu
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs  # noqa: E402

from configs.a2_flat import ACTUATOR_GROUPS, get_cfgs  # noqa: E402
from envs.go2_env import dc_motor_torque_limits, resolve_urdf  # noqa: E402

CONTROL_DT = 0.02
DECIMATION = 4
GROUPS = ("hip", "thigh", "calf")


def build(n_envs, dt, substeps, fixed, base_z, armature, batch):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False, tolerance=1e-5, max_collision_pairs=20, batch_dofs_info=batch
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(file=resolve_urdf(get_cfgs()[0]["urdf"]), pos=(0, 0, base_z), fixed=fixed, default_armature=armature)
    )
    scene.build(n_envs=n_envs)
    return scene, robot


class Actuators:
    """Per-dof actuator constants, in Genesis' own dof order (dofs 6..17)."""

    def __init__(self, robot):
        self.names = [j.name for j in robot.joints[1:]]
        self.dofs = [robot.get_joint(n).dof_idx_local for n in self.names]
        self.group = [n.split("_")[1] for n in self.names]
        col = lambda key: torch.tensor([ACTUATOR_GROUPS[g][key] for g in self.group], dtype=gs.tc_float, device=gs.device)
        self.kp, self.kd = col("kp"), col("kd")
        self.effort, self.sat, self.vlim = col("effort_limit"), col("saturation_effort"), col("velocity_limit")
        default = get_cfgs()[0]["default_joint_angles"]
        self.q0 = torch.tensor([default[n] for n in self.names], dtype=gs.tc_float, device=gs.device)

    def envelope(self, qd):
        """isaaclab DCMotor._clip_effort bounds for joint velocity qd."""
        return dc_motor_torque_limits(qd, self.sat, self.effort, self.vlim)


def targets(act, scenario, n_steps):
    """(n_steps, 12) joint targets at 50 Hz, Genesis dof order."""
    t = torch.arange(n_steps, dtype=gs.tc_float, device=gs.device) * CONTROL_DT
    q = act.q0.repeat(n_steps, 1)
    amp = {
        "swing": {"hip": 0.15, "thigh": 0.45, "calf": 0.6},
        "kick": {"hip": 0.3, "thigh": 0.8, "calf": 0.8},
        "stand": {"hip": 0.0, "thigh": 0.25, "calf": 0.45},
    }[scenario]
    for i, name in enumerate(act.names):
        leg, g = name[:2], act.group[i]
        if scenario == "swing":
            # trot: diagonal pairs FL+RR and FR+RL in antiphase; calf leads thigh by 90 deg
            phase = 0.0 if leg in ("FL", "RR") else math.pi
            phase += {"hip": 0.0, "thigh": 0.0, "calf": math.pi / 2}[g]
            sign = -1.0 if g == "calf" else 1.0
            q[:, i] += sign * amp[g] * torch.sin(2 * math.pi * 2.0 * t + phase)
            # step changes every 0.5 s exercise the saturated, high-velocity part of the envelope
            step = ((t / 0.5).floor() % 2) * 2 - 1
            q[:, i] += sign * 0.5 * amp[g] * step * (t > 2.0)
        elif scenario == "kick":
            # large square-wave targets every 0.3 s, staggered per leg: drives the joints into
            # the saturated, high-speed corner of the DCMotor torque-speed envelope
            leg_shift = {"FL": 0.0, "FR": 0.075, "RL": 0.15, "RR": 0.225}[leg]
            sq = (((t + leg_shift) / 0.3).floor() % 2) * 2 - 1
            sign = -1.0 if g == "calf" else 1.0
            q[:, i] += sign * amp[g] * 0.5 * (sq + 1) * (1.0 if g != "hip" else 0.5)
        else:
            # slow squat: thigh and calf fold together so the feet stay roughly under the hips
            s = 0.5 * (1 - torch.cos(2 * math.pi * 0.5 * t))
            q[:, i] += {"hip": 0.0, "thigh": amp["thigh"], "calf": -amp["calf"]}[g] * s
    return q


def rollout_reference(scenario, n_steps, armature):
    fixed, z = (False, 0.45) if scenario == "stand" else (True, 0.9)
    scene, robot = build(1, CONTROL_DT / DECIMATION, 1, fixed, z, armature, batch=False)
    act = Actuators(robot)
    robot.set_dofs_position(act.q0[None], act.dofs, zero_velocity=True)
    q_tgt = targets(act, scenario, n_steps)
    robot.set_dofs_force_range(-act.effort * 10, act.effort * 10, act.dofs)  # envelope applied here instead
    qs, taus = [], []
    for k in range(n_steps):
        for _ in range(DECIMATION):
            q, qd = robot.get_dofs_position(act.dofs), robot.get_dofs_velocity(act.dofs)
            tau = act.kp * (q_tgt[k] - q) - act.kd * qd
            lo, hi = act.envelope(qd)
            tau = torch.clip(tau, lo, hi)
            robot.control_dofs_force(tau, act.dofs)
            scene.step()
        qs.append(robot.get_dofs_position(act.dofs)[0])
        taus.append(tau[0])
    base_z = robot.get_pos()[0, 2].item()
    return torch.stack(qs), torch.stack(taus), q_tgt, base_z, act.group


def rollout_candidate(scenario, n_steps, armature, s_kp, s_kd, dc_cap=True, compensate=True, decimation=DECIMATION):
    """Batched over the (s_kp, s_kd) grid: one env per pair.

    compensate mirrors env_cfg["compensate_implicit_pd"]: armature is reduced by the kd * h the
    implicit integrator adds back (see Go2Env._effective_armature). decimation mirrors
    env_cfg["decimation"]: the torque-speed envelope is refreshed before each of that many physics
    steps per control step. Either way there are DECIMATION solver substeps per control step.
    """
    fixed, z = (False, 0.45) if scenario == "stand" else (True, 0.9)
    n = len(s_kp)
    scene, robot = build(n, CONTROL_DT / decimation, DECIMATION // decimation, fixed, z, armature, batch=True)
    act = Actuators(robot)
    s_kp = torch.as_tensor(s_kp, dtype=gs.tc_float, device=gs.device)[:, None]
    s_kd = torch.as_tensor(s_kd, dtype=gs.tc_float, device=gs.device)[:, None]
    robot.set_dofs_kp(act.kp * s_kp, act.dofs)
    robot.set_dofs_kv(act.kd * s_kd, act.dofs)
    if compensate:
        h = CONTROL_DT / DECIMATION
        robot.set_dofs_armature(torch.clip(armature - act.kd * s_kd * h, min=0.0), act.dofs)
    robot.set_dofs_force_range((-act.effort).repeat(n, 1), act.effort.repeat(n, 1), act.dofs)
    robot.set_dofs_position(act.q0.repeat(n, 1), act.dofs, zero_velocity=True)
    q_tgt = targets(act, scenario, n_steps)
    qs = []
    for k in range(n_steps):
        robot.control_dofs_position(q_tgt[k].repeat(n, 1), act.dofs)
        for _ in range(decimation):
            if dc_cap:
                lo, hi = act.envelope(robot.get_dofs_velocity(act.dofs))
                robot.set_dofs_force_range(lo, hi, act.dofs)
            scene.step()
        qs.append(robot.get_dofs_position(act.dofs))
    return torch.stack(qs, dim=1), robot.get_pos()[:, 2]  # (n, steps, 12), (n,)


def rmse_by_group(q, q_ref, group):
    """q: (n, steps, 12), q_ref: (steps, 12) -> {group: (n,)} RMSE in rad."""
    err = (q - q_ref[None]) ** 2
    out = {}
    for g in GROUPS:
        cols = [i for i, gi in enumerate(group) if gi == g]
        out[g] = err[..., cols].mean(dim=(1, 2)).sqrt()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="error")

    n_steps = int(args.seconds / CONTROL_DT)
    kp_grid = np.round(np.arange(0.80, 1.401, 0.05), 3)
    kd_grid = np.round(np.arange(0.50, 2.001, 0.125), 3)
    S_KP, S_KD = (a.ravel() for a in np.meshgrid(kp_grid, kd_grid, indexing="ij"))
    i_nom = int(np.flatnonzero((S_KP == 1.0) & (S_KD == 1.0))[0])
    arm = get_cfgs()[0]["armature"]
    row = lambda label, e, i=0: print(f"  {label:<34} " + "  ".join(f"{g}={1e3 * e[g][i]:6.2f}" for g in GROUPS))

    for scenario in ("swing", "kick", "stand"):
        q_ref, tau_ref, q_tgt, z_ref, group = rollout_reference(scenario, n_steps, arm)
        track = rmse_by_group(q_ref[None], q_tgt, group)  # how hard the reference itself lags
        peak = {g: tau_ref[:, [i for i, gi in enumerate(group) if gi == g]].abs().max().item() for g in GROUPS}

        print(f"\n=== {scenario}: {args.seconds:g} s, joint RMSE vs. explicit DCMotor reference [mrad]")
        print(f"  {'reference lag behind its targets':<34} " + "  ".join(f"{g}={1e3 * track[g][0]:6.1f}" for g in GROUPS))
        print(f"  {'reference peak |tau| [N*m]':<34} " + "  ".join(f"{g}={peak[g]:6.1f}" for g in GROUPS))

        for compensate in (False, True):
            q, z = rollout_candidate(scenario, n_steps, arm, S_KP, S_KD, compensate=compensate)
            err = rmse_by_group(q, q_ref, group)
            tag = "compensated armature" if compensate else "raw armature 0.03"
            print(f"  -- {tag}")
            row("trained gains (kp x1, kd x1)", err, i_nom)
            for g in GROUPS:
                i = int(torch.argmin(err[g]))
                print(f"  {'best for ' + g:<34} kp x{S_KP[i]:.2f} kd x{S_KD[i]:.3f} -> {1e3 * err[g][i]:.2f}")
            if scenario == "stand":
                print(f"  {'final base height [m]':<34} reference {z_ref:.4f}, trained gains {z[i_nom]:.4f}")

        q_dec1, _ = rollout_candidate(scenario, n_steps, arm, [1.0], [1.0], decimation=1)
        row("cap refreshed at 50 Hz only", rmse_by_group(q_dec1, q_ref, group))
        q_static, _ = rollout_candidate(scenario, n_steps, arm, [1.0], [1.0], dc_cap=False)
        row("no torque-speed cap", rmse_by_group(q_static, q_ref, group))
        q_arm01, _ = rollout_candidate(scenario, n_steps, 0.1, [1.0], [1.0], compensate=False)
        row("Genesis default armature 0.1", rmse_by_group(q_arm01, q_ref, group))


if __name__ == "__main__":
    main()
