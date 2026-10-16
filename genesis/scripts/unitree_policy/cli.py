"""Command-line flags and setup steps shared by the eval entry points (scripts/eval*.py).

Each eval script adds the flag groups it supports, builds its own scene, and leaves the rest
to these helpers:

    ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cli.load_run(args.policy)
    cli.drive_straight(command_cfg, args.vx)
    env = SomeEnv(...)
    reporter = cli.make_reporter(args, ckpt_path, env, scene="mpm", ...)
    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter)
    cli.finish_report(reporter, env)
"""

from pathlib import Path

import numpy as np

import genesis as gs

from scripts.unitree_policy.eval_cfg import REPO_ROOT, load_cfgs, resolve_policy
from scripts.unitree_policy.reporting import Reporter

# ---- flags -------------------------------------------------------------------------------------


def add_policy_arg(parser, default):
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        default=default,
        help="path to a model_*.pt checkpoint, absolute or relative to the repo root",
    )


def add_cpu_arg(parser, solver):
    parser.add_argument("--cpu", action="store_true", help=f"run on CPU (much slower for {solver})")


def add_reporting_args(parser, steps_help="rollout length when --reporting is set"):
    parser.add_argument(
        "--reporting",
        action="store_true",
        help="run a finite rollout and write plots + params to reports/<experiment>/<timestamp>/",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="skip the interactive viewer window to speed up the rollout. With --reporting, the offscreen "
             "camera still records rollout.mp4 as usual -- only the live GUI window is skipped.",
    )
    parser.add_argument("--steps", type=int, default=1000, help=steps_help)
    parser.add_argument("--out-root", type=str, default=None, metavar="DIR", help="Write reports under this directory instead of reports/.")
    parser.add_argument("--tag", type=str, default=None, metavar="NAME", help="Report sub-directory name instead of the checkpoint's experiment name.")


def add_phi_args(parser):
    """Damping for the adaptation module of RMA / privileged-obs actors (see eval_cfg.stabilise_adaptation)."""
    parser.add_argument("--phi-smooth", type=float, default=1.0, metavar="ALPHA", help="EMA weight on phi's e_t estimate: 1.0 = off, smaller = heavier filtering (0.1 keeps ~90%% of the previous estimate). Damps the step-to-step jitter that destabilises the gait out of distribution.")
    parser.add_argument("--phi-rate", type=float, default=None, metavar="HZ", help="Re-estimate the mud parameters at this rate instead of every control step, holding the value in between. The control loop runs at 50 Hz, so --phi-rate 10 refreshes phi once every 5 steps. Mud properties do not change at control rate, so a slow estimator is both more realistic and much less jittery.")
    parser.add_argument("--phi-clamp", action="store_true", help="Clamp phi's e_t estimate to the range the actor was trained on, read from the run's params/env.yaml.")


def phi_kwargs(args):
    """The --phi-* flags as load_policy keyword arguments."""
    return dict(phi_smooth=args.phi_smooth, phi_clamp=args.phi_clamp, phi_rate_hz=args.phi_rate)


def add_seed_args(parser):
    parser.add_argument("--seed", type=int, default=None, metavar="N", help="Jitter the spawn pose by a seeded offset. Without it every repeat starts from the same state (commands and heading are sampled from degenerate ranges) and differs only by the GPU particle solver's floating-point nondeterminism, about 1-5%% on the report metrics.")
    parser.add_argument("--spawn-jitter", type=float, default=0.03, metavar="M", help="Half-width of the uniform spawn offset in x and y applied when --seed is set.")


# ---- setup -------------------------------------------------------------------------------------


def init_genesis(cpu):
    gs.init(backend=gs.cpu if cpu else gs.gpu, precision="32", logging_level="warning")


def load_run(policy):
    """Resolve a checkpoint and rebuild its configs, with rewards switched off (eval never trains).

    Returns (ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg).
    """
    ckpt_path = resolve_policy(policy)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs(ckpt_path.parent)
    reward_cfg["reward_scales"] = {}
    return ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def drive_straight(command_cfg, vx):
    """Pin the command to `vx` straight ahead.

    Without a heading controller there is nothing to hold the line, so an open-loop zero yaw rate
    is the best available. With one configured (as every run here was trained), ang_vel_range is
    the *clip* on its correction rather than the command -- zeroing it would clamp the correction
    away and let the walk curve -- so it is left at the trained range.
    """
    command_cfg["lin_vel_x_range"] = [vx, vx]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    if not command_cfg.get("heading_command"):
        command_cfg["ang_vel_range"] = [0.0, 0.0]


def jitter_spawn(env_cfg, seed, half_width):
    """Offset the spawn x/y by a seeded uniform draw in [-half_width, half_width]. No-op without a seed.

    Otherwise every repeat starts from the same state, and differs only by floating-point
    nondeterminism in the GPU particle solvers (MPM, SPH): about 1-5% on the report metrics,
    while rigid-only rollouts repeat bit for bit. Offsetting the spawn changes which part of the
    patch each foot strikes and the gait phase at contact, without altering the commanded task.
    """
    if seed is None:
        return
    rng = np.random.default_rng(seed)
    dx, dy = rng.uniform(-half_width, half_width, size=2)
    x, y, z = env_cfg["base_init_pos"]
    env_cfg["base_init_pos"] = [x + float(dx), y + float(dy), z]
    print(f"seed {seed}: spawn offset dx={dx:+.4f} m dy={dy:+.4f} m")


# ---- reporting ---------------------------------------------------------------------------------


def make_reporter(args, ckpt_path, env, scene, **params):
    """A Reporter for this run when --reporting is set, else None.

    params.json carries the run identity, `scene`, the scene-specific `params`, then the env
    settings every report shares.
    """
    if not args.reporting:
        return None
    return Reporter(
        experiment=args.tag or ckpt_path.parent.parent.name,
        params={
            "policy": str(ckpt_path),
            "run": ckpt_path.parent.name,
            "scene": scene,
            **params,
            "num_envs": env.num_envs,
            "dt": env.dt,
            "substeps": env.env_cfg["substeps"],
            "env_cfg": env.env_cfg,
            "command_cfg": env.command_cfg,
            "obs_scales": env.obs_cfg["obs_scales"],
        },
        reports_dir=Path(args.out_root) if args.out_root else REPO_ROOT / "reports",
    )


def finish_report(reporter, env):
    """Write the report and print its metrics. No-op without a reporter."""
    if reporter is None:
        return
    out_dir, metrics = reporter.finalize(env)
    print(f"report written to {out_dir}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
