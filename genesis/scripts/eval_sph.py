"""Replay a locomotion policy wading through an SPH water pool.

Same policy and observation pipeline as scripts/eval.py -- the only difference is
that the scene carries an SPH solver (envs/sph_env.py) and the robot walks through
standing liquid instead of over bare ground.

    python scripts/eval_sph.py -p logs/2026-07-15_17-09-30/model_4999.pt
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from envs.sph_env import Go2SPHEnv
from scripts.unitree_policy import cli
from scripts.unitree_policy.eval_cfg import DEFAULT_POLICY, load_policy
from scripts.unitree_policy.reporting import run_rollout


def main():
    parser = argparse.ArgumentParser()
    cli.add_policy_arg(parser, DEFAULT_POLICY)
    cli.add_cpu_arg(parser, "SPH")
    parser.add_argument("--substeps", type=int, default=20, help="SPH substep target is ~1e-3 s")
    parser.add_argument("--particle_size", type=float, default=0.02, help="smaller = more particles, costs speed")
    parser.add_argument("--mu", type=float, default=0.01, help="viscosity")
    parser.add_argument("--vx", type=float, default=0.5, help="commanded forward velocity")
    cli.add_reporting_args(parser)
    cli.add_phi_args(parser)
    args = parser.parse_args()

    cli.init_genesis(args.cpu)

    ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cli.load_run(args.policy)
    # SPH is only stable with a much smaller substep than rigid-only locomotion needs.
    env_cfg["substeps"] = args.substeps
    cli.drive_straight(command_cfg, args.vx)  # into the pool

    sph_cfg = {
        "particle_size": args.particle_size,
        "coup_friction": 0.1,
        "rho": 19000.0,
        "mu": args.mu,
        # SPH solver domain. Unlike MPM these bounds are used as-is with no inset,
        # and particles are clamped to them at runtime.
        "bounds_lower": (0.4, -0.5, -0.05),
        "bounds_upper": (1.6, 0.5, 0.4),
        # Where the liquid spawns, as min/max corners of an axis-aligned box.
        # Must sit inside bounds_lower/bounds_upper. Robot starts at x = 0 and walks +x.
        "pool_lower": (0.41, -0.4, 0.0),
        "pool_upper": (1.4, 0.4, 0.24),
    }

    env = Go2SPHEnv(
        num_envs=1,  # SPH state is not batched cheaply; keep this at 1
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        record=args.reporting,
        sph_cfg=sph_cfg,
    )
    print(f"SPH particles: {env.water.n_particles}")

    policy = load_policy(env, ckpt_path, train_cfg, **cli.phi_kwargs(args))

    reporter = cli.make_reporter(
        args, ckpt_path, env, scene="sph", sph_cfg=sph_cfg, sph_particles=int(env.water.n_particles)
    )
    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter)
    cli.finish_report(reporter, env)


if __name__ == "__main__":
    main()
