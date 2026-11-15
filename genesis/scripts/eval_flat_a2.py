"""Replay an A2 locomotion policy on flat rigid ground.

    python scripts/eval_flat_a2.py
    python scripts/eval_flat_a2.py -p Policies/v3_none/model_4999.pt --reporting --headless

Checkpoint paths may be absolute or relative to the repo root, and may contain spaces.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs

from envs.unitree_env import A2Env
from scripts.unitree_policy import cli
from scripts.unitree_policy.eval_cfg import DEFAULT_A2_POLICY, load_policy
from scripts.unitree_policy.reporting import run_rollout


def main():
    parser = argparse.ArgumentParser()
    cli.add_policy_arg(parser, DEFAULT_A2_POLICY)
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    cli.add_reporting_args(parser)
    parser.add_argument("--vx", type=float, default=None, help="commanded forward velocity; default keeps the config's 0.5 m/s")
    cli.add_phi_args(parser)
    args = parser.parse_args()

    ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cli.load_run(args.policy)
    if "a2" not in str(env_cfg.get("urdf", "")):
        parser.error(f"{ckpt_path} does not resolve to an A2 config (got urdf={env_cfg.get('urdf')!r})")

    if args.vx is not None:
        cli.drive_straight(command_cfg, args.vx)

    gs.init(backend=gs.cpu)

    env = A2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        record=args.reporting,
    )
    policy = load_policy(env, ckpt_path, train_cfg, **cli.phi_kwargs(args))

    reporter = cli.make_reporter(
        args, ckpt_path, env, scene="flat", description="Flat ground", settings=f"vx{command_cfg['lin_vel_x_range'][0]:g}", robot="unitree_a2", decimation=int(env_cfg.get("decimation", 1))
    )
    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter)
    cli.finish_report(reporter, env)


if __name__ == "__main__":
    main()
