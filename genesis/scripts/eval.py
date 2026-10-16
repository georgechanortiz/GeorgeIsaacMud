"""Replay a locomotion policy on flat rigid ground.

    python scripts/eval.py -p logs/2026-07-15_17-09-30/model_4999.pt
    python scripts/eval.py -p "logs/unitree_go2_flat/2026-05-28 - Baseline/model_4999.pt" --reporting

Checkpoint paths may be absolute or relative to the repo root, and may contain spaces.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs

from envs.go2_env import Go2Env
from scripts.unitree_policy import cli
from scripts.unitree_policy.eval_cfg import DEFAULT_POLICY, load_policy
from scripts.unitree_policy.reporting import run_rollout


def main():
    parser = argparse.ArgumentParser()
    cli.add_policy_arg(parser, DEFAULT_POLICY)
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    cli.add_reporting_args(parser)
    cli.add_phi_args(parser)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cli.load_run(args.policy)
    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        record=args.reporting,
    )
    policy = load_policy(env, ckpt_path, train_cfg, **cli.phi_kwargs(args))

    reporter = cli.make_reporter(args, ckpt_path, env, scene="flat")
    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter)
    cli.finish_report(reporter, env)


if __name__ == "__main__":
    main()
