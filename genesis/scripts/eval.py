import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs

from envs.go2_env import Go2Env
from scripts.unitree_policy.eval_cfg import DEFAULT_POLICY, load_cfgs, load_policy, resolve_policy
from scripts.unitree_policy.reporting import Reporter, run_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        default=DEFAULT_POLICY,
        help="path to a model_*.pt checkpoint, absolute or relative to the repo root",
    )
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument(
        "--reporting",
        action="store_true",
        help="run a finite rollout and write plots + params to reports/<experiment>/<timestamp>/",
    )
    parser.add_argument("--steps", type=int, default=1000, help="rollout length when --reporting is set")
    parser.add_argument("--phi-smooth", type=float, default=1.0, metavar="ALPHA", help="EMA weight on phi's e_t estimate: 1.0 = off, smaller = heavier filtering (0.1 keeps ~90%% of the previous estimate). Damps the step-to-step jitter that destabilises the gait out of distribution.")
    parser.add_argument("--phi-rate", type=float, default=None, metavar="HZ", help="Re-estimate the mud parameters at this rate instead of every control step, holding the value in between. The control loop runs at 50 Hz, so --phi-rate 10 refreshes phi once every 5 steps. Mud properties do not change at control rate, so a slow estimator is both more realistic and much less jittery.")
    parser.add_argument("--phi-clamp", action="store_true", help="Clamp phi's e_t estimate to the range the actor was trained on, read from the run's params/env.yaml.")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    ckpt_path = resolve_policy(args.policy)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs(ckpt_path.parent)
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        record=args.reporting,
    )

    policy = load_policy(env, ckpt_path, train_cfg, phi_smooth=args.phi_smooth, phi_clamp=args.phi_clamp, phi_rate_hz=args.phi_rate)

    reporter = None
    if args.reporting:
        reporter = Reporter(
            experiment=ckpt_path.parent.parent.name,
            params={
                "policy": str(ckpt_path),
                "run": ckpt_path.parent.name,
                "scene": "flat",
                "num_envs": args.num_envs,
                "dt": env.dt,
                "env_cfg": env_cfg,
                "command_cfg": command_cfg,
                "obs_scales": obs_cfg["obs_scales"],
            },
            reports_dir=REPO_ROOT / "reports",
        )

    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter)

    if reporter is not None:
        out_dir, metrics = reporter.finalize(env)
        print(f"report written to {out_dir}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()


"""
# evaluation (relative to the repo root)
python scripts/eval.py -p logs/2026-07-15_17-09-30/model_4999.pt

# absolute paths and run names containing spaces both work
python scripts/eval.py -p "logs/unitree_go2_flat/2026-05-28 - Baseline/model_4999.pt"
"""
