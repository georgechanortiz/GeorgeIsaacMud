"""Replay the Go2 locomotion policy walking into a deformable MPM mud patch.

Same policy and observation pipeline as scripts/eval_flat_a2.py -- the only difference is
that the scene carries an MPM solver (envs/mpm_env.py) and the robot walks through
the two-layer Go2 mud pit (go2_mud_pit below) instead of bare ground: a firm
elastic base with a soft elasto-plastic layer on top, set by the --soft-* flags.

    python scripts/eval_mpm.py --reporting -p logs/Policy_B/model_1499.pt
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs

from envs import mud
from envs.mpm_env import A2MPMEnv
from scripts.unitree_policy import cli
from scripts.unitree_policy.eval_cfg import DEFAULT_POLICY, load_policy
from scripts.unitree_policy.reporting import run_rollout


def go2_mud_pit(soft_layer, grid_density=48):
    """The Go2 pit: a firm elastic base 6 cm deep, with `soft_layer` (one patch dict) on top."""
    stiff_base = {
        "name": "stiff_mud",
        "material": "elastic",  # firm base layer to hold the robot's weight
        "E": 1e6,
        "nu": 0.35,
        "rho": 1800.0,
        "patch_lower": (mud.PIT_X[0], -mud.PIT_HALF_WIDTH, 0.0),
        "patch_upper": (mud.PIT_X[1], mud.PIT_HALF_WIDTH, 0.06),
        "color": (0.30, 0.20, 0.12),
    }
    return mud._pit(grid_density, [stiff_base, soft_layer], wall_height=0.5)


def soft_layer(material, z_range, **params):
    """A soft top layer spanning the pit at heights z_range = (low, high)."""
    return {
        "name": "soft_mud",
        "material": material,
        **params,
        "patch_lower": (mud.PIT_X[0], -mud.PIT_HALF_WIDTH, z_range[0]),
        "patch_upper": (mud.PIT_X[1], mud.PIT_HALF_WIDTH, z_range[1]),
        "color": (0.78, 0.62, 0.46),
    }


class Go2MPMEnv(A2MPMEnv):
    """A2MPMEnv with the closer camera framing the smaller Go2 needs."""

    _POS, _LOOKAT = (2.6, -2.0, 1.3), (0.8, 0.0, 0.2)

    def _scene_kwargs(self, show_viewer):
        kwargs = super()._scene_kwargs(show_viewer)
        kwargs["viewer_options"] = gs.options.ViewerOptions(camera_pos=self._POS, camera_lookat=self._LOOKAT, camera_fov=40)
        return kwargs

    def _camera_kwargs(self):
        return dict(res=(1280, 720), pos=self._POS, lookat=self._LOOKAT, fov=40)


def main():
    parser = argparse.ArgumentParser()
    cli.add_policy_arg(parser, DEFAULT_POLICY)
    cli.add_cpu_arg(parser, "MPM")
    parser.add_argument("--substeps", type=int, default=60, help="MPM needs a small substep to stay stable; keep a bit above grid_density")
    parser.add_argument("--grid_density", type=int, default=48, help="cells per metre; raise to better resolve the small feet, costs speed (~cubic in particle count)")
    parser.add_argument("--vx", type=float, default=0.5, help="commanded forward velocity")
    parser.add_argument(
        "--effort-limit",
        type=float,
        default=None,
        metavar="NM",
        help="override the per-joint torque clip (N*m), replacing the run's env_cfg value (23.5 for the "
             "go2_mud_flat configs, the DCMotor effort_limit the policy trained under). Lowering it caps "
             "peak actuator effort and the resulting foot forces on the mud, at the cost of forward speed "
             "once the clip bites into the push-off torques.",
    )
    cli.add_reporting_args(parser)
    cli.add_phi_args(parser)
    # --- soft-mud material sweep (defaults reproduce the committed config) ---
    parser.add_argument("--yield-stress", type=float, default=600.0, metavar="PA", help="von Mises yield stress of the soft_mud layer (Pa).")
    parser.add_argument("--soft-E", type=float, default=2e4, metavar="PA", help="Young's modulus of the soft_mud layer (Pa).")
    parser.add_argument("--soft-nu", type=float, default=0.35, metavar="NU", help="Poisson ratio of the soft_mud layer. Saturated/undrained clay sits near 0.45; must stay < 0.5.")
    parser.add_argument("--soft-rho", type=float, default=1500.0, metavar="KGM3", help="Density of the soft_mud layer (kg/m3).")
    parser.add_argument("--soft-z", type=float, nargs=2, default=(0.06, 0.11), metavar=("LOW", "HIGH"), help="Vertical extent of the soft_mud layer (m). Default 0.06-0.11 sits flush on the stiff base (which ends at 0.06); a thicker layer or a gap stops the foot reaching the firm base and the robot cannot push off.")
    cli.add_seed_args(parser)
    args = parser.parse_args()

    cli.init_genesis(args.cpu)

    ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cli.load_run(args.policy)
    # MPM is only stable with a much smaller substep than rigid-only locomotion needs.
    env_cfg["substeps"] = args.substeps
    if args.effort_limit is not None:
        # Applied in UnitreeEnv via set_dofs_force_range, so this clips the PD output itself
        # rather than the action -- the policy is unchanged and simply operates against a
        # weaker actuator than it trained with.
        print(f"effort limit: {env_cfg.get('effort_limit')} -> {args.effort_limit} N*m")
        env_cfg["effort_limit"] = args.effort_limit
    cli.drive_straight(command_cfg, args.vx)  # into the patch

    # a soft top layer that yields and flows under a foot instead of resisting, much softer
    # than the base so it deforms easily before yielding
    mpm_cfg = go2_mud_pit(
        soft_layer(
            "elasto_plastic",
            args.soft_z,
            E=args.soft_E,
            nu=args.soft_nu,
            rho=args.soft_rho,
            yield_stress=args.yield_stress,
        ),
        grid_density=args.grid_density,
    )
    # Stand on the top layer instead of the ground, far enough past the leading edge that all
    # four feet land on it.
    mud.spawn_on_pit(env_cfg, mpm_cfg, offset=0.4)
    cli.jitter_spawn(env_cfg, args.seed, args.spawn_jitter)

    env = Go2MPMEnv(
        num_envs=1,  # MPM state is not batched cheaply; keep this at 1
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        record=args.reporting,
        mpm_cfg=mpm_cfg,
    )
    patch_counts = env.particle_counts()
    total_particles = sum(patch_counts.values())
    print(f"MPM patches: {patch_counts} | total particles: {total_particles}")

    policy = load_policy(env, ckpt_path, train_cfg, **cli.phi_kwargs(args))

    reporter = cli.make_reporter(
        args,
        ckpt_path,
        env,
        scene="mpm",
        mpm_cfg=mpm_cfg,
        soft_mud_sweep={
            "yield_stress": args.yield_stress,
            "E": args.soft_E,
            "nu": args.soft_nu,
            "rho": args.soft_rho,
            "seed": args.seed,
            "soft_z": list(args.soft_z),
        },
        mpm_particles=total_particles,
        mpm_particles_by_patch=patch_counts,
    )
    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter)
    cli.finish_report(reporter, env)


if __name__ == "__main__":
    main()
