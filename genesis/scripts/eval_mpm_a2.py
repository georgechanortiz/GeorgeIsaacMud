"""Replay an A2 locomotion policy walking through a pit of MPM mud.

The robot comes from configs/a2_flat.py, picked from the checkpoint's experiment directory.

Mud: one 4-inch layer on the floor, as in the hardware pit; the feet sink through it to the
bottom, which carries the robot's weight. The default (viscous liquid, E=1e5, nu=0.35,
rho=1500) was chosen by comparing mud-minus-flat in sim with mud-minus-grass in the A2 hardware
bags, at 60 substeps (scripts/_SS/mud_calibration):

    change in mud                   hardware   sim (default)
    front calf, mean |tau|            +18%        +21%
    front calf, 95th pct |tau|        +27%        +25%
    rear calf, mean |tau|              -9%         +2%
    forward speed                     -16%        -36%

No material reproduces the hardware's front-thigh torque and power increase. That comes from
suction when the foot is pulled out, and Genesis' MPM coupling only pushes on a foot.

The liquid's effective viscosity is mu * substep_dt, with mu = E / (2(1 + nu)), so changing
--substeps changes the mud: keep 60 to stay on the calibration, or scale --mud-E with it.

    python scripts/eval_mpm_a2.py --reporting --headless
    python scripts/eval_mpm_a2.py --reporting --headless --material elasto_plastic --yield-stress 80 --mud-E 2e4
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from envs import mud
from envs.mpm_env import A2MPMEnv
from scripts.unitree_policy import cli
from scripts.unitree_policy.eval_cfg import DEFAULT_A2_POLICY, load_policy
from scripts.unitree_policy.reporting import run_rollout


def _scale_effort(env_cfg, scale):
    """Scale every joint's torque limit and DCMotor saturation by `scale`, keeping their ratios."""
    for key in ("effort_limit", "saturation_effort"):
        value = env_cfg.get(key)
        if isinstance(value, dict):
            env_cfg[key] = {name: v * scale for name, v in value.items()}
        elif value is not None:
            env_cfg[key] = value * scale


def _sci(x):
    """1e5 -> "1e5", 3.5e6 -> "3.5e6"."""
    mantissa, exponent = f"{x:e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def _mud_description(args):
    return f"{args.material.replace('_', '-')} mud, {args.mud_depth * 100:.1f} cm deep"


def _mud_settings(args, parser):
    """Compact settings string for reports/index.csv: material, depth, E, density and substeps, plus any non-default setting."""
    parts = [f"mud-{args.material}", f"d{args.mud_depth:g}", f"E{_sci(args.mud_E)}", f"rho{args.mud_rho:g}", f"ss{args.substeps}"]
    if args.material == "elasto_plastic":
        parts.append(f"ys{args.yield_stress:g}")
    if args.material == "sand":
        parts.append(f"fa{args.friction_angle:g}")
    if args.inviscid:
        parts.append("inviscid")
    optional = {"vx": "vx", "mud_nu": "nu", "coup_friction": "cf", "grid_density": "gd", "effort_scale": "eff", "seed": "seed"}
    for dest, short in optional.items():
        value = getattr(args, dest)
        if value != parser.get_default(dest):
            parts.append(f"{short}{value:g}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser()
    cli.add_policy_arg(parser, DEFAULT_A2_POLICY)
    cli.add_cpu_arg(parser, "MPM")
    parser.add_argument("--substeps", type=int, default=60, help="MPM substeps per 50 Hz control step; must be a multiple of the A2's decimation (4). Keep a bit above grid_density")
    parser.add_argument("--grid_density", type=int, default=48, help="cells per metre; raise to better resolve the feet, costs speed (~cubic in particle count)")
    parser.add_argument("--vx", type=float, default=0.35, help="commanded forward velocity. 0.35 makes the sim walk at 0.41 m/s on flat ground, the hardware grass-walk speed the mud was calibrated against")
    parser.add_argument(
        "--effort-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="scale every joint's torque limit (and DCMotor saturation effort) by S. 1.0 is the trained "
             "actuator: 120 N*m hips/thighs, 180 N*m calves. Below 1 the policy runs against a weaker "
             "actuator than it trained with, capping push-off torques and foot forces on the mud.",
    )
    cli.add_reporting_args(
        parser,
        steps_help="maximum rollout length when --reporting is set. The rollout normally ends sooner, when the "
                   "robot reaches the far end of the pit (see --end-margin); this cap only ends runs where it stalls",
    )
    parser.add_argument("--end-margin", type=float, default=0.55, metavar="M", help="end the rollout once the base is within M of the pit's far wall. 0.55 puts the front feet at the end of the mud while the body is still clear of the wall")
    # mud layer; defaults are the calibrated material (see the module docstring)
    parser.add_argument("--mud-depth", type=float, default=0.1016, metavar="M", help="Depth of the mud layer on the floor (m). Default 0.1016 = 4 inches, the hardware pit.")
    parser.add_argument("--material", choices=("liquid", "elasto_plastic", "sand", "snow", "elastic"), default="liquid", help="MPM material of the mud layer.")
    parser.add_argument("--mud-E", type=float, default=1e5, metavar="PA", help="Young's modulus (Pa). For liquid it sets the bulk stiffness and, with --substeps, the viscosity.")
    parser.add_argument("--mud-nu", type=float, default=0.35, metavar="NU", help="Poisson ratio; must stay < 0.5.")
    parser.add_argument("--mud-rho", type=float, default=1500.0, metavar="KGM3", help="Density (kg/m3). Wet mud is 1300-1800; most of the drag on the legs scales with it.")
    parser.add_argument("--inviscid", action="store_true", help="liquid only: drop the viscous shear stress.")
    parser.add_argument("--yield-stress", type=float, default=80.0, metavar="PA", help="elasto_plastic only: von Mises yield stress (Pa). 80 was the best elasto-plastic match.")
    parser.add_argument("--friction-angle", type=float, default=15.0, metavar="DEG", help="sand only: Drucker-Prager friction angle (deg).")
    parser.add_argument("--coup-friction", type=float, default=0.4, help="friction between the feet/legs and the mud.")
    cli.add_seed_args(parser)
    args = parser.parse_args()

    ckpt_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cli.load_run(args.policy)
    if "a2" not in str(env_cfg.get("urdf", "")):
        parser.error(f"{ckpt_path} does not resolve to an A2 config (got urdf={env_cfg.get('urdf')!r})")
    decimation = int(env_cfg.get("decimation", 1))
    if args.substeps % decimation:
        parser.error(f"--substeps ({args.substeps}) must be a multiple of the A2's decimation ({decimation})")

    cli.init_genesis(args.cpu)

    # MPM needs far smaller substeps than rigid-only simulation.
    env_cfg["substeps"] = args.substeps
    if args.effort_scale != 1.0:
        print(f"effort limits scaled by {args.effort_scale}: hips/thighs {120 * args.effort_scale:g} N*m, calves {180 * args.effort_scale:g} N*m")
        _scale_effort(env_cfg, args.effort_scale)
    cli.drive_straight(command_cfg, args.vx)

    mud_material = {
        "material": args.material,
        "E": args.mud_E,
        "nu": args.mud_nu,
        "rho": args.mud_rho,
        "viscous": not args.inviscid,
        "yield_stress": args.yield_stress,
        "friction_angle": args.friction_angle,
    }
    mpm_cfg = mud.a2_mud_pit(mud_material, args.mud_depth, grid_density=args.grid_density, coup_friction=args.coup_friction)

    # 0.6 m in from the near wall: any closer and the A2's rear knees start inside the wall.
    mud.spawn_on_pit(env_cfg, mpm_cfg, offset=0.6)
    cli.jitter_spawn(env_cfg, args.seed, args.spawn_jitter)

    # Stop at the far end of the pit: time spent pinned against the wall would skew the averages.
    end_x = mud.pit_end_x(mpm_cfg, args.end_margin)

    def reached_pit_end(env):
        return float(env.base_pos[0, 0]) >= end_x

    env = A2MPMEnv(
        num_envs=1,  # MPM is not batched cheaply
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

    policy = load_policy(env, ckpt_path, train_cfg)

    reporter = cli.make_reporter(
        args,
        ckpt_path,
        env,
        scene="mpm",
        description=_mud_description(args),
        settings=_mud_settings(args, parser),
        robot="unitree_a2",
        mpm_cfg=mpm_cfg,
        mud={
            **mud_material,
            "depth_m": args.mud_depth,
            "coup_friction": args.coup_friction,
            "seed": args.seed,
        },
        effort_scale=args.effort_scale,
        pit_end_x=end_x,
        end_margin=args.end_margin,
        max_steps=args.steps,
        mpm_particles=total_particles,
        mpm_particles_by_patch=patch_counts,
        decimation=decimation,
    )

    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter, stop_when=reached_pit_end)
    reached = reached_pit_end(env)
    print(f"robot {'reached' if reached else 'did not reach'} the end of the pit (x >= {end_x:.3f} m)")
    if reporter is not None:
        reporter.params["reached_pit_end"] = reached
    cli.finish_report(reporter, env)


if __name__ == "__main__":
    main()
