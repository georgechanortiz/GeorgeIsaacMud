"""Replay the Unitree A2 locomotion policy walking into a deformable MPM mud patch.

The A2 counterpart of scripts/eval_mpm.py: same mud pit, same MPM materials, same reporting
(plots, metrics, params.json and rollout.mp4 under reports/<experiment>/<timestamp>/). What
differs is the robot, which comes from configs/a2_flat.py via the checkpoint's experiment
directory (a2_flat, a2_flat_erft, a2_flat_nomud):

  * actuators are the trained IsaacLab DCMotors, per group (hip/thigh 90/2.0 at 120 N*m,
    calf 120/2.5 at 180 N*m) with the torque-speed cap -- see scripts/tune_a2_pd.py;
  * the policy steps physics 4 times per control step (decimation), so --substeps must be a
    multiple of 4;
  * torque limits are scaled with --effort-scale rather than overridden with one value,
    because the A2's limits differ per joint group.

Mud. One 4-inch (0.1016 m) layer lying directly on the floor, as in the hardware runs, where
the feet sink through the mud to the firm bottom and the bottom, not the mud, carries the robot's
weight. The default material was chosen against the hardware bags (a2_mud/hardware_bags: one
grass walk, three mud walks with the same policy and gains). Those bags carry no foot forces, so
the comparison is on joint torques (/lowstate tau_est), as mud-minus-grass in the bags against
mud-minus-flat in the sim, each expressed as a share of the hardware grass value. Twenty
configurations of five materials were compared at 4 inches and 60 substeps:

    mud minus grass/flat            hardware   viscous liquid (default)
    front calf, mean |tau|            +18%          +21%
    front calf, 95th pct |tau|        +27%          +25%
    rear calf, mean |tau|              -9%           +2%
    forward speed                     -16%          -36%
    weight carried by the mud            -           4%   (feet reach the floor)

Ranked on calf load and speed (scripts/mud_calibration/score.py), viscous liquid takes the top
two places. E=2e4, a lower viscosity, ties on that score: it slows the robot a little less (-32%)
but under-shoots the calf peaks (+20%). E=1e5 is the default because it matches both calf force
terms within 3 points. Elasto-plastic (von Mises, yield 30-600 Pa), sand and snow follow within a
few points. Two hardware effects are not reproduced by
any MPM material here: the front-thigh torque rise and the near-doubling of mechanical power. Both
come from pulling the foot out of the mud during swing (suction), and Genesis' MPM-rigid coupling
only pushes on a foot, it never pulls. Every material also slows the sim robot about twice as much
as real mud did, so expect conservative speeds.

The liquid's viscosity is not a parameter of its own. Its shear stress is 2*mu*(F - R)F^T with F
the per-substep deformation, so the effective viscosity is mu * substep_dt, with mu = E/(2(1+nu)).
At E=1e5, nu=0.35 and 60 substeps per 20 ms step that is about 12 Pa*s. Changing --substeps
therefore changes the mud: keep it at 60 to stay on this calibration, and scale --mud-E with it.

    python scripts/eval_mpm_a2.py --reporting --headless
    python scripts/eval_mpm_a2.py --reporting --headless --material elasto_plastic --yield-stress 80 --mud-E 2e4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs

from scripts.eval_mpm import Go2MPMEnv
from scripts.unitree_policy.eval_cfg import load_cfgs, load_policy, resolve_policy
from scripts.unitree_policy.reporting import Reporter, run_rollout

DEFAULT_A2_POLICY = "/home/rml3/Documents/george/GeorgeIsaacMud/genesis/Policies/a2_flat/2026-10-08_21-07-46/model_4999.pt"

# The A2 is ~0.65 m long and stands ~0.45 m tall, against ~0.45 m / 0.30 m for the Go2, so the
# Go2 framing crops it. Same viewing direction, pulled back and aimed a little higher.
_A2_CAM_POS = (3.1, -2.6, 1.6)
_A2_CAM_LOOKAT = (0.9, 0.0, 0.3)


class A2MPMEnv(Go2MPMEnv):
    """Go2MPMEnv with camera framing sized for the A2. Physics and foot forces are unchanged:
    the A2's feet are merged into its *_calf links exactly as the Go2's are."""

    def _scene_kwargs(self, show_viewer):
        kwargs = super()._scene_kwargs(show_viewer)
        kwargs["viewer_options"] = gs.options.ViewerOptions(
            camera_pos=_A2_CAM_POS, camera_lookat=_A2_CAM_LOOKAT, camera_fov=40
        )
        return kwargs

    def _camera_kwargs(self):
        return dict(res=(1280, 720), pos=_A2_CAM_POS, lookat=_A2_CAM_LOOKAT, fov=40)


def _scale_effort(env_cfg, scale):
    """Scale every joint's torque limit and DCMotor saturation by `scale`, keeping their ratios."""
    for key in ("effort_limit", "saturation_effort"):
        value = env_cfg.get(key)
        if isinstance(value, dict):
            env_cfg[key] = {name: v * scale for name, v in value.items()}
        elif value is not None:
            env_cfg[key] = value * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--policy", type=str, default=DEFAULT_A2_POLICY, help="path to an A2 model_*.pt checkpoint, absolute or relative to the repo root")
    parser.add_argument("--cpu", action="store_true", help="run on CPU (much slower for MPM)")
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
    parser.add_argument("--reporting", action="store_true", help="run a finite rollout and write plots + params to reports/<experiment>/<timestamp>/")
    parser.add_argument("--headless", action="store_true", help="skip the interactive viewer window. With --reporting, the offscreen camera still records rollout.mp4.")
    parser.add_argument("--steps", type=int, default=1000, help="maximum rollout length when --reporting is set. The rollout normally ends sooner, when the robot reaches the far end of the pit (see --end-margin); this cap only ends runs where it stalls")
    parser.add_argument("--end-margin", type=float, default=0.55, metavar="M", help="end the rollout once the base is within M of the pit's far wall. 0.55 puts the front feet at the end of the mud while the body is still clear of the wall (the cut-off scripts/mud_calibration/run_one.py uses)")
    # --- mud layer (defaults are the hardware-calibrated viscous liquid, see the module docstring) ---
    parser.add_argument("--mud-depth", type=float, default=0.1016, metavar="M", help="Depth of the mud layer on the floor (m). Default 0.1016 = 4 inches, the hardware pit.")
    parser.add_argument("--material", choices=("liquid", "elasto_plastic", "sand", "snow", "elastic"), default="liquid", help="MPM material of the mud layer.")
    parser.add_argument("--mud-E", type=float, default=1e5, metavar="PA", help="Young's modulus (Pa). For liquid it sets the bulk stiffness and, with --substeps, the viscosity.")
    parser.add_argument("--mud-nu", type=float, default=0.35, metavar="NU", help="Poisson ratio; must stay < 0.5.")
    parser.add_argument("--mud-rho", type=float, default=1500.0, metavar="KGM3", help="Density (kg/m3). Wet mud is 1300-1800; most of the drag on the legs scales with it.")
    parser.add_argument("--inviscid", action="store_true", help="liquid only: drop the viscous shear stress.")
    parser.add_argument("--yield-stress", type=float, default=80.0, metavar="PA", help="elasto_plastic only: von Mises yield stress (Pa). 80 was the best elasto-plastic match.")
    parser.add_argument("--friction-angle", type=float, default=15.0, metavar="DEG", help="sand only: Drucker-Prager friction angle (deg).")
    parser.add_argument("--coup-friction", type=float, default=0.4, help="friction between the feet/legs and the mud.")
    parser.add_argument("--seed", type=int, default=None, metavar="N", help="Jitter the spawn pose by a seeded offset; otherwise repeats are bit-identical.")
    parser.add_argument("--spawn-jitter", type=float, default=0.03, metavar="M", help="Half-width of the uniform spawn offset in x and y applied when --seed is set.")
    parser.add_argument("--out-root", type=str, default=None, metavar="DIR", help="Write reports under this directory instead of reports/.")
    parser.add_argument("--tag", type=str, default=None, metavar="NAME", help="Report sub-directory name instead of the checkpoint's experiment name.")
    args = parser.parse_args()

    ckpt_path = resolve_policy(args.policy)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs(ckpt_path.parent)
    if "a2" not in str(env_cfg.get("urdf", "")):
        parser.error(f"{ckpt_path} does not resolve to an A2 config (got urdf={env_cfg.get('urdf')!r}); use scripts/eval_mpm.py for the Go2")
    decimation = int(env_cfg.get("decimation", 1))
    if args.substeps % decimation:
        parser.error(f"--substeps ({args.substeps}) must be a multiple of the A2's decimation ({decimation})")

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")
    reward_cfg["reward_scales"] = {}

    # MPM is only stable with a much smaller substep than rigid-only locomotion needs. The env
    # splits these across its `decimation` physics steps, so the substep length is unchanged.
    env_cfg["substeps"] = args.substeps
    if args.effort_scale != 1.0:
        print(f"effort limits scaled by {args.effort_scale}: hips/thighs {120 * args.effort_scale:g} N*m, calves {180 * args.effort_scale:g} N*m")
        _scale_effort(env_cfg, args.effort_scale)
    # Drive straight into the patch. With the heading controller on (as trained), ang_vel_range
    # is the clip on its correction, so it is left at the trained range.
    command_cfg["lin_vel_x_range"] = [args.vx, args.vx]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    if not command_cfg.get("heading_command"):
        command_cfg["ang_vel_range"] = [0.0, 0.0]

    # Same pit as scripts/eval_mpm.py (12' x 32" wide), so A2 and Go2 reports are comparable.
    mpm_cfg = {
        "grid_density": args.grid_density,
        "coup_friction": args.coup_friction,
        "coup_softness": 0.01,
        "wall_height": 0.6,                  # m; the A2 stands taller than the Go2's 0.5 m walls allowed for
        "wall_thickness": 0.03,
        "bounds_lower": (1, -0.4864, -0.1),
        "bounds_upper": (4.6576, 0.4864, 0.381),
        "patches": [
            {
                # one layer straight on the floor: the floor is the firm bottom the feet sink to
                "name": "mud",
                "material": args.material,
                "E": args.mud_E,
                "nu": args.mud_nu,
                "rho": args.mud_rho,
                "viscous": not args.inviscid,
                "yield_stress": args.yield_stress,
                "friction_angle": args.friction_angle,
                "patch_lower": (1.1, -0.38, 0.0),
                "patch_upper": (4.55, 0.38, args.mud_depth),
                "color": (0.55, 0.42, 0.30),
            },
        ],
    }

    # Spawn standing on the mud surface (the robot sinks through it to the floor), far enough past the leading edge to clear the near wall. The A2's
    # rear hips sit 0.26 m behind the base and, at the default 0.8 rad thigh angle, its knees fold
    # another 0.2 m back. The Go2's 0.4 m offset therefore put both rear knees inside the wall,
    # which pinned the rear legs and held the robot in place.
    patches = mpm_cfg["patches"]
    slab_top = max(p["patch_upper"][2] for p in patches)
    spawn_x = min(p["patch_lower"][0] for p in patches) + 0.6
    _, by, bz = env_cfg["base_init_pos"]
    env_cfg["base_init_pos"] = [spawn_x, by, bz + slab_top]
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        dx, dy = rng.uniform(-args.spawn_jitter, args.spawn_jitter, size=2)
        env_cfg["base_init_pos"] = [spawn_x + float(dx), by + float(dy), bz + slab_top]
        print(f"seed {args.seed}: spawn offset dx={dx:+.4f} m dy={dy:+.4f} m")

    # End the rollout at the far end of the pit rather than after a fixed time. Past it the robot
    # stands pinned against the far wall, and that idle time dilutes every averaged metric.
    pit_end_x = max(p["patch_upper"][0] for p in patches) - args.end_margin

    def reached_pit_end(env):
        return float(env.base_pos[0, 0]) >= pit_end_x

    env = A2MPMEnv(
        num_envs=1,  # MPM state is not batched cheaply; keep this at 1
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        record=args.reporting,
        mpm_cfg=mpm_cfg,
    )
    patch_counts = {p["name"]: int(e.n_particles) for p, e in zip(mpm_cfg["patches"], env.mud_patches)}
    total_particles = sum(patch_counts.values())
    print(f"MPM patches: {patch_counts} | total particles: {total_particles}")

    policy = load_policy(env, ckpt_path, train_cfg)

    reporter = None
    if args.reporting:
        reporter = Reporter(
            experiment=args.tag or ckpt_path.parent.parent.name,
            params={
                "robot": "unitree_a2",
                "policy": str(ckpt_path),
                "run": ckpt_path.parent.name,
                "scene": "mpm",
                "mpm_cfg": mpm_cfg,
                "mud": {
                    "material": args.material,
                    "depth_m": args.mud_depth,
                    "E": args.mud_E,
                    "nu": args.mud_nu,
                    "rho": args.mud_rho,
                    "viscous": not args.inviscid,
                    "yield_stress": args.yield_stress,
                    "friction_angle": args.friction_angle,
                    "coup_friction": args.coup_friction,
                    "seed": args.seed,
                },
                "effort_scale": args.effort_scale,
                "pit_end_x": pit_end_x,
                "end_margin": args.end_margin,
                "max_steps": args.steps,
                "mpm_particles": total_particles,
                "mpm_particles_by_patch": patch_counts,
                "num_envs": 1,
                "dt": env.dt,
                "decimation": decimation,
                "substeps": env_cfg["substeps"],
                "env_cfg": env_cfg,
                "command_cfg": command_cfg,
                "obs_scales": obs_cfg["obs_scales"],
            },
            reports_dir=Path(args.out_root) if args.out_root else REPO_ROOT / "reports",
        )

    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter, stop_when=reached_pit_end)
    reached = reached_pit_end(env)
    print(f"robot {'reached' if reached else 'did not reach'} the end of the pit (x >= {pit_end_x:.3f} m)")

    if reporter is not None:
        reporter.params["reached_pit_end"] = reached
        out_dir, metrics = reporter.finalize(env)
        print(f"report written to {out_dir}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
