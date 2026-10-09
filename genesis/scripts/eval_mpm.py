"""Replay the Go2 locomotion policy walking into a deformable MPM mud patch.

Same policy and observation pipeline as scripts/eval.py -- the only difference is
that the scene carries an MPM solver and the robot walks through a patch of
elasto-plastic material instead of bare ground.

    python scripts/eval_mpm.py -reporting -p logs/Policy_B/model_1499.pt
    
    
    2026-08-01_13-52-45_phase2/model_500.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs

from envs.go2_env import Go2Env
from scripts.unitree_policy.eval_cfg import DEFAULT_POLICY, load_cfgs, load_policy, resolve_policy
from scripts.unitree_policy.reporting import Reporter, run_rollout


class Go2MPMEnv(Go2Env):
    """Go2Env plus a slab of MPM material lying in the robot's path."""

    def __init__(self, *args, mpm_cfg, **kwargs):
        self.mpm_cfg = mpm_cfg
        self._calf_idx = None  # global link indices of the foot-bearing links, filled lazily post-build
        self.foot_names = None
        super().__init__(*args, **kwargs)

    def _scene_kwargs(self, show_viewer):
        kwargs = super()._scene_kwargs(show_viewer)
        kwargs["mpm_options"] = gs.options.MPMOptions(
            lower_bound=self.mpm_cfg["bounds_lower"],
            upper_bound=self.mpm_cfg["bounds_upper"],
            grid_density=self.mpm_cfg["grid_density"],
            # feet are smaller than a grid cell; CPIC is needed for small rigid bodies to couple properly
            enable_CPIC=True,
        )
        # walls add collision pairs on top of the ~20 the bare locomotion policy uses
        kwargs["rigid_options"] = gs.options.RigidOptions(
            enable_self_collision=False,
            tolerance=1e-5,
            max_collision_pairs=100,
        )
        kwargs["vis_options"] = gs.options.VisOptions(
            visualize_mpm_boundary=True,
            rendered_envs_idx=[0],
        )
        kwargs["viewer_options"] = gs.options.ViewerOptions(
            camera_pos=(2.6, -2.0, 1.3),
            camera_lookat=(0.8, 0.0, 0.2),
            camera_fov=40,
        )
        return kwargs

    def _rigid_material(self):
        # needs_coup is already the default, but coupling is the point of this demo, so make it explicit
        # coup_softness widens the coupling reach past the default 2mm -- needed since the grid spacing
        # (~21mm at grid_density 48) is otherwise too coarse to catch the small feet
        return gs.materials.Rigid(
            needs_coup=True,
            coup_friction=self.mpm_cfg["coup_friction"],
            coup_softness=self.mpm_cfg["coup_softness"],
        )

    def _camera_kwargs(self):
        # match the viewer angle set in _scene_kwargs
        return dict(res=(1280, 720), pos=(2.6, -2.0, 1.3), lookat=(0.8, 0.0, 0.2), fov=40)

    def _patch_box(self, patch):
        """Resolve one patch's patch_lower/patch_upper into the (pos, size) gs.morphs.Box wants.

        The MPM solver insets its usable region by 3 grid cells on every axis, so the
        valid spawn region shrinks as grid_density drops. The solver rejects out-of-bounds
        entities but doesn't say which axis overflowed, so check it here first.
        """
        inset = 3.0 / self.mpm_cfg["grid_density"]
        label = patch.get("name", "patch")
        lower = np.asarray(patch["patch_lower"], dtype=float)
        upper = np.asarray(patch["patch_upper"], dtype=float)
        bounds_lower = np.asarray(self.mpm_cfg["bounds_lower"], dtype=float) + inset
        bounds_upper = np.asarray(self.mpm_cfg["bounds_upper"], dtype=float) - inset

        for axis, name in enumerate("xyz"):
            if upper[axis] <= lower[axis]:
                raise ValueError(
                    f"patch '{label}': patch_upper must exceed patch_lower on {name}: "
                    f"got {lower[axis]:.3f} .. {upper[axis]:.3f}"
                )
            if lower[axis] < bounds_lower[axis] or upper[axis] > bounds_upper[axis]:
                raise ValueError(
                    f"patch '{label}' overflows the MPM domain on {name}: patch spans "
                    f"[{lower[axis]:.3f}, {upper[axis]:.3f}] but the usable region is "
                    f"[{bounds_lower[axis]:.3f}, {bounds_upper[axis]:.3f}] "
                    f"(bounds inset by 3/grid_density = {inset:.4f}). "
                    f"Widen bounds_{'lower' if lower[axis] < bounds_lower[axis] else 'upper'}, "
                    f"raise grid_density, or shrink the patch."
                )

        pos = tuple(float(v) for v in (lower + upper) / 2)
        size = tuple(float(v) for v in (upper - lower))
        return pos, size

    def _check_no_overlap(self, patches):
        """Overlapping patches sample particles into the same cells -> instant blow-up.

        Two axis-aligned boxes overlap iff they overlap on all three axes.
        """
        boxes = [
            (np.asarray(p["patch_lower"], float), np.asarray(p["patch_upper"], float), p.get("name", i))
            for i, p in enumerate(patches)
        ]
        for i in range(len(boxes)):
            la, ua, ni = boxes[i]
            for j in range(i + 1, len(boxes)):
                lb, ub, nj = boxes[j]
                if np.all(la < ub) and np.all(lb < ua):
                    raise ValueError(
                        f"MPM patches '{ni}' and '{nj}' overlap; give them disjoint "
                        f"patch_lower/patch_upper boxes (a gap on any one axis is enough)."
                    )

    def _add_extra_entities(self):
        # One entity per patch, all sharing the single MPM solver built in _scene_kwargs.
        # self.mud_patches keeps them in config order for reporting / introspection.
        patches = self.mpm_cfg["patches"]
        self._check_no_overlap(patches)
        self.mud_patches = []
        for patch in patches:
            pos, size = self._patch_box(patch)
            if patch["material"] == "elastic":
                material = gs.materials.MPM.Elastic(E=patch["E"], nu=patch["nu"], rho=patch["rho"])
            elif patch["material"] == "sand":
                material = gs.materials.MPM.Sand(E=patch["E"], nu=patch["nu"], rho=patch["rho"], friction_angle=patch["friction_angle"])
            else:
                material = gs.materials.MPM.ElastoPlastic(
                    E=patch["E"],
                    nu=patch["nu"],
                    rho=patch["rho"],
                    use_von_mises=True,
                    von_mises_yield_stress=patch["yield_stress"],
                )
            entity = self.scene.add_entity(
                material=material,
                morph=gs.morphs.Box(pos=pos, size=size),
                surface=gs.surfaces.Default(color=patch.get("color", (0.35, 0.25, 0.15)), vis_mode="particle"),
            )
            self.mud_patches.append(entity)
        self._add_containment_walls()

    def _add_containment_walls(self):
        """Thin fixed rigid walls around the patch perimeter so the robot can't walk off
        the raised slab and fall to the ground.

        needs_coup=False keeps the walls invisible to the MPM solver -- they only block
        the robot, not the mud. Surface is semi-transparent so the near wall doesn't
        occlude the camera.
        """
        lowers = np.asarray([p["patch_lower"] for p in self.mpm_cfg["patches"]], dtype=float)
        uppers = np.asarray([p["patch_upper"] for p in self.mpm_cfg["patches"]], dtype=float)
        x_lo, y_lo = lowers[:, 0].min(), lowers[:, 1].min()
        x_hi, y_hi = uppers[:, 0].max(), uppers[:, 1].max()
        t = float(self.mpm_cfg["wall_thickness"])
        h = float(self.mpm_cfg["wall_height"])
        cx, cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
        x_len, y_len = x_hi - x_lo, y_hi - y_lo
        # (pos, size); the +/-y boards run the full length so the four corners close.
        boards = [
            ((x_lo - t / 2, cy, h / 2), (t, y_len, h)),          # near (-x), behind the spawn
            ((x_hi + t / 2, cy, h / 2), (t, y_len, h)),          # far (+x), the edge it marches toward
            ((cx, y_lo - t / 2, h / 2), (x_len + 2 * t, t, h)),  # -y side
            ((cx, y_hi + t / 2, h / 2), (x_len + 2 * t, t, h)),  # +y side
        ]
        wall_material = gs.materials.Rigid(needs_coup=False)
        self.walls = [
            self.scene.add_entity(
                material=wall_material,
                morph=gs.morphs.Box(pos=pos, size=size, fixed=True),
                surface=gs.surfaces.Default(color=(0.55, 0.57, 0.62), opacity=0.2, vis_mode="visual"),
            )
            for pos, size in boards
        ]

    def get_foot_forces(self):
        """Per-foot reaction force from the mud, world frame, shape (num_envs, num_feet, 3).

        get_links_net_contact_force() only reports rigid-rigid contacts, which are ~zero
        for feet standing on MPM material -- the mud pushes on the feet through the
        MPM<->rigid coupler instead, landing in dyn_state.links.cfrc_coupling_vel. The
        foot sphere is fixed to the calf link, so the *_calf links carry the foot forces.
        """
        if self._calf_idx is None:
            calves = [link for link in self.robot.links if link.name.endswith("_calf")]
            self._calf_idx = [link.idx for link in calves]  # global indices into the solver's link block
            self.foot_names = [link.name.replace("_calf", " foot") for link in calves]  # FL/FR/RL/RR
        cfrc = self.scene.rigid_solver.dyn_state.links.cfrc_coupling_vel.to_numpy()
        forces = -cfrc[self._calf_idx]           # (num_feet, num_envs, 3)
        return np.transpose(forces, (1, 0, 2))   # (num_envs, num_feet, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--policy", type=str, default=DEFAULT_POLICY, help="path to a model_*.pt checkpoint, absolute or relative to the repo root")
    parser.add_argument("--cpu", action="store_true", help="run on CPU (much slower for MPM)")
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
    parser.add_argument("--steps", type=int, default=1000, help="rollout length when --reporting is set")
    parser.add_argument("--phi-smooth", type=float, default=1.0, metavar="ALPHA", help="EMA weight on phi's e_t estimate: 1.0 = off, smaller = heavier filtering (0.1 keeps ~90%% of the previous estimate). Damps the step-to-step jitter that destabilises the gait out of distribution.")
    parser.add_argument("--phi-rate", type=float, default=None, metavar="HZ", help="Re-estimate the mud parameters at this rate instead of every control step, holding the value in between. The control loop runs at 50 Hz, so --phi-rate 10 refreshes phi once every 5 steps. Mud properties do not change at control rate, so a slow estimator is both more realistic and much less jittery.")
    parser.add_argument("--phi-clamp", action="store_true", help="Clamp phi's e_t estimate to the range the actor was trained on, read from the run's params/env.yaml.")
    # --- soft-mud material sweep (defaults reproduce the committed config) ---
    parser.add_argument("--yield-stress", type=float, default=600.0, metavar="PA", help="von Mises yield stress of the soft_mud layer (Pa).")
    parser.add_argument("--soft-E", type=float, default=2e4, metavar="PA", help="Young's modulus of the soft_mud layer (Pa).")
    parser.add_argument("--soft-nu", type=float, default=0.35, metavar="NU", help="Poisson ratio of the soft_mud layer. Saturated/undrained clay sits near 0.45; must stay < 0.5.")
    parser.add_argument("--soft-rho", type=float, default=1500.0, metavar="KGM3", help="Density of the soft_mud layer (kg/m3).")
    parser.add_argument("--seed", type=int, default=None, metavar="N", help="Jitter the spawn pose by a seeded offset. The rollout is otherwise fully deterministic (commands and heading are sampled from degenerate ranges), so repeats without this are bit-identical and carry no variance information.")
    parser.add_argument("--spawn-jitter", type=float, default=0.03, metavar="M", help="Half-width of the uniform spawn offset in x and y applied when --seed is set.")
    parser.add_argument("--soft-z", type=float, nargs=2, default=(0.06, 0.11), metavar=("LOW","HIGH"), help="Vertical extent of the soft_mud layer (m). Default 0.06-0.11 sits flush on the stiff base (which ends at 0.06); a thicker layer or a gap stops the foot reaching the firm base and the robot cannot push off.")
    parser.add_argument("--out-root", type=str, default=None, metavar="DIR", help="Write reports under this directory instead of reports/.")
    parser.add_argument("--tag", type=str, default=None, metavar="NAME", help="Use this as the report sub-directory name instead of the checkpoint's experiment name.")

    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    ckpt_path = resolve_policy(args.policy)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs(ckpt_path.parent)
    reward_cfg["reward_scales"] = {}

    # MPM is only stable with a much smaller substep than rigid-only locomotion needs.
    env_cfg["substeps"] = args.substeps
    if args.effort_limit is not None:
        # Applied in Go2Env via set_dofs_force_range, so this clips the PD output itself
        # rather than the action -- the policy is unchanged and simply operates against a
        # weaker actuator than it trained with.
        print(f"effort limit: {env_cfg.get('effort_limit')} -> {args.effort_limit} N*m")
        env_cfg["effort_limit"] = args.effort_limit
    # Drive straight into the patch.
    command_cfg["lin_vel_x_range"] = [args.vx, args.vx]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    if not command_cfg.get("heading_command"):
        # No heading controller to hold the line, so an open-loop zero is the best available.
        # With one configured, ang_vel_range is the *clip* on its correction rather than the
        # command -- zeroing it here would clamp the correction away and let the walk curve.
        command_cfg["ang_vel_range"] = [0.0, 0.0]

    mpm_cfg = {
        # --- solver-level: one MPM domain, shared by every patch below ---
        "grid_density": args.grid_density,   # cells per metre. higher = finer particles, costs speed (~cubic)
        "coup_friction": 0.4,                # higher = more grip between the feet and the material
        "coup_softness": 0.01,               # metres; reach of the foot<->material coupling, see _rigid_material

        # containment walls around the patch perimeter, see _add_containment_walls
        "wall_height": 0.5,                  # m, from the ground up; taller than the mud + robot stance
        "wall_thickness": 0.03,              # m

        "bounds_lower": (1, -0.4864, -0.1),
        "bounds_upper": (4.6576, 0.4864, 0.381), # 12' x 32" wide
        "patches": [
            {
                "name": "stiff_mud",
                "material": "elastic",       # Firm base layer to hold the robot's weight
                "E": 1e6,                    # Young's modulus (Pa)
                "nu": 0.35,                  # Poisson ratio. higher = more volume-preserving; must stay < 0.5 (lam blows up near it)
                "rho": 1800.0,               # density kg/m3. Not a force-magnitude knob (that's E) -- this sets particle inertia / elastic wave speed, so keep it physical.
                "patch_lower": (1.1, -0.38, 0.0),
                "patch_upper": (4.55, 0.38, 0.06),
                "color": (0.30, 0.20, 0.12),
            },


            {
                "name": "soft_mud",
                "material": "elasto_plastic",  # thin top layer; yields and flows under a foot instead of resisting
                "E": args.soft_E,              # much softer than the base layer, so it deforms easily before yielding
                "nu": args.soft_nu,
                "rho": args.soft_rho,
                # "friction_angle": 30.0,        # degrees
                "yield_stress": args.yield_stress,
                "patch_lower": (1.1, -0.38, args.soft_z[0]),
                "patch_upper": (4.55, 0.38, args.soft_z[1]),
                "color": (0.78, 0.62, 0.46),
            },



            # {
            #     "name": "medium_mud",
            #     "material": "sand",  # thin top layer; yields and flows under a foot instead of resisting
            #     "E": 2e4,                      # much softer than the base layer, so it deforms easily before yielding
            #     "nu": 0.35,
            #     "rho": 1500.0,
            #     "yield_stress": 600,
            #     "patch_lower": (1.1, -0.38, 0.065),
            #     "patch_upper": (4.55, 0.38, 0.15),
            #     "color": (0.78, 0.62, 0.46),
            # },


            # {
            #     "name": "hard_mud",
            #     "material": "sand",  # thin top layer; yields and flows under a foot instead of resisting
            #     "E": 2e4,                      # much softer than the base layer, so it deforms easily before yielding
            #     "nu": 0.35,
            #     "rho": 1500.0,
            #     "yield_stress": 600,
            #     "patch_lower": (1.1, -0.38, 0.065),
            #     "patch_upper": (4.55, 0.38, 0.15),
            #     "color": (0.78, 0.62, 0.46),
            # },






        ],
    }

    # spawn on top of the pit, lifted to the top layer's surface so the feet start there
    # instead of on the ground; offset past the leading edge so all four feet land on it
    patches = mpm_cfg["patches"]
    slab_top = max(p["patch_upper"][2] for p in patches)
    spawn_x = min(p["patch_lower"][0] for p in patches) + 0.4
    _, by, bz = env_cfg["base_init_pos"]
    env_cfg["base_init_pos"] = [spawn_x, by, bz + slab_top]
    if args.seed is not None:
        # The rollout is otherwise deterministic, so repeats need a perturbation to carry any
        # variance. Offsetting the spawn changes which part of the patch each foot strikes and
        # the gait phase at contact, without altering the commanded task.
        _rng = np.random.default_rng(args.seed)
        _dx, _dy = _rng.uniform(-args.spawn_jitter, args.spawn_jitter, size=2)
        env_cfg["base_init_pos"] = [spawn_x + float(_dx), by + float(_dy), bz + slab_top]
        print(f"seed {args.seed}: spawn offset dx={_dx:+.4f} m dy={_dy:+.4f} m")

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
    patch_counts = {p["name"]: int(e.n_particles) for p, e in zip(mpm_cfg["patches"], env.mud_patches)}
    total_particles = sum(patch_counts.values())
    print(f"MPM patches: {patch_counts} | total particles: {total_particles}")

    policy = load_policy(env, ckpt_path, train_cfg, phi_smooth=args.phi_smooth, phi_clamp=args.phi_clamp, phi_rate_hz=args.phi_rate)

    reporter = None
    if args.reporting:
        reporter = Reporter(
            experiment=args.tag or ckpt_path.parent.parent.name,
            params={
                "policy": str(ckpt_path),
                "run": ckpt_path.parent.name,
                "scene": "mpm",
                "mpm_cfg": mpm_cfg,
                "soft_mud_sweep": {
                    "yield_stress": args.yield_stress,
                    "E": args.soft_E,
                    "nu": args.soft_nu,
                    "rho": args.soft_rho,
                    "seed": args.seed,
                    "soft_z": list(args.soft_z),
                },
                "mpm_particles": total_particles,
                "mpm_particles_by_patch": patch_counts,
                "num_envs": 1,
                "dt": env.dt,
                "substeps": env_cfg["substeps"],
                "env_cfg": env_cfg,
                "command_cfg": command_cfg,
                "obs_scales": obs_cfg["obs_scales"],
            },
            reports_dir=Path(args.out_root) if args.out_root else REPO_ROOT / "reports",
        )

    run_rollout(env, policy, steps=args.steps if args.reporting else None, reporter=reporter)

    if reporter is not None:
        out_dir, metrics = reporter.finalize(env)
        print(f"report written to {out_dir}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
