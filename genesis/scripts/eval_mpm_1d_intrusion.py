"""1D single-leg mud intrusion bench test.

For validating a reduced-order mud reaction force model (e.g. "A Reduced-Order Mud
Reaction Force Model for Robotic Foot-Mud Interactions") against Genesis's MPM ground
truth. Unlike scripts/eval_mpm.py, no policy walks the robot through the pit -- the base
is welded to the world above it, three legs hold a default stance in the air, and the
fourth (--leg, default FR) is driven straight down into the mud at a controlled rate via
inverse kinematics, held, and withdrawn. This is a penetrometer-style constant-rate-of-
penetration test: the same methodology real MPM/CPT validation studies use (a probe
pushed into a chamber sample at constant rate, measuring resistance vs. depth), applied
to an actual robot foot instead of a bare cone.

Standalone by design: it does not subclass Go2Env/Go2MPMEnv from envs/go2_env.py or
scripts/eval_mpm.py. Those are built around a floating-base robot driven by a trained
locomotion policy's 12-dof position actions each control step, with a full observation/
reward/command/termination pipeline behind them -- none of which applies to a bench test
with a welded base, no policy, and one leg under direct IK control. Reusing them would
mean fighting that machinery more than it would save; this script owns its own minimal
scene setup instead. The mud patches/materials/walls intentionally mirror eval_mpm.py's
build_mpm_cfg by hand (see MPM_CFG below) so the two scripts test the same mud -- if you
change one, change the other.

    python scripts/eval_mpm_1d_intrusion.py --depth 0.06 --rate 0.02 --headless
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat

import matplotlib

matplotlib.use("Agg")  # report files only; never open a GUI alongside the genesis viewer
import matplotlib.pyplot as plt

# Style tokens duplicated from scripts/unitree_policy/reporting.py (not imported) to keep
# this script's only dependency on the rest of the repo at the CLI/config-loading layer.
SERIES = ("#2a78d6", "#008300", "#e87ba4")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e0e0dc"

# Go2 URDF, matching envs/go2_env.py's GO2_URDF but resolved independently so this script
# doesn't import from that module.
GO2_URDF = str(REPO_ROOT / "assets" / "go2" / "urdf" / "go2.urdf")

# Every *_foot link is a fixed child of its *_calf link, at this local offset in all four
# legs (confirmed in the URDF) -- and Genesis's URDF loader collapses fixed-joint children
# into their parent at build time, so no separate "*_foot" RigidLink exists to query
# directly. The foot's world position is therefore computed by hand each step:
# calf_pos + R(calf_quat) @ FOOT_LOCAL_OFFSET. get_foot_forces() (below) already reads
# reaction force off the *_calf link for the same reason.
FOOT_LOCAL_OFFSET = (0.0, 0.0, -0.213)
# The foot's collision geometry is a sphere of this radius; "depth into the mud" is
# measured at its bottom -- the actual contact surface -- not the offset point above.
FOOT_SPHERE_RADIUS = 0.022

# Thigh and calf links are each 0.213 m, so a fully-extended leg reaches this far below
# the hip. Used to check up front that the commanded depth is actually reachable.
MAX_LEG_REACH = 2 * 0.213

LEGS = ("FR", "FL", "RR", "RL")


def build_mpm_cfg(grid_density):
    """Same mud pit as scripts/eval_mpm.py's build_mpm_cfg -- kept as a hand-copied
    duplicate rather than an import so this script has zero coupling to that file.
    """
    return {
        "grid_density": grid_density,
        "coup_friction": 0.4,
        "coup_softness": 0.01,
        "wall_height": 0.5,
        "wall_thickness": 0.03,
        "bounds_lower": (1, -0.4864, -0.1),
        "bounds_upper": (4.6576, 0.4864, 0.381),
        "patches": [
            {
                "name": "stiff_mud",
                "material": "elastic",
                "E": 1e6,
                "nu": 0.35,
                "rho": 1800.0,
                "patch_lower": (1.1, -0.38, 0.0),
                "patch_upper": (4.55, 0.38, 0.06),
                "color": (0.30, 0.20, 0.12),
            },
            {
                "name": "soft_mud",
                "material": "sand",
                "E": 2e4,
                "nu": 0.35,
                "rho": 1500.0,
                "friction_angle": 30.0,
                "patch_lower": (1.1, -0.38, 0.065),
                "patch_upper": (4.55, 0.38, 0.15),
                "color": (0.78, 0.62, 0.46),
            },
        ],
    }


def _patch_box(patch, mpm_cfg):
    """Resolve one patch's patch_lower/patch_upper into the (pos, size) gs.morphs.Box wants.

    Copied from Go2MPMEnv._patch_box (scripts/eval_mpm.py) -- see that copy for the
    reasoning behind the grid-cell inset check.
    """
    inset = 3.0 / mpm_cfg["grid_density"]
    label = patch.get("name", "patch")
    lower = np.asarray(patch["patch_lower"], dtype=float)
    upper = np.asarray(patch["patch_upper"], dtype=float)
    bounds_lower = np.asarray(mpm_cfg["bounds_lower"], dtype=float) + inset
    bounds_upper = np.asarray(mpm_cfg["bounds_upper"], dtype=float) - inset

    for axis, name in enumerate("xyz"):
        if upper[axis] <= lower[axis]:
            raise ValueError(f"patch '{label}': patch_upper must exceed patch_lower on {name}")
        if lower[axis] < bounds_lower[axis] or upper[axis] > bounds_upper[axis]:
            raise ValueError(
                f"patch '{label}' overflows the MPM domain on {name}: spans "
                f"[{lower[axis]:.3f}, {upper[axis]:.3f}], usable region is "
                f"[{bounds_lower[axis]:.3f}, {bounds_upper[axis]:.3f}]"
            )

    pos = tuple(float(v) for v in (lower + upper) / 2)
    size = tuple(float(v) for v in (upper - lower))
    return pos, size


class Go2MPM1DIntrusionEnv:
    """A Go2 with its base welded to the world above an MPM mud pit.

    Not a subclass of Go2Env: that class's constructor hardcodes a floating-base URDF
    morph and builds an RL-shaped observation/reward/command pipeline this bench test has
    no use for. This class builds only what a fixed-base single-leg intrusion test needs.
    """

    def __init__(self, env_cfg, mpm_cfg, bench_dof_pos, show_viewer=False, record=False, cam_kwargs=None):
        self.env_cfg = env_cfg
        self.mpm_cfg = mpm_cfg
        self.dt = 0.02  # 50 Hz control rate, matching Go2Env
        self.num_envs = 1  # MPM state is not batched cheaply

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg["substeps"]),
            mpm_options=gs.options.MPMOptions(
                lower_bound=mpm_cfg["bounds_lower"],
                upper_bound=mpm_cfg["bounds_upper"],
                grid_density=mpm_cfg["grid_density"],
                enable_CPIC=True,  # feet are smaller than a grid cell; needed for small rigid bodies to couple
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=100,
            ),
            vis_options=gs.options.VisOptions(visualize_mpm_boundary=True, rendered_envs_idx=[0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.6, -2.0, 1.3), camera_lookat=(0.8, 0.0, 0.2), camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        rigid_material = gs.materials.Rigid(
            needs_coup=True,
            coup_friction=mpm_cfg["coup_friction"],
            coup_softness=mpm_cfg["coup_softness"],
        )

        self.plane = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True), material=rigid_material
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=GO2_URDF,
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
                fixed=True,  # welded to the world -- only the 12 leg dofs are free
            ),
            material=rigid_material,
        )

        self._calf_idx = None
        self.foot_names = None
        self.mud_patches = []
        self._build_mud_and_walls()

        self.cam = None
        if record:
            self.cam = self.scene.add_camera(GUI=False, **cam_kwargs)

        self.scene.build(n_envs=self.num_envs)

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.robot.set_dofs_kp([env_cfg["kp"]] * 12, self.motors_dof_idx)
        self.robot.set_dofs_kv([env_cfg["kd"]] * 12, self.motors_dof_idx)
        effort_limit = env_cfg.get("effort_limit")
        if effort_limit is not None:
            self.robot.set_dofs_force_range([-effort_limit] * 12, [effort_limit] * 12, self.motors_dof_idx)

        self.bench_dof_pos = bench_dof_pos.to(gs.device)
        # Hard-init to the bench pose (idle legs tucked, see build_bench_pose). Unlike
        # Go2Env.reset(), there's no floating-base pos/quat to fold in -- the base is
        # welded, so qpos is just the 12 leg joints.
        self.robot.set_dofs_position(self.bench_dof_pos.unsqueeze(0), self.motors_dof_idx, zero_velocity=True)
        if self.cam is not None:
            self.cam.start_recording()

    # ---- mud construction (mirrors Go2MPMEnv._add_extra_entities / _add_containment_walls) --
    def _build_mud_and_walls(self):
        patches = self.mpm_cfg["patches"]
        for patch in patches:
            pos, size = _patch_box(patch, self.mpm_cfg)
            if patch["material"] == "elastic":
                material = gs.materials.MPM.Elastic(E=patch["E"], nu=patch["nu"], rho=patch["rho"])
            elif patch["material"] == "sand":
                material = gs.materials.MPM.Sand(
                    E=patch["E"], nu=patch["nu"], rho=patch["rho"], friction_angle=patch["friction_angle"]
                )
            else:
                material = gs.materials.MPM.ElastoPlastic(
                    E=patch["E"], nu=patch["nu"], rho=patch["rho"],
                    use_von_mises=True, von_mises_yield_stress=patch["yield_stress"],
                )
            entity = self.scene.add_entity(
                material=material,
                morph=gs.morphs.Box(pos=pos, size=size),
                surface=gs.surfaces.Default(color=patch.get("color", (0.35, 0.25, 0.15)), vis_mode="particle"),
            )
            self.mud_patches.append(entity)

        lowers = np.asarray([p["patch_lower"] for p in patches], dtype=float)
        uppers = np.asarray([p["patch_upper"] for p in patches], dtype=float)
        x_lo, y_lo = lowers[:, 0].min(), lowers[:, 1].min()
        x_hi, y_hi = uppers[:, 0].max(), uppers[:, 1].max()
        t = float(self.mpm_cfg["wall_thickness"])
        h = float(self.mpm_cfg["wall_height"])
        cx, cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
        x_len, y_len = x_hi - x_lo, y_hi - y_lo
        boards = [
            ((x_lo - t / 2, cy, h / 2), (t, y_len, h)),
            ((x_hi + t / 2, cy, h / 2), (t, y_len, h)),
            ((cx, y_lo - t / 2, h / 2), (x_len + 2 * t, t, h)),
            ((cx, y_hi + t / 2, h / 2), (x_len + 2 * t, t, h)),
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

        Identical to Go2MPMEnv.get_foot_forces -- see that copy (scripts/eval_mpm.py) for
        why this reads dyn_state.links.cfrc_coupling_vel off the *_calf links.
        """
        if self._calf_idx is None:
            calves = [link for link in self.robot.links if link.name.endswith("_calf")]
            self._calf_idx = [link.idx for link in calves]
            self.foot_names = [link.name.replace("_calf", " foot") for link in calves]
        cfrc = self.scene.rigid_solver.dyn_state.links.cfrc_coupling_vel.to_numpy()
        forces = -cfrc[self._calf_idx]
        return np.transpose(forces, (1, 0, 2))


def _style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def build_bench_pose(env_cfg, leg, tuck_thigh, tuck_calf):
    """Joint targets for the bench test: the intruding leg at its default angles (IK
    overrides it every step anyway), the other three folded up out of the way.

    Tucking matters because the two geometric constraints otherwise fight each other. At
    the run's default stance the feet hang ~0.30 m below the base, while a fully extended
    leg only reaches ~0.43 m -- so a base high enough to keep the idle feet out of the mud
    leaves the intruding leg unable to reach into it. Folding the idle legs (measured:
    ~0.085 m below base at thigh 1.8 / calf -2.7) decouples the two.
    """
    pose = []
    for name in env_cfg["joint_names"]:
        if name.startswith(f"{leg}_"):
            pose.append(env_cfg["default_joint_angles"][name])
        elif name.endswith("_thigh_joint"):
            pose.append(tuck_thigh)
        elif name.endswith("_calf_joint"):
            pose.append(tuck_calf)
        else:  # hip: keep the default abduction, tucking is a thigh/calf motion
            pose.append(env_cfg["default_joint_angles"][name])
    return torch.tensor(pose, dtype=gs.tc_float)


def _foot_world_pos(calf_link):
    """World position of the foot sphere centre -- calf_pos + R(calf_quat) @ offset.

    There is no RigidLink for the foot itself to query (see FOOT_LOCAL_OFFSET above), so
    this is computed by hand each call.
    """
    pos = calf_link.get_pos(relative=False)  # (1, 3)
    quat = calf_link.get_quat(relative=False)  # (1, 4)
    offset = torch.tensor([FOOT_LOCAL_OFFSET], dtype=gs.tc_float, device=gs.device)
    return pos + transform_by_quat(offset, quat)


def _measure_mud_surface(env, foot_xy, percentile, radius=0.05):
    """Free-surface height of the settled mud directly under the foot.

    Sampled in a small column around the foot's (x, y) rather than over the whole pit: the
    patch is 3.4 m long, and what the intrusion test needs is the surface at the point of
    contact, not a basin-wide average that a distant bulge could skew.
    """
    cols = []
    for entity in env.mud_patches:
        pos = entity.get_particles_pos()
        pos = pos[0] if pos.ndim == 3 else pos  # (n_particles, 3), env 0
        pos = np.asarray(pos.detach().cpu() if hasattr(pos, "detach") else pos)
        near = (np.abs(pos[:, 0] - foot_xy[0]) < radius) & (np.abs(pos[:, 1] - foot_xy[1]) < radius)
        if near.any():
            cols.append(pos[near, 2])
    if not cols:
        raise SystemExit(
            f"no MPM particles found within {radius} m of the foot at "
            f"({foot_xy[0]:.3f}, {foot_xy[1]:.3f}) -- is the foot over the patch?"
        )
    return float(np.percentile(np.concatenate(cols), percentile))


def _solve_leg_target(env, calf_link, leg_dof_idx, leg_joint_idx, ik_init, foot_xy, slab_top, depth_cmd):
    """IK the intruding foot to (foot_xy, depth_cmd below slab_top), holding x/y fixed.

    Returns (next ik warm-start, full 12-dof position target) -- the other three legs keep
    their tucked bench pose.
    """
    # The sphere's bottom is the contact surface, so the IK target for the sphere *centre*
    # sits one radius above the depth being commanded.
    target_z = slab_top - depth_cmd + FOOT_SPHERE_RADIUS
    target_pos = torch.tensor([[foot_xy[0], foot_xy[1], target_z]], dtype=gs.tc_float, device=gs.device)

    qpos = env.robot.inverse_kinematics(
        link=calf_link,
        pos=target_pos,
        local_point=FOOT_LOCAL_OFFSET,
        dofs_idx_local=leg_dof_idx,
        init_qpos=ik_init.unsqueeze(0),
        respect_joint_limit=True,
    )
    solved = qpos[0] if qpos.dim() == 2 else qpos
    full_target = env.bench_dof_pos.clone()
    full_target[leg_joint_idx] = solved[leg_dof_idx]
    return solved.detach().clone(), full_target


def build_depth_trajectory(dt, start_clearance, depth, rate, hold_s, retract):
    """Depth-below-mud-surface command over time: ramp down at `rate`, hold, ramp back up.

    Negative depth means the foot is still above the surface (start_clearance above it).
    Constant-rate ramps rather than a smooth profile on purpose -- this is a constant-
    rate-of-penetration test, the same loading protocol real CPT/penetrometer chamber
    tests use, so the recorded force-vs-depth curve is directly comparable to that
    literature instead of being confounded by an arbitrary velocity profile.
    """
    t_descend = (depth + start_clearance) / rate
    n_descend = max(1, round(t_descend / dt))
    descend = np.linspace(-start_clearance, depth, n_descend)

    n_hold = max(1, round(hold_s / dt))
    hold = np.full(n_hold, depth)

    if retract:
        ascend = descend[::-1].copy()
        return np.concatenate([descend, hold, ascend])
    return np.concatenate([descend, hold])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-p", "--policy", type=str, required=True,
        help="path to a model_*.pt checkpoint, absolute or relative to the repo root. Not loaded or run -- "
             "only used to source env_cfg (kp/kd/effort_limit/joint_names/default_joint_angles) from that "
             "run's params/env.yaml, so recorded torques are comparable to a specific trained policy's actuator.",
    )
    parser.add_argument("--leg", choices=LEGS, default="FR", help="which leg intrudes")
    parser.add_argument("--cpu", action="store_true", help="run on CPU (much slower for MPM)")
    parser.add_argument("--substeps", type=int, default=60, help="MPM needs a small substep to stay stable")
    parser.add_argument("--grid_density", type=int, default=48, help="cells per metre; same knob as eval_mpm.py")
    parser.add_argument("--kp", type=float, default=None, help="override the run's PD stiffness (leg holds position "
                         "more or less stiffly against mud resistance); default uses the run's own kp")
    parser.add_argument("--effort-limit", type=float, default=None, help="override the run's torque clip (N*m)")
    parser.add_argument("--start-clearance", type=float, default=0.03, metavar="M",
                         help="foot height above the mud surface at the start of the ramp")
    parser.add_argument("--depth", type=float, default=0.06, metavar="M",
                         help="max intrusion depth below the mud surface")
    parser.add_argument("--rate", type=float, default=0.02, metavar="M/S",
                         help="constant intrusion (and retraction) velocity -- quasi-static, matching "
                              "CPT/penetrometer chamber-test convention")
    parser.add_argument("--hold-s", type=float, default=1.0, help="dwell time at max depth before retracting")
    parser.add_argument("--mud-settle-s", type=float, default=1.0, metavar="S",
                         help="step the sim with the leg held clear before the test, letting the MPM material "
                              "settle under gravity, then re-measure the free surface from particle positions "
                              "and use that as the depth datum. The patches are spawned as uniform boxes that "
                              "are not in gravitational equilibrium, so the nominal slab top is not where the "
                              "mud actually is -- without this, depth is measured against the wrong reference.")
    parser.add_argument("--surface-percentile", type=float, default=99.0, metavar="P",
                         help="percentile of settled particle z used as the free-surface height. Not the max: "
                              "a few stray splashed particles would otherwise set the datum.")
    parser.add_argument("--settle-s", type=float, default=0.5, metavar="S",
                         help="hold the foot at its start position (above the mud) for this long before the "
                              "ramp begins, and discard those steps. The leg starts at the tucked bench pose, "
                              "so without this the first commanded step is a jump that rings the leg -- a "
                              "dynamic transient contaminating what should be a quasi-static measurement.")
    parser.add_argument("--no-retract", action="store_true", help="skip the withdrawal phase (descend + hold only)")
    parser.add_argument("--suspend-height", type=float, default=0.30, metavar="M",
                         help="base height above the mud surface. Bounded on both sides: too high and the "
                              "intruding leg (max reach ~0.43 m) cannot get to --depth; too low and the tucked "
                              "idle legs dip into the mud. Checked at startup.")
    parser.add_argument("--tuck-thigh", type=float, default=1.8, metavar="RAD",
                         help="thigh angle for the three non-intruding legs; folds them clear of the mud")
    parser.add_argument("--tuck-calf", type=float, default=-2.7, metavar="RAD",
                         help="calf angle for the three non-intruding legs (URDF limit is -2.72)")
    parser.add_argument("--headless", action="store_true", help="skip the interactive viewer window")
    parser.add_argument("--record", action="store_true", help="save rollout.mp4 alongside the plots")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.unitree_policy.eval_cfg import load_cfgs, resolve_policy

    ckpt_path = resolve_policy(args.policy)
    env_cfg, _, _, _, _ = load_cfgs(ckpt_path.parent)
    env_cfg["substeps"] = args.substeps
    if args.kp is not None:
        env_cfg["kp"] = args.kp
    if args.effort_limit is not None:
        env_cfg["effort_limit"] = args.effort_limit

    mpm_cfg = build_mpm_cfg(args.grid_density)
    slab_top = max(p["patch_upper"][2] for p in mpm_cfg["patches"])
    cx = (mpm_cfg["bounds_lower"][0] + mpm_cfg["bounds_upper"][0]) / 2.0
    _, base_y, _ = env_cfg["base_init_pos"]
    # Suspended above the pit's centre. Hips sit at base height, so a foot can be driven at
    # most MAX_LEG_REACH below the base -- fail loudly here rather than silently recording a
    # run where the foot never touches the mud (all-zero forces).
    reachable_depth = MAX_LEG_REACH - args.suspend_height
    if args.depth >= reachable_depth:
        raise SystemExit(
            f"--depth {args.depth:.3f} m is out of reach: with --suspend-height "
            f"{args.suspend_height:.3f} m the {args.leg} foot only reaches "
            f"{reachable_depth:.3f} m below the mud surface (leg reach {MAX_LEG_REACH:.3f} m). "
            f"Lower --suspend-height or reduce --depth."
        )
    env_cfg["base_init_pos"] = [cx, base_y, slab_top + args.suspend_height]

    leg = args.leg
    bench_dof_pos = build_bench_pose(env_cfg, leg, args.tuck_thigh, args.tuck_calf)

    cam_kwargs = dict(res=(1280, 720), pos=(cx + 0.6, -1.0, slab_top + 0.6), lookat=(cx, 0.0, slab_top), fov=40)
    env = Go2MPM1DIntrusionEnv(
        env_cfg, mpm_cfg, bench_dof_pos,
        show_viewer=not args.headless, record=args.record, cam_kwargs=cam_kwargs,
    )
    patch_counts = {p["name"]: int(e.n_particles) for p, e in zip(mpm_cfg["patches"], env.mud_patches)}
    print(f"MPM patches: {patch_counts} | total particles: {sum(patch_counts.values())}")

    leg_joint_idx = [i for i, name in enumerate(env_cfg["joint_names"]) if name.startswith(f"{leg}_")]
    leg_dof_idx = env.motors_dof_idx[leg_joint_idx]
    calf_link = env.robot.get_link(f"{leg}_calf")

    # Confirm the tucked idle legs actually clear the mud in the built scene, rather than
    # trusting the nominal numbers in build_bench_pose's docstring. A tucked foot dipping
    # into the patch would add a second contact and quietly corrupt the measurement.
    idle_clearances = {}
    for other in LEGS:
        if other == leg:
            continue
        z = float(_foot_world_pos(env.robot.get_link(f"{other}_calf"))[0, 2]) - FOOT_SPHERE_RADIUS
        idle_clearances[other] = z - slab_top
    worst = min(idle_clearances.values())
    if worst <= 0.0:
        raise SystemExit(
            f"tucked idle legs reach the mud (clearances above surface: "
            f"{ {k: round(v, 4) for k, v in idle_clearances.items()} }). Raise --suspend-height, "
            f"or fold them further with --tuck-thigh / --tuck-calf."
        )
    print(f"idle leg clearance above mud: {worst:.3f} m (min over {sorted(idle_clearances)})")

    # Foot's neutral (x, y), measured rather than assumed -- robust to whatever the leg
    # geometry actually is instead of hand-deriving it from link lengths. Intrusion moves
    # z only, holding x/y fixed: a genuine 1D test.
    foot_xy = _foot_world_pos(calf_link)[0, :2].detach().cpu().numpy().copy()

    # Let the mud reach gravitational equilibrium, then take the depth datum from where the
    # material actually ended up rather than the nominal slab top it was spawned at.
    n_mud_settle = max(0, round(args.mud_settle_s / env.dt))
    for _ in range(n_mud_settle):
        env.robot.control_dofs_position(env.bench_dof_pos.unsqueeze(0), env.motors_dof_idx)
        env.scene.step()
    if n_mud_settle:
        settled_top = _measure_mud_surface(env, foot_xy, args.surface_percentile)
        print(f"mud settled {args.mud_settle_s:.2f}s: surface {slab_top:.4f} -> {settled_top:.4f} m "
              f"({(settled_top - slab_top) * 1e3:+.1f} mm)")
        slab_top = settled_top

    depth_traj = build_depth_trajectory(
        env.dt, args.start_clearance, args.depth, args.rate, args.hold_s, retract=not args.no_retract
    )
    print(f"{leg} intrusion: {len(depth_traj)} steps, depth 0 -> {args.depth:.3f} m at {args.rate:.3f} m/s"
          + ("" if args.no_retract else " (with retraction)"))

    ik_init = env.bench_dof_pos.clone()

    # Settle: drive to the ramp's start height and let the leg come to rest there, so the
    # recorded run begins from a quiet state instead of mid-swing.
    n_settle = max(0, round(args.settle_s / env.dt))
    if n_settle:
        settle_traj = np.full(n_settle, depth_traj[0])
        for depth_cmd in settle_traj:
            ik_init, full_target = _solve_leg_target(
                env, calf_link, leg_dof_idx, leg_joint_idx, ik_init, foot_xy, slab_top, depth_cmd
            )
            env.robot.control_dofs_position(full_target.unsqueeze(0), env.motors_dof_idx)
            env.scene.step()
        residual = abs((slab_top - (float(_foot_world_pos(calf_link)[0, 2]) - FOOT_SPHERE_RADIUS)) - depth_traj[0])
        print(f"settled: {n_settle} steps, residual depth error {residual * 1e3:.2f} mm")

    rows = {k: [] for k in ("t", "depth_cmd", "depth_measured", "fx", "fy", "fz", "tau_hip", "tau_thigh", "tau_calf")}

    for i, depth_cmd in enumerate(depth_traj):
        ik_init, full_target = _solve_leg_target(
            env, calf_link, leg_dof_idx, leg_joint_idx, ik_init, foot_xy, slab_top, depth_cmd
        )
        env.robot.control_dofs_position(full_target.unsqueeze(0), env.motors_dof_idx)
        env.scene.step()

        torque = env.robot.get_dofs_control_force(env.motors_dof_idx)[0].detach().cpu().numpy()
        tau_leg = torque[leg_joint_idx]

        foot = env.get_foot_forces()[0]  # (num_feet, 3)
        leg_foot_idx = env.foot_names.index(f"{leg} foot")
        fx, fy, fz = foot[leg_foot_idx]

        foot_z_now = float(_foot_world_pos(calf_link)[0, 2])
        depth_measured = slab_top - (foot_z_now - FOOT_SPHERE_RADIUS)

        rows["t"].append(i * env.dt)
        rows["depth_cmd"].append(float(depth_cmd))
        rows["depth_measured"].append(depth_measured)
        rows["fx"].append(float(fx))
        rows["fy"].append(float(fy))
        rows["fz"].append(float(fz))
        rows["tau_hip"].append(float(tau_leg[0]))
        rows["tau_thigh"].append(float(tau_leg[1]))
        rows["tau_calf"].append(float(tau_leg[2]))

        if env.cam is not None:
            env.cam.render()

    data = {k: np.asarray(v) for k, v in rows.items()}

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPO_ROOT / "logs_for_1d" / f"{stamp}_{ckpt_path.parent.name}_{leg}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_force_time(out_dir, data)
    _plot_depth_time(out_dir, data)
    _plot_force_depth(out_dir, data, n_descend=int(np.argmax(data["depth_cmd"] >= args.depth - 1e-9)) + 1)
    _plot_torque_time(out_dir, data, leg)

    np.savez(out_dir / "intrusion_data.npz", **data)

    metrics = {
        "peak_abs_fz_n": float(np.abs(data["fz"]).max()),
        "peak_abs_fx_n": float(np.abs(data["fx"]).max()),
        "peak_abs_fy_n": float(np.abs(data["fy"]).max()),
        "peak_abs_tau_hip_nm": float(np.abs(data["tau_hip"]).max()),
        "peak_abs_tau_thigh_nm": float(np.abs(data["tau_thigh"]).max()),
        "peak_abs_tau_calf_nm": float(np.abs(data["tau_calf"]).max()),
        "fz_at_max_depth_n": float(data["fz"][np.argmax(data["depth_cmd"])]),
    }
    params = {
        "policy": str(ckpt_path),
        "run": ckpt_path.parent.name,
        "leg": leg,
        "foot_sphere_radius_m": FOOT_SPHERE_RADIUS,
        "mpm_cfg": mpm_cfg,
        "intrusion": {
            "start_clearance_m": args.start_clearance,
            "depth_m": args.depth,
            "rate_m_s": args.rate,
            "hold_s": args.hold_s,
            "retract": not args.no_retract,
        },
        "env_cfg": {k: env_cfg[k] for k in ("kp", "kd", "effort_limit", "joint_names", "default_joint_angles")},
        "dt": env.dt,
        "substeps": env_cfg["substeps"],
        "metrics": metrics,
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))

    if env.cam is not None:
        env.cam.stop_recording(save_to_filename=str(out_dir / "rollout.mp4"), fps=round(1.0 / env.dt))

    print(f"report written to {out_dir}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def _plot_force_time(out_dir, data):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    for k, comp in zip(("fx", "fy", "fz"), ("F_x", "F_y", "F_z")):
        ax.plot(data["t"], data[k], linewidth=1.6, label=comp)
    _style_axes(ax, "Foot reaction force (world frame)", "time (s)", "force (N)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "force_time.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _plot_depth_time(out_dir, data):
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--", label="mud surface")
    ax.plot(data["t"], data["depth_cmd"], color=INK_MUTED, linewidth=1.2, linestyle=":", label="commanded")
    ax.plot(data["t"], data["depth_measured"], color=SERIES[0], linewidth=1.6, label="measured")
    ax.invert_yaxis()  # depth increases downward
    _style_axes(ax, "Foot depth (below mud surface)", "time (s)", "depth (m)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "depth_time.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _plot_force_depth(out_dir, data, n_descend):
    """The actual pressure-sinkage-style curve: F_z vs. measured depth, loading and
    unloading phases distinguished. The gap between them (if withdrawal was recorded) is
    hysteresis -- energy the mud didn't return -- exactly what would read as suction-like
    drag in a raw time series but is really a loading/unloading path difference.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    depth, fz = data["depth_measured"], data["fz"]
    ax.plot(depth[:n_descend], fz[:n_descend], color=SERIES[0], linewidth=1.8, label="loading (descent)")
    if n_descend < len(depth):
        ax.plot(depth[n_descend:], fz[n_descend:], color=SERIES[2], linewidth=1.8, label="unloading (retraction/hold)")
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    _style_axes(ax, "Foot force vs. depth (pressure-sinkage)", "depth below mud surface (m)", "F_z (N)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "force_depth.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _plot_torque_time(out_dir, data, leg):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    for k, name in zip(("tau_hip", "tau_thigh", "tau_calf"), ("hip", "thigh", "calf")):
        ax.plot(data["t"], data[k], linewidth=1.6, label=name)
    _style_axes(ax, f"{leg} leg motor torque", "time (s)", "torque (N m)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "torque_time.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
