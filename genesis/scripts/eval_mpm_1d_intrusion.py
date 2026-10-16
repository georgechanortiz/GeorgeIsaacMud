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

It does not subclass Go2Env/Go2MPMEnv (envs/go2_env.py, envs/mpm_env.py). Those are built
around a floating-base robot driven by a trained locomotion policy's 12-dof position actions
each control step, with a full observation/reward/command/termination pipeline behind them --
none of which applies to a bench test with a welded base, no policy, and one leg under direct
IK control. This script owns its own minimal scene, built from the same mud building blocks
(envs/mud.py) as the walking evals.

The pit is the Go2 pit (envs/mud.py: go2_mud_pit) with a different soft layer: sand from 6.5 to
15 cm (INTRUSION_SOFT_LAYER below) where scripts/eval_mpm.py uses elasto-plastic mud from 6 to
11 cm. Earlier results from this bench used this sand layer, so it is kept.

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
from genesis.utils.geom import transform_by_quat

from envs import mud
from envs.go2_env import GO2_URDF
from scripts.unitree_policy import plots
from scripts.unitree_policy.eval_cfg import load_cfgs, resolve_policy

# Every *_foot link is a fixed child of its *_calf link, at this local offset in all four
# legs (confirmed in the URDF) -- and Genesis's URDF loader collapses fixed-joint children
# into their parent at build time, so no separate "*_foot" RigidLink exists to query
# directly. The foot's world position is therefore computed by hand each step:
# calf_pos + R(calf_quat) @ FOOT_LOCAL_OFFSET. get_foot_forces() (envs/mud.py) already reads
# reaction force off the *_calf link for the same reason.
FOOT_LOCAL_OFFSET = (0.0, 0.0, -0.213)
# The foot's collision geometry is a sphere of this radius; "depth into the mud" is
# measured at its bottom -- the actual contact surface -- not the offset point above.
FOOT_SPHERE_RADIUS = 0.022

# Thigh and calf links are each 0.213 m, so a fully-extended leg reaches this far below
# the hip. Used to check up front that the commanded depth is actually reachable.
MAX_LEG_REACH = 2 * 0.213

LEGS = ("FR", "FL", "RR", "RL")

# Sand top layer of the bench pit; see the module docstring.
INTRUSION_SOFT_LAYER = mud.soft_layer(
    "sand", (0.065, 0.15), E=2e4, nu=0.35, rho=1500.0, friction_angle=30.0
)


class Go2MPM1DIntrusionEnv(mud.MudFootForces):
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
            mpm_options=mud.mpm_options(mpm_cfg),
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

        rigid_material = mud.coupling_material(mpm_cfg)

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

        self.mud_patches = mud.add_mud_patches(self.scene, mpm_cfg)
        self.walls = mud.add_containment_walls(self.scene, mpm_cfg)

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
    parser.add_argument("--no-video", action="store_true",
                         help="skip rollout.mp4. Video is on by default (as in eval_mpm.py --reporting) and "
                              "covers the mud-settle and leg-settle phases as well as the ramp, so the "
                              "approach and first contact are visible, not just the recorded window.")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    ckpt_path = resolve_policy(args.policy)
    env_cfg, _, _, _, _ = load_cfgs(ckpt_path.parent)
    env_cfg["substeps"] = args.substeps
    if args.kp is not None:
        env_cfg["kp"] = args.kp
    if args.effort_limit is not None:
        env_cfg["effort_limit"] = args.effort_limit

    mpm_cfg = mud.go2_mud_pit(INTRUSION_SOFT_LAYER, grid_density=args.grid_density)
    slab_top = mud.surface_height(mpm_cfg)
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

    # Provisional framing; re-aimed at the foot once the scene is built and its actual
    # position is known (the camera has to exist before build()).
    cam_kwargs = dict(res=(1280, 720), pos=(cx + 0.6, -1.0, slab_top + 0.6), lookat=(cx, 0.0, slab_top), fov=40)
    env = Go2MPM1DIntrusionEnv(
        env_cfg, mpm_cfg, bench_dof_pos,
        show_viewer=not args.headless, record=not args.no_video, cam_kwargs=cam_kwargs,
    )
    patch_counts = mud.particle_counts(mpm_cfg, env.mud_patches)
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

    # Re-frame on the foot now that its position is known: this test is about a few
    # centimetres of penetration at one point, so a body-centred shot would waste the frame.
    if env.cam is not None:
        env.cam.set_pose(
            pos=(foot_xy[0] + 0.35, foot_xy[1] - 0.45, slab_top + 0.22),
            lookat=(foot_xy[0], foot_xy[1], slab_top),
        )

    # Let the mud reach gravitational equilibrium, then take the depth datum from where the
    # material actually ended up rather than the nominal slab top it was spawned at.
    n_mud_settle = max(0, round(args.mud_settle_s / env.dt))
    for _ in range(n_mud_settle):
        env.robot.control_dofs_position(env.bench_dof_pos.unsqueeze(0), env.motors_dof_idx)
        env.scene.step()
        if env.cam is not None:
            env.cam.render()
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
            if env.cam is not None:
                env.cam.render()
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

    plots.plot_intrusion_force_time(out_dir, data)
    plots.plot_intrusion_depth_time(out_dir, data)
    plots.plot_intrusion_force_depth(out_dir, data, n_descend=int(np.argmax(data["depth_cmd"] >= args.depth - 1e-9)) + 1)
    plots.plot_intrusion_torque_time(out_dir, data, leg)

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

    if env.cam is not None:
        # fps = control rate, so the video plays at wall-clock speed
        fps = round(1.0 / env.dt)
        env.cam.stop_recording(save_to_filename=str(out_dir / "rollout.mp4"), fps=fps)
        params["video"] = {"file": "rollout.mp4", "fps": fps, "includes_settle_phases": True}

    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))

    print(f"report written to {out_dir}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
