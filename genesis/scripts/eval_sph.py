"""Replay the Go2 locomotion policy wading through an SPH water pool.

Same policy and observation pipeline as scripts/eval.py -- the only difference is
that the scene carries an SPH solver and the robot walks through standing liquid
instead of over bare ground.

    python scripts/eval_sph.py -p logs/2026-07-15_17-09-30/model_4999.pt
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


class Go2SPHEnv(Go2Env):
    """Go2Env plus a pool of SPH liquid lying in the robot's path."""

    def __init__(self, *args, sph_cfg, **kwargs):
        self.sph_cfg = sph_cfg
        super().__init__(*args, **kwargs)

    def _scene_kwargs(self, show_viewer):
        kwargs = super()._scene_kwargs(show_viewer)
        kwargs["sph_options"] = gs.options.SPHOptions(
            lower_bound=self.sph_cfg["bounds_lower"],
            upper_bound=self.sph_cfg["bounds_upper"],
            particle_size=self.sph_cfg["particle_size"],
        )
        kwargs["vis_options"] = gs.options.VisOptions(
            visualize_sph_boundary=True,
            rendered_envs_idx=[0],
        )
        kwargs["viewer_options"] = gs.options.ViewerOptions(
            camera_pos=(2.6, -2.0, 1.3),
            camera_lookat=(0.8, 0.0, 0.2),
            camera_fov=40,
        )
        return kwargs

    def _rigid_material(self):
        # needs_coup is already the default, but coupling is the point of this demo.
        return gs.materials.Rigid(needs_coup=True, coup_friction=self.sph_cfg["coup_friction"])

    def _camera_kwargs(self):
        # match the viewer angle set in _scene_kwargs
        return dict(res=(1280, 720), pos=(2.6, -2.0, 1.3), lookat=(0.8, 0.0, 0.2), fov=40)

    def _pool_box(self):
        """Resolve pool_lower/pool_upper into the (pos, size) that gs.morphs.Box wants.

        The SPH solver rejects any entity with particles outside its domain, but the
        message does not say which axis overflowed, so check it here first.
        """
        lower = np.asarray(self.sph_cfg["pool_lower"], dtype=float)
        upper = np.asarray(self.sph_cfg["pool_upper"], dtype=float)
        bounds_lower = np.asarray(self.sph_cfg["bounds_lower"], dtype=float)
        bounds_upper = np.asarray(self.sph_cfg["bounds_upper"], dtype=float)

        for axis, name in enumerate("xyz"):
            if upper[axis] <= lower[axis]:
                raise ValueError(
                    f"pool_upper must exceed pool_lower on {name}: "
                    f"got {lower[axis]:.3f} .. {upper[axis]:.3f}"
                )
            if lower[axis] < bounds_lower[axis] or upper[axis] > bounds_upper[axis]:
                raise ValueError(
                    f"pool box overflows the SPH domain on {name}: pool spans "
                    f"[{lower[axis]:.3f}, {upper[axis]:.3f}] but bounds are "
                    f"[{bounds_lower[axis]:.3f}, {bounds_upper[axis]:.3f}]. "
                    f"Widen bounds_{'lower' if lower[axis] < bounds_lower[axis] else 'upper'} "
                    f"or shrink the pool."
                )

        pos = tuple(float(v) for v in (lower + upper) / 2)
        size = tuple(float(v) for v in (upper - lower))
        return pos, size

    def _add_extra_entities(self):
        pos, size = self._pool_box()
        self.water = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(
                rho=self.sph_cfg["rho"],
                mu=self.sph_cfg["mu"],
                sampler="regular",
            ),
            morph=gs.morphs.Box(pos=pos, size=size),
            surface=gs.surfaces.Default(color=(0.35, 0.6, 0.9, 1.0), vis_mode="particle"),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        default=DEFAULT_POLICY,
        help="path to a model_*.pt checkpoint, absolute or relative to the repo root",
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU (much slower for SPH)")
    parser.add_argument("--substeps", type=int, default=20, help="SPH substep target is ~1e-3 s")
    parser.add_argument("--particle_size", type=float, default=0.02, help="smaller = more particles, costs speed")
    # parser.add_argument("--depth", type=float, default=0.12, help="pool depth in metres")
    parser.add_argument("--mu", type=float, default=0.01, help="viscosity")
    parser.add_argument("--vx", type=float, default=0.5, help="commanded forward velocity")
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

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    ckpt_path = resolve_policy(args.policy)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs(ckpt_path.parent)
    reward_cfg["reward_scales"] = {}

    # SPH is only stable with a much smaller substep than rigid-only locomotion needs.
    env_cfg["substeps"] = args.substeps
    # Drive straight into the pool.
    command_cfg["lin_vel_x_range"] = [args.vx, args.vx]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    if not command_cfg.get("heading_command"):
        # No heading controller to hold the line, so an open-loop zero is the best available.
        # With one configured, ang_vel_range is the *clip* on its correction rather than the
        # command -- zeroing it here would clamp the correction away and let the walk curve.
        command_cfg["ang_vel_range"] = [0.0, 0.0]

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
        show_viewer=True,
        record=args.reporting,
        sph_cfg=sph_cfg,
    )
    print(f"SPH particles: {env.water.n_particles}")

    policy = load_policy(env, ckpt_path, train_cfg, phi_smooth=args.phi_smooth, phi_clamp=args.phi_clamp, phi_rate_hz=args.phi_rate)

    reporter = None
    if args.reporting:
        reporter = Reporter(
            experiment=ckpt_path.parent.parent.name,
            params={
                "policy": str(ckpt_path),
                "run": ckpt_path.parent.name,
                "scene": "sph",
                "sph_cfg": sph_cfg,
                "sph_particles": int(env.water.n_particles),
                "num_envs": 1,
                "dt": env.dt,
                "substeps": env_cfg["substeps"],
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
