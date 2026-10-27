"""Sphere drop bench: a rigid sphere dropped into a small MPM sand pit.

For validating a reduced-order mud reaction force model against Genesis's MPM ground truth
with the simplest possible intruder. The pit is 2 ft x 2 ft and 6 in deep (envs/mud.py:
drop_pit), filled with one layer of the sand the earlier leg-intrusion bench used, at
the same grid density and so the same particle size. A free 6-DOF
sphere is held above it while the sand settles under gravity, then released and tracked:
position, velocity, penetration below the settled surface, and the force the sand puts on it.

The drop height is measured from the rigid floor (z = 0, the bottom of the pit) to the
sphere's centre, so with the default 6 in fill the sphere's bottom clears the sand by
drop_height - radius - 0.1524 m.

Every parameter (sphere, pit, sand, sim) lives in a YAML config, configs/sphere_drop.yaml
by default; a copy of the one used is saved with each run.

    python scripts/eval_mpm_1d_intrusion.py [--config configs/sphere_drop.yaml]
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs

from envs import mud
from scripts.unitree_policy import plots

GRAVITY = 9.81
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sphere_drop.yaml"

# The keys each config section must have -- exactly these, so a typo fails loudly instead of
# silently falling back to nothing.
CONFIG_KEYS = {
    "sphere": {"radius", "mass", "drop_height", "drop_xy", "init_vel", "init_ang_vel", "coup_friction",
               "coup_softness", "coup_restitution", "friction", "sdf_cell_size"},
    "pit": {"side", "depth"},
    "sand": {"grid_density", "E", "nu", "rho", "friction_angle", "settle_s", "surface_percentile"},
    "sim": {"duration", "dt", "substeps", "cpu", "headless", "video"},
}


class SphereDropEnv:
    """The drop pit, a rigid floor under it, and one free sphere."""

    def __init__(self, mpm_cfg, sphere_cfg, dt, substeps, show_viewer=False, record=False, cam_kwargs=None):
        self.mpm_cfg = mpm_cfg
        self.dt = dt

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
            mpm_options=mud.mpm_options(mpm_cfg),
            rigid_options=gs.options.RigidOptions(tolerance=1e-5, max_collision_pairs=100),
            vis_options=gs.options.VisOptions(visualize_mpm_boundary=True, rendered_envs_idx=[0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.9, -1.1, 0.7), camera_lookat=(0.0, 0.0, 0.1), camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        self.plane = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True), material=mud.coupling_material(mpm_cfg)
        )

        # Mass is set through density: a solid sphere of this radius and rho weighs exactly `mass`,
        # and gets the matching solid-sphere inertia (2/5 m r^2).
        r = sphere_cfg["radius"]
        rho = sphere_cfg["mass"] / (4.0 / 3.0 * math.pi * r**3)
        self.sphere = self.scene.add_entity(
            gs.morphs.Sphere(radius=r, pos=sphere_cfg["pos"]),
            material=gs.materials.Rigid(
                rho=rho,
                friction=sphere_cfg["friction"],
                needs_coup=True,
                coup_friction=sphere_cfg["coup_friction"],
                coup_softness=sphere_cfg["coup_softness"],
                coup_restitution=sphere_cfg["coup_restitution"],
                sdf_cell_size=sphere_cfg["sdf_cell_size"],
            ),
            surface=gs.surfaces.Default(color=(0.75, 0.75, 0.78)),
        )

        self.mud_patches = mud.add_mud_patches(self.scene, mpm_cfg)
        self.walls = mud.add_containment_walls(self.scene, mpm_cfg)

        self.cam = None
        if record:
            self.cam = self.scene.add_camera(GUI=False, **cam_kwargs)

        self.scene.build(n_envs=1)
        if self.cam is not None:
            self.cam.start_recording()

    def hold(self, pos):
        """Pin the sphere at `pos`, upright and at rest (used while the sand settles)."""
        p = torch.tensor([pos], dtype=gs.tc_float, device=gs.device)
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=gs.tc_float, device=gs.device)
        self.sphere.set_pos(p, zero_velocity=True)
        self.sphere.set_quat(q, zero_velocity=True)

    def sphere_force(self):
        """Force the sand exerts on the sphere, world frame, shape (3,)."""
        return mud.coupling_force(self.scene, [self.sphere.base_link_idx])[0, 0]


def measure_surface(env, xy, percentile, radius):
    """Free-surface height of the settled sand in a column of half-width `radius` around `xy`.

    Local to the impact point rather than pit-wide: the sand piles against the domain
    boundary at the edges, and the surface that matters is the one the sphere lands on.
    """
    cols = []
    for entity in env.mud_patches:
        pos = entity.get_particles_pos()
        pos = pos[0] if pos.ndim == 3 else pos  # (n_particles, 3), env 0
        pos = np.asarray(pos.detach().cpu() if hasattr(pos, "detach") else pos)
        near = (np.abs(pos[:, 0] - xy[0]) < radius) & (np.abs(pos[:, 1] - xy[1]) < radius)
        if near.any():
            cols.append(pos[near, 2])
    if not cols:
        raise SystemExit(f"no MPM particles within {radius} m of ({xy[0]:.3f}, {xy[1]:.3f}) -- is it over the pit?")
    return float(np.percentile(np.concatenate(cols), percentile))


def load_config(path):
    """Read and validate a sphere drop config (see configs/sphere_drop.yaml)."""
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != set(CONFIG_KEYS):
        raise SystemExit(f"{path}: sections must be exactly {sorted(CONFIG_KEYS)}, got {sorted(cfg)}")
    for section, keys in CONFIG_KEYS.items():
        missing, unknown = keys - set(cfg[section]), set(cfg[section]) - keys
        if missing or unknown:
            raise SystemExit(f"{path} [{section}]: missing {sorted(missing)}, unknown {sorted(unknown)}")

    sph, pit = cfg["sphere"], cfg["pit"]
    if sph["radius"] <= 0 or sph["mass"] <= 0:
        raise SystemExit(f"{path}: sphere radius and mass must be positive")
    if sph["drop_height"] - sph["radius"] <= pit["depth"]:
        raise SystemExit(
            f"{path}: sphere starts in the sand: bottom at {sph['drop_height'] - sph['radius']:.3f} m, sand top "
            f"at {pit['depth']:.3f} m. Raise sphere.drop_height."
        )
    if max(abs(v) for v in sph["drop_xy"]) + sph["radius"] > pit["side"] / 2.0:
        raise SystemExit(f"{path}: sphere.drop_xy puts the sphere over the pit edge (half-width {pit['side'] / 2:.3f} m)")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML config, absolute or relative to the working directory")
    config_path = parser.parse_args().config
    cfg = load_config(config_path)
    sph, pit, sand, sim = cfg["sphere"], cfg["pit"], cfg["sand"], cfg["sim"]

    gs.init(backend=gs.cpu if sim["cpu"] else gs.gpu, precision="32", logging_level="warning")

    layer = {"name": "sand", "material": "sand",
             **{k: sand[k] for k in ("E", "nu", "rho", "friction_angle")}}
    mpm_cfg = mud.drop_pit(layer, side=pit["side"], depth=pit["depth"], grid_density=sand["grid_density"])
    mpm_cfg["coup_friction"] = sph["coup_friction"]  # the floor's coupling, matched to the sphere's
    mpm_cfg["coup_softness"] = sph["coup_softness"]

    release_pos = (sph["drop_xy"][0], sph["drop_xy"][1], sph["drop_height"])
    sphere_cfg = {**sph, "pos": release_pos}
    cam_kwargs = dict(
        res=(1280, 720),
        pos=(release_pos[0] + 0.55, release_pos[1] - 0.7, pit["depth"] + 0.35),
        lookat=(release_pos[0], release_pos[1], pit["depth"]),
        fov=40,
    )
    env = SphereDropEnv(
        mpm_cfg, sphere_cfg, sim["dt"], sim["substeps"],
        show_viewer=not sim["headless"], record=sim["video"], cam_kwargs=cam_kwargs,
    )
    print(f"MPM particles: {mud.particle_counts(mpm_cfg, env.mud_patches)}")
    print(f"sphere: r {sph['radius']:.3f} m, mass {float(env.sphere.get_mass()):.3f} kg")

    # Render at ~50 fps whatever --dt is, so the video plays at wall-clock speed without
    # rendering every fine step.
    render_every = max(1, round(0.02 / sim["dt"]))
    step_i = 0

    def step():
        nonlocal step_i
        env.scene.step()
        if env.cam is not None and step_i % render_every == 0:
            env.cam.render()
        step_i += 1

    # Settle the sand with the sphere pinned above it, then take the datum from where the sand ended up.
    surface = pit["depth"]
    n_settle = max(0, round(sand["settle_s"] / sim["dt"]))
    for _ in range(n_settle):
        env.hold(release_pos)
        step()
    if n_settle:
        surface = measure_surface(env, release_pos[:2], sand["surface_percentile"], radius=max(sph["radius"], 0.05))
        print(f"sand settled {sand['settle_s']:.2f}s: surface {pit['depth']:.4f} -> {surface:.4f} m "
              f"({(surface - pit['depth']) * 1e3:+.1f} mm)")

    # Release.
    env.hold(release_pos)
    vel0 = torch.tensor([[*sph["init_vel"], *sph["init_ang_vel"]]], dtype=gs.tc_float, device=gs.device)
    env.sphere.set_dofs_velocity(vel0)

    # Free-fall prediction of the speed at first contact, to check the sim against.
    fall = sph["drop_height"] - sph["radius"] - surface
    v_impact_pred = math.sqrt(sph["init_vel"][2] ** 2 + 2.0 * GRAVITY * fall)

    keys = ("t", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "fx", "fy", "fz", "depth")
    rows = {k: [] for k in keys}
    n_steps = max(1, round(sim["duration"] / sim["dt"]))
    for i in range(n_steps):
        step()
        pos = env.sphere.get_pos()[0].detach().cpu().numpy()
        vel = env.sphere.get_vel()[0].detach().cpu().numpy()
        ang = env.sphere.get_ang()[0].detach().cpu().numpy()
        force = env.sphere_force()
        for k, v in zip(keys, (
            (i + 1) * sim["dt"], *pos, *vel, *ang, *force, surface - (pos[2] - sph["radius"]),
        )):
            rows[k].append(float(v))

    data = {k: np.asarray(v) for k, v in rows.items()}

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPO_ROOT / "logs_for_1d" / f"{stamp}_sphere_m{sph['mass']:g}_r{sph['radius']:g}_h{sph['drop_height']:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plots.plot_drop_force_time(out_dir, data)
    plots.plot_drop_depth_time(out_dir, data)
    plots.plot_drop_force_depth(out_dir, data)
    np.savez(out_dir / "drop_data.npz", **data)

    contact = data["depth"] > 0.0
    metrics = {
        "impact_speed_pred_m_s": v_impact_pred,
        "max_depth_m": float(data["depth"].max()),
        "final_depth_m": float(data["depth"][-1]),
        "peak_fz_n": float(data["fz"].max()),
        "t_peak_fz_s": float(data["t"][np.argmax(data["fz"])]),
        "weight_n": sph["mass"] * GRAVITY,
    }
    if contact.any():
        first = int(np.argmax(contact))
        metrics["t_contact_s"] = float(data["t"][first])
        metrics["impact_speed_sim_m_s"] = float(-data["vz"][first])
    else:
        print("warning: the sphere never reached the sand surface -- lengthen sim.duration")

    # The config as run, so the run can be reproduced from its own folder.
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    params = {
        "config": str(Path(config_path).resolve()),
        "settled_surface_m": surface,
        "mpm_cfg": mpm_cfg,
        "metrics": metrics,
    }

    if env.cam is not None:
        fps = round(1.0 / (sim["dt"] * render_every))
        env.cam.stop_recording(save_to_filename=str(out_dir / "rollout.mp4"), fps=fps)
        params["video"] = {"file": "rollout.mp4", "fps": fps, "includes_settle_phase": True}

    (out_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))

    print(f"report written to {out_dir}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
