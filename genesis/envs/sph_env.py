"""Go2Env with a pool of SPH liquid in the robot's path."""

import numpy as np

import genesis as gs

from envs.go2_env import Go2Env


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
        message does not say which axis overflowed, so check it here first. Unlike MPM
        (envs/mud.py), SPH uses its bounds as-is, with no inset.
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
