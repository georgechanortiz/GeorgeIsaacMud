"""UnitreeEnv with an MPM mud pit in the robot's path (the pit is built by envs/mud.py)."""

import genesis as gs

from envs import mud
from envs.unitree_env import UnitreeEnv

_CAM_POS = (3.1, -2.6, 1.6)
_CAM_LOOKAT = (0.9, 0.0, 0.3)


class A2MPMEnv(mud.MudFootForces, UnitreeEnv):
    """UnitreeEnv plus the walled MPM mud described by `mpm_cfg`, with the camera framed for the A2."""

    def __init__(self, *args, mpm_cfg, **kwargs):
        self.mpm_cfg = mpm_cfg
        super().__init__(*args, **kwargs)

    def _scene_kwargs(self, show_viewer):
        kwargs = super()._scene_kwargs(show_viewer)
        kwargs["mpm_options"] = mud.mpm_options(self.mpm_cfg)
        # the walls add collision pairs
        kwargs["rigid_options"] = gs.options.RigidOptions(
            enable_self_collision=False, tolerance=1e-5, max_collision_pairs=100
        )
        kwargs["vis_options"] = gs.options.VisOptions(visualize_mpm_boundary=True, rendered_envs_idx=[0])
        kwargs["viewer_options"] = gs.options.ViewerOptions(camera_pos=_CAM_POS, camera_lookat=_CAM_LOOKAT, camera_fov=40)
        return kwargs

    def _rigid_material(self):
        return mud.coupling_material(self.mpm_cfg)

    def _camera_kwargs(self):
        return dict(res=(1280, 720), pos=_CAM_POS, lookat=_CAM_LOOKAT, fov=40)

    def _add_extra_entities(self):
        self.mud_patches = mud.add_mud_patches(self.scene, self.mpm_cfg)
        self.walls = mud.add_containment_walls(self.scene, self.mpm_cfg)

    def particle_counts(self):
        return mud.particle_counts(self.mpm_cfg, self.mud_patches)
