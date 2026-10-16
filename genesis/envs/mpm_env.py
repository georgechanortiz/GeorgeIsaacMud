"""Go2Env with an MPM mud pit in the robot's path (see envs/mud.py for the pit itself)."""

import genesis as gs

from envs import mud
from envs.go2_env import Go2Env


class Go2MPMEnv(mud.MudFootForces, Go2Env):
    """Go2Env plus the patches of MPM material described by `mpm_cfg`, walled in."""

    def __init__(self, *args, mpm_cfg, **kwargs):
        self.mpm_cfg = mpm_cfg
        super().__init__(*args, **kwargs)

    def _scene_kwargs(self, show_viewer):
        kwargs = super()._scene_kwargs(show_viewer)
        kwargs["mpm_options"] = mud.mpm_options(self.mpm_cfg)
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
        return mud.coupling_material(self.mpm_cfg)

    def _camera_kwargs(self):
        # match the viewer angle set in _scene_kwargs
        return dict(res=(1280, 720), pos=(2.6, -2.0, 1.3), lookat=(0.8, 0.0, 0.2), fov=40)

    def _add_extra_entities(self):
        # self.mud_patches keeps the patches in config order for reporting / introspection.
        self.mud_patches = mud.add_mud_patches(self.scene, self.mpm_cfg)
        self.walls = mud.add_containment_walls(self.scene, self.mpm_cfg)

    def particle_counts(self):
        return mud.particle_counts(self.mpm_cfg, self.mud_patches)


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
