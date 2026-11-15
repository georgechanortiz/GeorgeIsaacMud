"""Genesis env for replaying IsaacLab-trained locomotion policies.

The robot (URDF, default pose, actuators, decimation) comes from env_cfg. Eval only: rewards are
always zero and only the "policy" observation group is produced.
"""

import math
from pathlib import Path

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
GO2_URDF = str(ASSETS_DIR / "go2" / "urdf" / "go2.urdf")


def resolve_urdf(path):
    """env_cfg["urdf"] may be absolute or relative to assets/; unset means the Go2."""
    if path is None:
        return GO2_URDF
    path = Path(path)
    return str(path if path.is_absolute() else ASSETS_DIR / path)


def wrap_to_pi(angle):
    """Fold an angle into [-pi, pi). Matches isaaclab.utils.math.wrap_to_pi."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def dc_motor_torque_limits(joint_vel, saturation_effort, effort_limit, velocity_limit):
    """(lower, upper) torque bounds of IsaacLab's DCMotor at joint velocity `joint_vel`.

    Mirrors isaaclab.actuators.DCMotor._clip_effort: available torque falls off along the
    torque-speed line saturation * (1 -/+ v / velocity_limit), clipped to +/- effort_limit.
    """
    ratio = joint_vel / velocity_limit
    upper = torch.clip(saturation_effort * (1.0 - ratio), min=torch.zeros_like(ratio))
    upper = torch.minimum(upper, effort_limit)
    lower = torch.clip(saturation_effort * (-1.0 - ratio), max=torch.zeros_like(ratio))
    lower = torch.maximum(lower, -effort_limit)
    return lower, upper


def _per_joint(value, joint_names):
    """Expand a scalar or a {joint_name: value} dict into a list in joint_names order."""
    if value is None:
        return None
    if isinstance(value, dict):
        return [float(value[name]) for name in joint_names]
    return [float(value)] * len(joint_names)


class UnitreeEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, record=False):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device
        self.dt = 0.02  # 50 Hz control, as on the robot
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.cfg = self.env_cfg = env_cfg  # rsl-rl's runner reads env.cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]

        # kp/kd/effort_limit/velocity_limit/armature may each be a scalar or a {joint_name: value} dict.
        joint_names = env_cfg["joint_names"]
        self._kd = _per_joint(env_cfg["kd"], joint_names)
        effort_limit = _per_joint(env_cfg.get("effort_limit"), joint_names)
        # IsaacLab's DCMotor limits torque along a torque-speed line. Genesis only has a static
        # force range, so with velocity_limit set it is recomputed before every physics step.
        velocity_limit = _per_joint(env_cfg.get("velocity_limit"), joint_names)
        saturation_effort = _per_joint(env_cfg.get("saturation_effort", env_cfg.get("effort_limit")), joint_names)
        self._dc_motor = velocity_limit is not None
        if self._dc_motor and effort_limit is None:
            raise ValueError("velocity_limit (DCMotor torque-speed cap) needs effort_limit as well")
        armature = env_cfg.get("armature")

        # IsaacLab's decimation: physics steps per control step. substeps is the total per control
        # step, split evenly across them.
        self.decimation = int(env_cfg.get("decimation", 1))
        substeps = env_cfg.get("substeps", 2)
        if substeps % self.decimation:
            raise ValueError(f"substeps ({substeps}) must be a multiple of decimation ({self.decimation})")

        scene_kwargs = self._scene_kwargs(show_viewer)
        if self._dc_motor:
            # The torque-speed cap differs per env, so force_range must be batched. Set here so
            # subclasses that replace rigid_options still get it.
            rigid_options = scene_kwargs.get("rigid_options") or gs.options.RigidOptions()
            rigid_options.batch_dofs_info = True
            scene_kwargs["rigid_options"] = rigid_options
        self.scene = gs.Scene(**scene_kwargs)

        self.plane = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True), material=self._rigid_material()
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=resolve_urdf(env_cfg.get("urdf")),
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
                # A scalar armature is applied here; a per-joint dict is set after build.
                **({"default_armature": armature} if isinstance(armature, (int, float)) else {}),
            ),
            material=self._rigid_material(),
        )

        self._add_extra_entities()

        # Offscreen recording camera; cameras must be added before build().
        self.cam = None
        if record:
            cam_kwargs = self._camera_kwargs()
            self.cam = self.scene.add_camera(GUI=False, **cam_kwargs)
            # update_camera() keeps this offset from the robot as it moves.
            lookat = cam_kwargs["lookat"]
            self._cam_lookat_z = float(lookat[2])
            self._cam_offset = tuple(float(p - l) for p, l in zip(cam_kwargs["pos"], lookat))

        self.scene.build(n_envs=num_envs)

        as_t = lambda v: torch.tensor(v, dtype=gs.tc_float, device=gs.device)
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in joint_names], dtype=gs.tc_int, device=gs.device
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp(_per_joint(env_cfg["kp"], joint_names), self.motors_dof_idx)
        self.robot.set_dofs_kv(self._kd, self.motors_dof_idx)
        if armature is not None:
            self.robot.set_dofs_armature(self._effective_armature(armature, joint_names), self.motors_dof_idx)
        if effort_limit is not None:
            self.robot.set_dofs_force_range([-e for e in effort_limit], effort_limit, self.motors_dof_idx)
        if self._dc_motor:
            self._effort_limit_t = as_t(effort_limit)
            self._saturation_effort_t = as_t(saturation_effort)
            self._velocity_limit_t = as_t(velocity_limit)

        self.global_gravity = as_t([0.0, 0.0, -1.0])

        self.init_base_pos = as_t(env_cfg["base_init_pos"])
        self.init_base_quat = as_t(env_cfg["base_init_quat"])
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        # Genesis orders joints FL,FR,RL,RR; joint_names is FR,FL,RR,RL. init_dof_pos is in Genesis
        # order (for set_qpos); every other per-joint buffer is in joint_names order.
        init_dof_pos = as_t([env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]])
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)
        self.default_dof_pos = as_t([env_cfg["default_joint_angles"][name] for name in joint_names])
        action_scale = env_cfg["action_scale"]
        if isinstance(action_scale, dict):
            action_scale = [action_scale[name] for name in joint_names]
        self.action_scale = as_t(action_scale)

        vec = lambda n: torch.empty((num_envs, n), dtype=gs.tc_float, device=gs.device)
        self.base_pos, self.base_quat, self.base_euler = vec(3), vec(4), vec(3)
        self.base_lin_vel, self.base_ang_vel, self.projected_gravity = vec(3), vec(3), vec(3)
        self.actions = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.rew_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((num_envs,), dtype=gs.tc_int, device=gs.device)
        self.extras = dict()

        self.commands = vec(command_cfg["num_commands"])
        self.commands_scale = as_t(self.obs_scales["commands"])
        self.commands_limits = tuple(
            as_t(values)
            for values in zip(command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"], command_cfg["ang_vel_range"])
        )

        # Heading controller from IsaacLab's UniformVelocityCommand. The policies were trained with
        # it: the yaw command is stiffness * wrap_to_pi(heading_target - yaw), recomputed each step.
        self.heading_command = bool(command_cfg.get("heading_command", False))
        self.heading_stiffness = float(command_cfg.get("heading_control_stiffness", 0.5))
        self._heading_range = tuple(float(v) for v in command_cfg.get("heading_range", (0.0, 0.0)))
        self.heading_target = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        if self.heading_command and self.commands_limits[0][2] == self.commands_limits[1][2]:
            print(
                f"[WARN] heading_command is on but ang_vel_range is degenerate "
                f"({command_cfg['ang_vel_range']}); it is the clip on the generated yaw "
                "correction, so the heading hold cannot act. Set it to the trained range."
            )

        self.reset()

    def _effective_armature(self, armature, joint_names):
        """Armature to give Genesis so the PD-driven joints respond as in training.

        Genesis' implicit integrator adds kd * substep_dt to the mass-matrix diagonal of every
        PD-driven dof; IsaacLab's explicit DCMotor adds nothing. With
        env_cfg["compensate_implicit_pd"] set, the armature is reduced by that amount.
        """
        arm = _per_joint(armature, joint_names)
        if not self.env_cfg.get("compensate_implicit_pd", False):
            return arm
        h = self.dt / self.env_cfg.get("substeps", 2)
        out = []
        for name, a, kd in zip(joint_names, arm, self._kd):
            eff = a - kd * h
            if eff < 0.0:
                print(
                    f"[WARN] {name}: armature {a} is smaller than the implicit PD term kd*dt = {kd * h:.4f}; "
                    "clamping to 0. Raise substeps to replay this actuator faithfully."
                )
                eff = 0.0
            out.append(eff)
        return out

    def _apply_dc_motor_limits(self):
        """Set this physics step's force range from the DCMotor torque-speed curve."""
        lower, upper = dc_motor_torque_limits(
            self.robot.get_dofs_velocity(self.motors_dof_idx),
            self._saturation_effort_t,
            self._effort_limit_t,
            self._velocity_limit_t,
        )
        self.robot.set_dofs_force_range(lower, upper, self.motors_dof_idx)

    # ------------ scene hooks (overridden by subclasses) ------------
    def _scene_kwargs(self, show_viewer):
        """Constructor kwargs for gs.Scene. Override to add mpm_options."""
        return dict(
            sim_options=gs.options.SimOptions(
                dt=self.dt / self.decimation,
                substeps=self.env_cfg.get("substeps", 2) // self.decimation,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(camera_pos=(2.0, 0.0, 2.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=40),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

    def _rigid_material(self):
        """Material for the plane and robot. None uses the genesis default Rigid."""
        return None

    def _camera_kwargs(self):
        """Offscreen recording camera pose. Override to match a subclass's viewer angle."""
        return dict(res=(960, 540), pos=(2.0, -2.2, 1.3), lookat=(0.0, 0.0, 0.3), fov=40)

    def _add_extra_entities(self):
        """Hook called after the robot is added and before the scene is built."""

    def update_camera(self):
        """Move the recording camera to follow env 0. No-op when not recording."""
        if self.cam is None:
            return
        lx, ly = float(self.base_pos[0, 0]), float(self.base_pos[0, 1])
        pos = (lx + self._cam_offset[0], ly + self._cam_offset[1], self._cam_lookat_z + self._cam_offset[2])
        self.cam.set_pose(pos=pos, lookat=(lx, ly, self._cam_lookat_z))

    # ------------ commands ------------
    def _resample_commands(self, envs_idx):
        lower, upper = self.commands_limits
        commands = (upper - lower) * torch.rand((self.num_envs, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def _resample_heading(self, envs_idx):
        """Draw a new heading target, once per episode."""
        if not self.heading_command:
            return
        target = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device).uniform_(*self._heading_range)
        if envs_idx is None:
            self.heading_target.copy_(target)
        else:
            torch.where(envs_idx, target, self.heading_target, out=self.heading_target)

    def _update_heading_command(self):
        """Overwrite the yaw command with the heading controller's output.

        Must run after resets (so it uses the new pose) and before the observation is built.
        """
        if not self.heading_command:
            return
        yaw = quat_to_xyz(self.base_quat, rpy=True, degrees=False)[:, 2]
        correction = self.heading_stiffness * wrap_to_pi(self.heading_target - yaw)
        self.commands[:, 2] = torch.clip(correction, self.commands_limits[0][2], self.commands_limits[1][2])

    # ------------ stepping ------------
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        for _ in range(self.decimation):
            if self._dc_motor:
                self._apply_dc_motor_limits()
            self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # reset on timeout, a fall, or a solver error
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        self._reset_idx(self.reset_buf)

        self._update_heading_command()
        self._update_observation()
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)
            for buf in (self.base_lin_vel, self.base_ang_vel, self.dof_vel, self.actions, self.episode_length_buf):
                buf.zero_()
            self.reset_buf.fill_(True)
        else:
            mask = envs_idx[:, None]
            torch.where(mask, self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(mask, self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(mask, self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(mask, self.default_dof_pos, self.dof_pos, out=self.dof_pos)
            for buf in (self.base_lin_vel, self.base_ang_vel, self.dof_vel, self.actions):
                buf.masked_fill_(mask, 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        self._resample_commands(envs_idx)
        self._resample_heading(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_heading_command()
        self._update_observation()
        return self.get_observations()


_A2_CAM_POS = (2.2, -2.6, 1.6)
_A2_CAM_LOOKAT = (0.0, 0.0, 0.3)


class A2Env(UnitreeEnv):
    """UnitreeEnv with the camera pulled back for the larger A2."""

    def _scene_kwargs(self, show_viewer):
        kwargs = super()._scene_kwargs(show_viewer)
        kwargs["viewer_options"] = gs.options.ViewerOptions(camera_pos=_A2_CAM_POS, camera_lookat=_A2_CAM_LOOKAT, camera_fov=40)
        return kwargs

    def _camera_kwargs(self):
        return dict(res=(1280, 720), pos=_A2_CAM_POS, lookat=_A2_CAM_LOOKAT, fov=40)
