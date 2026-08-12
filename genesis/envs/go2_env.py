import math
from pathlib import Path

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

# Robot meshes/urdf live in this repo, not in the genesis-world install.
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
GO2_URDF = str(ASSETS_DIR / "go2" / "urdf" / "go2.urdf")


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


def wrap_to_pi(angle):
    """Fold an angle into [-pi, pi). Matches isaaclab.utils.math.wrap_to_pi."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, record=False):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]  # 1 step latency on the real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(**self._scene_kwargs(show_viewer))

        # add plain
        self.plane = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            ),
            material=self._rigid_material(),
        )

        # add robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=GO2_URDF,
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
            material=self._rigid_material(),
        )

        # subclasses add particle/deformable entities here, while the scene is still unbuilt
        self._add_extra_entities()

        # Optional offscreen camera for video capture. Must be added before build(),
        # so it is created here rather than by the Reporter (which exists only post-build).
        self.cam = None
        if record:
            cam_kwargs = self._camera_kwargs()
            self.cam = self.scene.add_camera(GUI=False, **cam_kwargs)
            # Keep the initial framing but let update_camera() slide it along with the
            # robot: offset is camera position relative to its look-at target.
            lookat = cam_kwargs["lookat"]
            self._cam_lookat_z = float(lookat[2])
            self._cam_offset = tuple(float(p - l) for p, l in zip(cam_kwargs["pos"], lookat))

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Without a force range the PD controller can deliver torques the real actuator
        # (and the training sim's actuator model) could never produce.
        effort_limit = self.env_cfg.get("effort_limit")
        if effort_limit is not None:
            self.robot.set_dofs_force_range(
                [-effort_limit] * self.num_actions, [effort_limit] * self.num_actions, self.motors_dof_idx
            )

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        # NOTE: genesis orders joints FL,FR,RL,RR while env_cfg["joint_names"] is FR,FL,RR,RL.
        # init_dof_pos is in genesis order because set_qpos expects that order; every other
        # per-joint buffer (dof_pos, dof_vel, actions, default_dof_pos) is in joint_names order.
        # Do not mix the two.
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # initialize buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        # Command scaling is independent of obs_scales["ang_vel"], which applies to base_ang_vel.
        # Falls back to the legacy [lin_vel, lin_vel, ang_vel] behaviour when unset.
        commands_scale = self.obs_scales.get(
            "commands", (self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"])
        )
        self.commands_scale = torch.as_tensor(commands_scale, device=gs.device, dtype=gs.tc_float)
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )

        # Heading controller, ported from IsaacLab's UniformVelocityCommand. Not optional for
        # faithful replay: every go2_mud run was trained with `heading_command: true`, so the yaw
        # command the policy saw was never a constant -- it was
        # `stiffness * wrap_to_pi(heading_target - yaw)`, regenerated every step. Replaying with a
        # pinned ang_vel hands it a dead channel instead, and since nothing else in this env closes
        # a loop on yaw, any small bias integrates into a curved path and stays there.
        self.heading_command = bool(command_cfg.get("heading_command", False))
        self.heading_stiffness = float(command_cfg.get("heading_control_stiffness", 0.5))
        self._heading_range = tuple(float(v) for v in command_cfg.get("heading_range", (0.0, 0.0)))
        self.heading_target = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        if self.heading_command and self.commands_limits[0][2] == self.commands_limits[1][2]:
            # The clip would pin every correction to that one value, silently reducing this back to
            # the open-loop case the heading controller exists to avoid.
            print(
                f"[WARN] heading_command is on but ang_vel_range is degenerate "
                f"({self.command_cfg['ang_vel_range']}); it is the clip on the generated yaw "
                "correction, so the heading hold cannot act. Set it to the trained range."
            )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # action_scale is either a scalar or a per-joint dict keyed by joint name
        action_scale = self.env_cfg["action_scale"]
        if isinstance(action_scale, dict):
            action_scale = [action_scale[name] for name in self.env_cfg["joint_names"]]
        self.action_scale = torch.as_tensor(action_scale, dtype=gs.tc_float, device=gs.device)

        self.extras = dict()  # extra information for logging

        # Optional extra observation groups, needed by RMA / privileged-obs actors.
        # obs_groups mirrors the runner cfg: {"actor": [...], "critic": [...],
        # "encoder": [...], "history": [...]}. Absent it, only "policy" is produced and
        # the env behaves exactly as before.
        self.obs_groups = obs_cfg.get("obs_groups") or {"actor": ["policy"], "critic": ["policy"]}
        self.history_length = int(obs_cfg.get("history_length", 0))
        self.privileged_dim = int(obs_cfg.get("privileged_dim", 12))
        self.num_feet = int(obs_cfg.get("num_feet", 4))
        self.mud_force_scale = float(obs_cfg.get("mud_force_scale", 0.1))
        self._group_names = {role: names[0] for role, names in self.obs_groups.items() if names}
        self._needs_critic = self._group_names.get("critic", "policy") != "policy"
        self._needs_encoder = "encoder" in self._group_names
        self._needs_history = "history" in self._group_names and self.history_length > 0

        # Runs trained with the straight-line cross_track_error term carry one extra
        # observation channel appended after `actions` (45 -> 46). Without it the
        # checkpoint's first layer is a column wider than what this env emits.
        self._needs_cross_track = bool(obs_cfg.get("cross_track_error", False))
        self._cross_track_clip = float(obs_cfg.get("cross_track_clip", 3.0))  # env.yaml term clip
        self.straight_line_anchor = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.straight_line_normal = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)

        if self._needs_history:
            # One circular buffer per term, oldest at index 0 -- this matches IsaacLab's
            # ObservationManager, which buffers each term separately and flattens
            # (num_envs, history_length, dim) per term before concatenating across terms.
            # Term order and scales below must match the run's env.yaml history group.
            self._history_terms = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
            if self._group_names["history"].endswith("mud_force") or obs_cfg.get("history_has_mud_force"):
                self._history_terms = (*self._history_terms, "mud_force")
            dims = {
                "base_ang_vel": 3,
                "projected_gravity": 3,
                "joint_pos": self.num_actions,
                "joint_vel": self.num_actions,
                "actions": self.num_actions,
                "mud_force": self.num_feet * 3,
            }
            self._history_buf = {
                name: torch.zeros(
                    (self.num_envs, self.history_length, dims[name]), dtype=gs.tc_float, device=gs.device
                )
                for name in self._history_terms
            }
            # IsaacLab's CircularBuffer back-fills every slot with the first sample after a
            # reset rather than leaving zeros, so track pushes per env and do the same.
            self._history_pushes = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    # ------------ scene hooks (overridden by coupled-sim subclasses) ----------------
    def _scene_kwargs(self, show_viewer):
        """Constructor kwargs for gs.Scene. Override to add mpm_options / sph_options."""
        return dict(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=self.env_cfg.get("substeps", 2),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

    def _rigid_material(self):
        """Material for the plane and robot. None uses the genesis default Rigid."""
        return None

    def _camera_kwargs(self):
        """Offscreen recording camera pose. Override to match a subclass's viewer angle."""
        return dict(res=(960, 540), pos=(2.0, -2.2, 1.3), lookat=(0.0, 0.0, 0.3), fov=40)

    def update_camera(self):
        """Slide the recording camera to follow env 0's base, keeping the initial framing.

        No-op when recording is off. Call once per step before cam.render().
        """
        if self.cam is None:
            return
        base = self.base_pos[0]
        lx, ly = float(base[0]), float(base[1])
        lookat = (lx, ly, self._cam_lookat_z)
        pos = (lx + self._cam_offset[0], ly + self._cam_offset[1], self._cam_lookat_z + self._cam_offset[2])
        self.cam.set_pose(pos=pos, lookat=lookat)

    def _add_extra_entities(self):
        """Hook called after the robot is added and before the scene is built."""

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def _resample_heading(self, envs_idx):
        """Draw a new heading target, once per episode, as IsaacLab does at command resample."""
        if not self.heading_command:
            return
        target = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device).uniform_(*self._heading_range)
        if envs_idx is None:
            self.heading_target.copy_(target)
        else:
            torch.where(envs_idx, target, self.heading_target, out=self.heading_target)

    def _update_heading_command(self):
        """Regenerate the yaw command from heading error, overwriting whatever was sampled.

        This is the closed loop the policy was trained inside. It has to run *after* the reset
        pose has been written into base_quat (so a freshly reset env measures its error against
        its new pose) and *before* the observation is assembled (so the policy is handed the
        command that corresponds to where it is actually pointing).
        """
        if not self.heading_command:
            return
        yaw = quat_to_xyz(self.base_quat, rpy=True, degrees=False)[:, 2]
        correction = self.heading_stiffness * wrap_to_pi(self.heading_target - yaw)
        self.commands[:, 2] = torch.clip(correction, self.commands_limits[0][2], self.commands_limits[1][2])

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        # update buffers
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

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # resample commands
        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # after the reset pose is in place, so a reset env measures error against its new pose
        self._update_heading_command()

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        groups = {"policy": self.obs_buf}
        if self._needs_critic:
            # asymmetric critic: base_lin_vel prepended to the actor observation
            groups[self._group_names["critic"]] = torch.concatenate(
                (self.base_lin_vel * self.obs_scales["lin_vel"], self.obs_buf), dim=-1
            )
        if self._needs_encoder:
            # Privileged env factors are only read by the encoder mu, which deployment
            # replaces with the adaptation module phi. Sized so the model builds; never used.
            groups[self._group_names["encoder"]] = torch.zeros(
                (self.num_envs, self.privileged_dim), dtype=gs.tc_float, device=gs.device
            )
        if self._needs_history:
            groups[self._group_names["history"]] = torch.concatenate(
                [self._history_buf[name].reshape(self.num_envs, -1) for name in self._history_terms], dim=-1
            )
        return TensorDict(groups, batch_size=[self.num_envs])

    def _mud_force(self):
        """Per-foot mud reaction force (num_envs, num_feet * 3), world frame, unscaled.

        Zeros here: there is no analytical mud force model in this env. IsaacLab returns
        zeros for this term too until its force term first runs, so the shape and units
        match, but a policy trained with live mud forces sees a constant-zero channel.
        Override in a subclass to feed real contact forces.
        """
        return torch.zeros((self.num_envs, self.num_feet * 3), dtype=gs.tc_float, device=gs.device)

    def _latch_straight_line(self, envs_idx):
        """Latch the reference path that cross-track error is measured against.

        Mirrors go2_mud's `latch_straight_line_anchor` reset event: the path is the ray
        from the robot's spawn XY along its spawn heading. Call after `_reset_idx` has
        written the reset pose into base_pos/base_quat, not before.
        """
        yaw = quat_to_xyz(self.base_quat, rpy=True, degrees=False)[:, 2]
        # Left-hand normal of the path direction, so positive is left of the path.
        normal = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
        anchor = self.base_pos[:, :2]
        if envs_idx is None:
            self.straight_line_anchor.copy_(anchor)
            self.straight_line_normal.copy_(normal)
        else:
            torch.where(envs_idx[:, None], anchor, self.straight_line_anchor, out=self.straight_line_anchor)
            torch.where(envs_idx[:, None], normal, self.straight_line_normal, out=self.straight_line_normal)

    def _cross_track_error(self):
        """Signed perpendicular offset from the commanded straight-line path, (num_envs, 1).

        Positive is left of the path. Along-track progress is deliberately excluded --
        that is what the velocity tracking reward already covers.
        """
        offset = self.base_pos[:, :2] - self.straight_line_anchor
        error = (offset * self.straight_line_normal).sum(dim=-1, keepdim=True)
        return torch.clip(error, -self._cross_track_clip, self._cross_track_clip)

    def _update_history(self):
        """Append the current step to each term buffer, oldest-first, scales pre-applied."""
        samples = {
            "base_ang_vel": self.base_ang_vel * self.obs_scales["ang_vel"],
            "projected_gravity": self.projected_gravity,
            "joint_pos": (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            "joint_vel": self.dof_vel * self.obs_scales["dof_vel"],
            "actions": self.actions,
            "mud_force": self._mud_force() * self.mud_force_scale,
        }
        fresh = (self._history_pushes == 0)[:, None]
        for name in self._history_terms:
            buf = self._history_buf[name]
            sample = samples[name].unsqueeze(1)
            buf.copy_(torch.roll(buf, shifts=-1, dims=1))
            buf[:, -1] = samples[name]
            # first push after a reset back-fills the whole window
            torch.where(fresh.unsqueeze(-1), sample, buf, out=buf)
        self._history_pushes += 1

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)  # joint_names order, not init_dof_pos
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            if self._needs_history:
                self._history_pushes.zero_()
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.default_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            if self._needs_history:
                self._history_pushes.masked_fill_(envs_idx, 0)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        self._resample_commands(envs_idx)
        self._resample_heading(envs_idx)

        # after the reset pose is in place, so the anchor is the pose the robot starts from
        self._latch_straight_line(envs_idx)

    def _update_observation(self):
        if self._needs_history:
            self._update_history()
        terms = [
            self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            self.commands * self.commands_scale,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
        ]
        if self._needs_cross_track:
            terms.append(self._cross_track_error())  # 1, last in the run's policy group
        self.obs_buf = torch.concatenate(terms, dim=-1)

    def reset(self):
        self._reset_idx()
        self._update_heading_command()
        self._update_observation()
        return self.get_observations()

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
