"""Config matching the IsaacLab `go2_mud_flat` training run, for replaying its
checkpoints (e.g. logs/2026-07-15_17-09-30) inside Genesis.

Every value here was read out of that run's params/env.yaml and params/agent.yaml.
Deviations from the training setup are marked DEVIATION below; everything else is
a faithful port. Do not "tidy" these values toward configs/go2_walking.py -- the
differences are what make the policy see the observations it was trained on.
"""

from configs.go2_walking import get_train_cfg  # PPO/actor/critic hyperparams are identical

__all__ = ["get_cfgs", "get_train_cfg"]

_HIP_ACTION_SCALE = 0.125  # env.yaml: actions.joint_pos.scale ".*_hip_joint"
_LEG_ACTION_SCALE = 0.25  # env.yaml: actions.joint_pos.scale "^(?!.*_hip_joint).*"

_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # Asymmetric hips: env.yaml init_state.joint_pos is `.*L_hip_joint: 0.1`,
        # `.*R_hip_joint: -0.1`. go2_walking uses 0.0 for all four.
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        # Order matters: matches actions.joint_pos.joint_names and the joint_pos/joint_vel
        # observation terms, all of which set preserve_order: True.
        "joint_names": _JOINT_NAMES,
        # PD -- env.yaml actuators.base_legs (DCMotor stiffness/damping)
        "kp": 25.0,
        "kd": 0.5,
        # DCMotor effort_limit/saturation_effort. Genesis applies PD with no torque
        # ceiling by default, which let peaks reach 35.5 N*m (51% over what the training
        # actuator could produce) on ~6% of steps.
        "effort_limit": 23.5,
        # sim dt 0.005 = 200 Hz physics; the env's dt is the 50 Hz control rate, so
        # 4 substeps reproduces the training physics rate (the default 2 gives 100 Hz).
        "substeps": 4,
        # DEVIATION: training terminated only on time_out and terrain_out_of_bounds;
        # illegal_contact was disabled and there was no attitude termination at all.
        # go2_walking's 10 deg would fire during a normal trot. These thresholds only
        # trip on a genuine fall, so a tipped robot still resets instead of flailing.
        "termination_if_roll_greater_than": 45,  # degree
        "termination_if_pitch_greater_than": 45,
        "base_init_pos": [0.0, 0.0, 0.4],  # env.yaml init_state.pos (go2_walking uses 0.42)
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,  # env.yaml commands.base_velocity.resampling_time_range
        "action_scale": {
            name: (_HIP_ACTION_SCALE if name.endswith("_hip_joint") else _LEG_ACTION_SCALE)
            for name in _JOINT_NAMES
        },
        # IsaacLab applies the action in the same step it is produced; the 1-step delay
        # go2_walking simulates was not present during training.
        "simulate_action_latency": False,
        "clip_actions": 100.0,  # env.yaml actions.joint_pos.clip
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,  # unused by the policy obs group; kept for interface compatibility
            "ang_vel": 0.25,  # env.yaml observations.policy.base_ang_vel.scale
            "dof_pos": 1.0,  # joint_pos.scale
            "dof_vel": 0.05,  # joint_vel.scale
            # velocity_commands.scale is 1.0 -- go2_walking instead folds in
            # [lin_vel, lin_vel, ang_vel] = [2.0, 2.0, 0.25], which corrupts obs[6:9].
            "commands": (1.0, 1.0, 1.0),
        },
    }
    # Rewards are not used at eval (eval.py clears reward_scales). These mirror
    # go2_walking so the config stays usable for fine-tuning.
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    # Trained ranges were lin_vel_x/y and ang_vel_z all (-1.0, 1.0) with a heading
    # controller driving yaw. Linear commands are pinned to a steady forward walk for
    # replay; widen to (-1.0, 1.0) to sample the full commanded distribution.
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0.0, 0.0],
        # NOT the yaw command. With heading_command on this is the *clip* applied to the
        # generated correction, so it has to stay at the trained range -- pinning it to
        # [0.0, 0.0] clamps every correction to zero and reduces this to open-loop yaw,
        # which is what made the replay drift.
        "ang_vel_range": [-1.0, 1.0],
        # env.yaml commands.base_velocity: heading_command true, heading_control_stiffness
        # 0.5, rel_heading_envs 1.0. Every checkpoint under logs/ was trained this way, so
        # a replay without it feeds the policy a yaw channel it has never seen.
        "heading_command": True,
        "heading_control_stiffness": 0.5,
        # Trained range was (-pi, pi), redrawn each episode. Pinned to 0 here = hold the
        # spawn heading, i.e. walk straight down +X. Widen to [-3.14159, 3.14159] to
        # sample headings as trained.
        "heading_range": [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
