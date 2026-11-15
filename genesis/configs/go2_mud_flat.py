"""Config for the IsaacLab `go2_mud_flat` training run, ported to Genesis.

configs/a2_flat.py builds on this: the A2 shares its observation scales, commands, terminations
and action scaling. Values come from the run's params/env.yaml; deviations are marked DEVIATION.
"""

from configs.go2_walking import get_train_cfg

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
        # Order matters: it is the order of the policy's actions and joint observations.
        "joint_names": _JOINT_NAMES,
        "kp": 25.0,
        "kd": 0.5,
        # Without a torque limit Genesis' PD can exceed what the trained actuator could produce.
        "effort_limit": 23.5,
        # 4 substeps of the 50 Hz control step = the 200 Hz training physics rate.
        "substeps": 4,
        # DEVIATION: training had no attitude termination. These only trip on a real fall,
        # so a tipped robot resets instead of flailing.
        "termination_if_roll_greater_than": 45,  # degree
        "termination_if_pitch_greater_than": 45,
        "base_init_pos": [0.0, 0.0, 0.4],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,
        "action_scale": {
            name: (_HIP_ACTION_SCALE if name.endswith("_hip_joint") else _LEG_ACTION_SCALE)
            for name in _JOINT_NAMES
        },
        "simulate_action_latency": False,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "commands": (1.0, 1.0, 1.0),
        },
    }
    # Unused at eval (rewards are cleared); kept so the config can be used for fine-tuning.
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
    # Linear commands are pinned to a steady forward walk; training sampled (-1.0, 1.0).
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0.0, 0.0],
        # With heading_command on, this is the clip on the heading controller's yaw correction,
        # not the yaw command. It must stay at the trained range: [0, 0] would disable the
        # heading hold.
        "ang_vel_range": [-1.0, 1.0],
        "heading_command": True,
        "heading_control_stiffness": 0.5,
        # Heading target, drawn per episode. [0, 0] holds the spawn heading (walk straight down
        # +x); training sampled (-pi, pi).
        "heading_range": [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
