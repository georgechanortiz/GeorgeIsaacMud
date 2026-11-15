"""Config for the IsaacLab `a2_flat` training runs (Unitree A2), ported to Genesis.

Serves a2_flat, a2_flat_erft and a2_flat_nomud, which share the robot, actuators, actions and
policy observations (source: a2_mud/logs/rsl_rl/a2_flat/2026-10-08_21-07-46/params/env.yaml).
Observation scales, commands and terminations come from configs/go2_mud_flat.py; this file sets
the robot and its actuators.

Training randomized stiffness and damping by a factor in [0.8, 1.25] at every reset; replay uses
the nominal values.
"""

from configs.go2_mud_flat import get_cfgs as _get_go2_cfgs
from configs.go2_walking import get_train_cfg

__all__ = ["get_cfgs", "get_train_cfg", "ACTUATOR_GROUPS", "JOINT_NAMES"]

# Order of the policy's actions and joint observations.
JOINT_NAMES = [
    f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")
]

# IsaacLab DCMotor actuator groups (env.yaml scene.robot.actuators).
ACTUATOR_GROUPS = {
    "hip": dict(kp=90.0, kd=2.0, effort_limit=120.0, saturation_effort=120.0, velocity_limit=22.0, armature=0.03),
    "thigh": dict(kp=90.0, kd=2.0, effort_limit=120.0, saturation_effort=120.0, velocity_limit=22.0, armature=0.03),
    "calf": dict(kp=120.0, kd=2.5, effort_limit=180.0, saturation_effort=180.0, velocity_limit=14.6667, armature=0.03),
}


def _group_of(joint_name):
    return joint_name.split("_")[1]  # "FR_calf_joint" -> "calf"


def _by_joint(key):
    return {name: ACTUATOR_GROUPS[_group_of(name)][key] for name in JOINT_NAMES}


def get_cfgs():
    env_cfg, obs_cfg, reward_cfg, command_cfg = _get_go2_cfgs()

    env_cfg.update(
        {
            "urdf": "a2/urdf/a2.urdf",
            "joint_names": JOINT_NAMES,
            "default_joint_angles": {
                name: {"hip": 0.0, "thigh": 0.8, "calf": -1.5}[_group_of(name)] for name in JOINT_NAMES
            },
            "base_init_pos": [0.0, 0.0, 0.45],
            "kp": _by_joint("kp"),
            "kd": _by_joint("kd"),
            # Setting velocity_limit enables the DCMotor torque-speed cap in the env.
            "effort_limit": _by_joint("effort_limit"),
            "saturation_effort": _by_joint("saturation_effort"),
            "velocity_limit": _by_joint("velocity_limit"),
            # Genesis would otherwise default every joint to 0.1, over 3x the trained value.
            "armature": 0.03,
            # See UnitreeEnv._effective_armature.
            "compensate_implicit_pd": True,
            # 200 Hz physics under the 50 Hz policy, as in training. Scripts that raise substeps
            # (MPM) must keep it a multiple of decimation.
            "decimation": 4,
            "substeps": 4,
        }
    )
    env_cfg["action_scale"] = {
        name: (0.125 if _group_of(name) == "hip" else 0.25) for name in JOINT_NAMES
    }

    # Unused at eval; the A2 stands taller than the Go2.
    reward_cfg["base_height_target"] = 0.40

    return env_cfg, obs_cfg, reward_cfg, command_cfg
