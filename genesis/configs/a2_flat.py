"""Config matching the IsaacLab `a2_flat` training runs (Unitree A2), for replaying their
checkpoints inside Genesis.

Source: a2_mud/logs/rsl_rl/a2_flat/2026-10-08_21-07-46/params/env.yaml. The a2_flat_erft
and a2_flat_nomud runs from the same day have an identical robot, actuator, action and
policy-observation block, so this config serves all three.

Observation layout, scales, command handling and action scales are the same as the Go2
go2_mud_flat port, so those are imported. What differs is the robot itself: its URDF, base
height, default pose, and the actuator model, which is ported here group by group.
Deviations from the training setup are marked DEVIATION below.

Actuators (env.yaml scene.robot.actuators, all isaaclab DCMotor):

    group   stiffness  damping  effort_limit  saturation_effort  velocity_limit  armature
    hip        90.0      2.0       120.0           120.0             22.0          0.03
    thigh      90.0      2.0       120.0           120.0             22.0          0.03
    calf      120.0      2.5       180.0           180.0             14.6667       0.03

Training randomized stiffness and damping by a uniform scale in [0.8, 1.25] at every reset
(events.randomize_actuator_gains). Replay uses the nominal values, the centre of that range.
"""

from configs.go2_mud_flat import get_cfgs as _get_go2_cfgs
from configs.go2_walking import get_train_cfg

__all__ = ["get_cfgs", "get_train_cfg", "ACTUATOR_GROUPS"]

# Same order as env.yaml actions.joint_pos.joint_names (preserve_order: true).
_JOINT_NAMES = [
    f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")
]

# One entry per IsaacLab actuator group, keyed by the joint-name suffix it matches.
ACTUATOR_GROUPS = {
    "hip": dict(kp=90.0, kd=2.0, effort_limit=120.0, saturation_effort=120.0, velocity_limit=22.0, armature=0.03),
    "thigh": dict(kp=90.0, kd=2.0, effort_limit=120.0, saturation_effort=120.0, velocity_limit=22.0, armature=0.03),
    "calf": dict(kp=120.0, kd=2.5, effort_limit=180.0, saturation_effort=180.0, velocity_limit=14.6667, armature=0.03),
}


def _group_of(joint_name):
    return joint_name.split("_")[1]  # "FR_calf_joint" -> "calf"


def _by_joint(key):
    return {name: ACTUATOR_GROUPS[_group_of(name)][key] for name in _JOINT_NAMES}


def get_cfgs():
    env_cfg, obs_cfg, reward_cfg, command_cfg = _get_go2_cfgs()

    env_cfg.update(
        {
            "urdf": "a2/urdf/a2.urdf",
            "joint_names": _JOINT_NAMES,
            # env.yaml init_state.joint_pos. Unlike the Go2 port, the hips are 0.0 and the
            # rear thighs match the front ones.
            "default_joint_angles": {
                name: {"hip": 0.0, "thigh": 0.8, "calf": -1.5}[_group_of(name)] for name in _JOINT_NAMES
            },
            "base_init_pos": [0.0, 0.0, 0.45],  # env.yaml init_state.pos
            # PD gains, per actuator group
            "kp": _by_joint("kp"),
            "kd": _by_joint("kd"),
            # DCMotor torque limits. Setting velocity_limit enables the torque-speed cap
            # in the env, so available torque falls off with joint speed as in training.
            "effort_limit": _by_joint("effort_limit"),
            "saturation_effort": _by_joint("saturation_effort"),
            "velocity_limit": _by_joint("velocity_limit"),
            # Rotor inertia. Genesis would otherwise default every joint to 0.1, over 3x the
            # trained value, which slows every joint's response.
            "armature": 0.03,
            # Cancel the kd * substep_dt inertia Genesis' implicit integrator adds to every
            # PD-driven joint, so the trained kp/kd produce the trained response. Validated by
            # scripts/tune_a2_pd.py against an explicit DCMotor reference.
            "compensate_implicit_pd": True,
            # sim dt 0.005 with decimation 4: 200 Hz physics under a 50 Hz policy. The env
            # steps the scene 4 times per control step and refreshes the torque-speed cap
            # before each, as IsaacLab does. substeps is the total per control step and must
            # stay a multiple of decimation (eval_mpm's 60 and eval_sph's 20 both are).
            "decimation": 4,
            "substeps": 4,
        }
    )
    # env.yaml actions.joint_pos.scale: hips 0.125, everything else 0.25. This is the same
    # rule the Go2 config applies, rebuilt here in case joint naming ever diverges.
    env_cfg["action_scale"] = {
        name: (0.125 if _group_of(name) == "hip" else 0.25) for name in _JOINT_NAMES
    }

    # Rewards are unused at eval. base_height_target is only a placeholder for fine-tuning:
    # the A2 stands taller than the Go2.
    reward_cfg["base_height_target"] = 0.40

    # Commands: same trained ranges, heading controller and pinned replay command as the Go2
    # port (velocity_commands scale 1.0, heading_command true, stiffness 0.5).
    return env_cfg, obs_cfg, reward_cfg, command_cfg
