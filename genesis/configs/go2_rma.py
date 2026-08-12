"""Config matching the IsaacLab `unitree_go2_flat_rma` runs.

Identical to configs/go2_mud_flat.py except for two values read out of that run's
params/env.yaml: the base spawns lower and the hips default to zero rather than
+/-0.1. Everything else (PD gains, per-joint action scale, command scaling,
substeps, effort limit) is shared, so it is imported rather than duplicated.
"""

from configs.go2_mud_flat import get_cfgs as _get_mud_flat_cfgs
from configs.go2_walking import get_train_cfg

__all__ = ["get_cfgs", "get_train_cfg"]


def get_cfgs():
    env_cfg, obs_cfg, reward_cfg, command_cfg = _get_mud_flat_cfgs()

    # env.yaml init_state.pos -- 0.38 here vs 0.40 for go2_mud_flat
    env_cfg["base_init_pos"] = [0.0, 0.0, 0.38]
    # env.yaml init_state.joint_pos -- hips are 0.0 here, not the +/-0.1 of go2_mud_flat
    env_cfg["default_joint_angles"].update(
        {
            "FL_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
        }
    )

    return env_cfg, obs_cfg, reward_cfg, command_cfg
