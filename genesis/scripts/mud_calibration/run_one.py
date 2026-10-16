"""One A2 rollout (flat ground or 4-inch mud) -> joint-torque/power metrics comparable to the hardware bags.

usage: run_one.py '<json config>' <out.json>
config: {"name":..., "vx":0.35, "mud": null | {"material":..., <params>}, "depth":0.1016, "grid_density":48, "substeps":60}
    optional: "policy" (checkpoint), "stand_s" (1.0, time standing before the walk command),
    "max_s" (9.0, rollout cap). Inside "mud", "coup_friction" and "coup_softness" set the coupling.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import genesis as gs  # noqa: E402
from genesis.utils.geom import transform_by_quat  # noqa: E402

from envs import mud  # noqa: E402
from envs.go2_env import Go2Env  # noqa: E402
from envs.mpm_env import A2MPMEnv  # noqa: E402
from scripts.mud_calibration.metrics import torque_metrics  # noqa: E402
from scripts.unitree_policy.eval_cfg import DEFAULT_A2_POLICY, load_cfgs, load_policy, resolve_policy  # noqa: E402

SPAWN_OFFSET = 0.6  # m past the pit's leading edge, clear of the near wall (see eval_mpm_a2.py)
END_MARGIN = 0.55   # stop this far short of the far wall: the front feet reach it
FOOT_OFFSET = (0.0, 0.0, -0.275)  # foot sphere centre in the A2's calf link frame
TILT_LIMIT_DEG = 40.0  # roll or pitch beyond this counts as a fall


def build_env(cfg):
    """The A2 standing at the pit's spawn point, on flat ground (cfg["mud"] is None) or in the mud pit."""
    ckpt = resolve_policy(cfg.get("policy", DEFAULT_A2_POLICY))
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs(ckpt.parent)
    reward_cfg["reward_scales"] = {}
    # Stand still first; the walk command is set once the robot has settled (see rollout).
    command_cfg["lin_vel_x_range"] = [0.0, 0.0]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]

    mud_cfg = cfg.get("mud")
    if mud_cfg is None:
        env_cfg["base_init_pos"] = [mud.PIT_X[0] + SPAWN_OFFSET, 0.0, env_cfg["base_init_pos"][2]]
        env_cfg["substeps"] = cfg.get("substeps", env_cfg["substeps"])
        env = Go2Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg)
    else:
        env_cfg["substeps"] = cfg.get("substeps", 60)
        mpm_cfg = mud.a2_mud_pit(
            mud_cfg,
            cfg.get("depth", 0.1016),
            grid_density=cfg.get("grid_density", 48),
            coup_friction=mud_cfg.get("coup_friction", 0.4),
            coup_softness=mud_cfg.get("coup_softness", 0.01),
        )
        env_cfg["base_init_pos"][1] = 0.0
        mud.spawn_on_pit(env_cfg, mpm_cfg, offset=SPAWN_OFFSET)
        env = A2MPMEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg, mpm_cfg=mpm_cfg)
    return env, load_policy(env, ckpt, train_cfg)


def rollout(env, policy, cfg):
    """Stand, then walk at cfg["vx"] until the far end of the pit, a fall, or cfg["max_s"].

    Returns (per-step rows, fell).
    """
    robot = env.robot
    calf_local = [link.idx_local for link in robot.links if link.name.endswith("_calf")]
    foot_off = torch.tensor(FOOT_OFFSET, device=gs.device).expand(len(calf_local), 3)
    on_mud = cfg.get("mud") is not None
    stand_s, max_s = cfg.get("stand_s", 1.0), cfg.get("max_s", 9.0)
    stop_x = mud.PIT_X[1] - END_MARGIN

    rows = {k: [] for k in ("tau", "dq", "speed", "x", "foot_z", "fz_mud", "fz_floor", "z")}
    fell, walking = False, False
    obs = env.reset()
    with torch.no_grad():
        for i in range(int(max_s / env.dt)):
            if i * env.dt >= stand_s and not walking:
                lo, hi = env.commands_limits
                lo[0] = hi[0] = cfg["vx"]
                env.commands[:, 0] = cfg["vx"]
                walking = True
            obs, *_ = env.step(policy(obs))
            x = float(env.base_pos[0, 0])
            if abs(env.base_euler[0, 0]) > TILT_LIMIT_DEG or abs(env.base_euler[0, 1]) > TILT_LIMIT_DEG:
                fell = True
                break
            if x > stop_x:
                break
            rows["tau"].append(robot.get_dofs_control_force(env.motors_dof_idx)[0].cpu().numpy())
            rows["dq"].append(env.dof_vel[0].cpu().numpy())
            rows["speed"].append(float(env.base_lin_vel[0, :2].norm()))
            rows["x"].append(x)
            rows["z"].append(float(env.base_pos[0, 2]))
            pos, quat = robot.get_links_pos()[0, calf_local], robot.get_links_quat()[0, calf_local]
            rows["foot_z"].append((pos + transform_by_quat(foot_off, quat))[:, 2].cpu().numpy())
            rows["fz_floor"].append(float(robot.get_links_net_contact_force()[0, calf_local, 2].sum()))
            rows["fz_mud"].append(float(env.get_foot_forces()[0, :, 2].sum()) if on_mud else 0.0)
    return rows, fell


def main():
    cfg = json.loads(sys.argv[1])
    out_path = sys.argv[2]
    gs.init(backend=gs.gpu, precision="32", logging_level="error")

    env, policy = build_env(cfg)
    weight = float(env.robot.get_mass()) * 9.81
    t_wall = time.time()
    rows, fell = rollout(env, policy, cfg)

    # Metrics over the steady walk: from 1 s after the walk command to the end.
    n = len(rows["tau"])
    s0 = int((cfg.get("stand_s", 1.0) + 1.0) / env.dt)
    sl = slice(s0, n)
    res = {"name": cfg.get("name"), "cfg": cfg, "steps": n, "window_s": max(0, n - s0) * env.dt, "fell": fell,
           "wall_s": time.time() - t_wall}
    if n - s0 > 50:
        tau, dq = np.array(rows["tau"])[sl], np.array(rows["dq"])[sl]
        res.update(torque_metrics(tau, dq, rows["speed"][sl]))
        # how deep the feet go (sphere centre; 0.032 = on the floor)
        res["stance_foot_z_p10"] = float(np.percentile(np.array(rows["foot_z"])[sl], 10))
        res["base_z"] = float(np.mean(rows["z"][sl]))
        res["mud_weight_frac"] = float(np.mean(rows["fz_mud"][sl]) / weight)
        res["floor_weight_frac"] = float(np.mean(rows["fz_floor"][sl]) / weight)
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "cfg"}))


if __name__ == "__main__":
    main()
