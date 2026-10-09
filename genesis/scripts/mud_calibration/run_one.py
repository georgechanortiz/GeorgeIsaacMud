"""One A2 rollout (flat ground or 4-inch mud) -> joint-torque/power metrics comparable to the hardware bags.

usage: run_one.py '<json config>' <out.json>
config: {"name":..., "vx":0.35, "mud": null | {"material":..., <params>}, "depth":0.1016, "grid_density":48, "substeps":60}
"""
import json, sys, time
import numpy as np, torch
from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
import genesis as gs
cfg = json.loads(sys.argv[1]); out_path = sys.argv[2]
gs.init(backend=gs.gpu, precision="32", logging_level="error")
from envs.go2_env import Go2Env
from scripts.eval_mpm_a2 import A2MPMEnv, DEFAULT_A2_POLICY
from scripts.unitree_policy.eval_cfg import load_cfgs, load_policy, resolve_policy
from genesis.utils.geom import transform_by_quat

GROUPS = {"front_hip": [0, 3], "front_thigh": [1, 4], "front_calf": [2, 5],
          "rear_hip": [6, 9], "rear_thigh": [7, 10], "rear_calf": [8, 11]}
ckpt = resolve_policy(cfg.get("policy", DEFAULT_A2_POLICY))
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs(ckpt.parent)
reward_cfg["reward_scales"] = {}
command_cfg["lin_vel_x_range"] = [0.0, 0.0]; command_cfg["lin_vel_y_range"] = [0.0, 0.0]
mud = cfg.get("mud")
PIT_X = (1.1, 4.55); PIT_Y = 0.38
if mud is None:
    env_cfg["base_init_pos"] = [PIT_X[0] + 0.6, 0.0, env_cfg["base_init_pos"][2]]
    env_cfg["substeps"] = cfg.get("substeps", env_cfg["substeps"])
    env = Go2Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg)
else:
    depth = cfg.get("depth", 0.1016)
    env_cfg["substeps"] = cfg.get("substeps", 60)
    patch = dict(mud, name="mud", patch_lower=(PIT_X[0], -PIT_Y, 0.0), patch_upper=(PIT_X[1], PIT_Y, depth), color=(0.55, 0.42, 0.30))
    mpm_cfg = {"grid_density": cfg.get("grid_density", 48), "coup_friction": mud.get("coup_friction", 0.4),
               "coup_softness": mud.get("coup_softness", 0.01), "wall_height": 0.6, "wall_thickness": 0.03,
               "bounds_lower": (1, -0.4864, -0.1), "bounds_upper": (4.6576, 0.4864, 0.381), "patches": [patch]}
    env_cfg["base_init_pos"] = [PIT_X[0] + 0.6, 0.0, env_cfg["base_init_pos"][2] + depth]
    env = A2MPMEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg, mpm_cfg=mpm_cfg)
policy = load_policy(env, ckpt, train_cfg)
r = env.robot
calves = [l for l in r.links if l.name.endswith("_calf")]
calf_local = [l.idx_local for l in calves]
foot_off = torch.tensor([0.0, 0.0, -0.275], device=gs.device)
mass = float(r.get_mass()); weight = mass * 9.81

stand_s, max_s = cfg.get("stand_s", 1.0), cfg.get("max_s", 9.0)
rows = {k: [] for k in ("tau", "dq", "speed", "x", "foot_z", "fz_mud", "fz_floor", "z")}
obs = env.reset(); t_wall = time.time(); walking_from = None
with torch.no_grad():
    for i in range(int(max_s / env.dt)):
        t = i * env.dt
        if t >= stand_s and walking_from is None:
            lo, hi = env.commands_limits; lo[0] = hi[0] = cfg["vx"]; env.commands[:, 0] = cfg["vx"]; walking_from = i
        obs, *_ = env.step(policy(obs))
        x = float(env.base_pos[0, 0])
        if abs(env.base_euler[0, 0]) > 40 or abs(env.base_euler[0, 1]) > 40: rows["fell"] = True; break
        if x > PIT_X[1] - 0.55: break  # front feet reach the far wall
        rows["tau"].append(r.get_dofs_control_force(env.motors_dof_idx)[0].cpu().numpy())
        rows["dq"].append(env.dof_vel[0].cpu().numpy())
        rows["speed"].append(float(env.base_lin_vel[0, :2].norm())); rows["x"].append(x); rows["z"].append(float(env.base_pos[0, 2]))
        lp, lq = r.get_links_pos()[0, calf_local], r.get_links_quat()[0, calf_local]
        rows["foot_z"].append((lp + transform_by_quat(foot_off.expand(4, 3), lq))[:, 2].cpu().numpy())
        rows["fz_floor"].append(float(r.get_links_net_contact_force()[0, calf_local, 2].sum()))
        rows["fz_mud"].append(float(env.get_foot_forces()[0, :, 2].sum()) if mud is not None else 0.0)
n = len(rows["tau"]); s0 = int((stand_s + 1.0) / env.dt)   # steady walk: 1 s after the walk command
sl = slice(s0, n)
tau, dq = np.array(rows["tau"])[sl], np.array(rows["dq"])[sl]
res = {"name": cfg.get("name"), "cfg": cfg, "steps": n, "window_s": max(0, n - s0) * env.dt, "fell": rows.get("fell", False),
       "wall_s": time.time() - t_wall}
if n - s0 > 50:
    res.update(speed=float(np.mean(rows["speed"][sl])), power_w=float(np.clip(tau * dq, 0, None).sum(1).mean()))
    for g, idx in GROUPS.items():
        a = np.abs(tau[:, idx]); res[f"{g}_mean"] = float(a.mean()); res[f"{g}_p95"] = float(np.percentile(a, 95))
    fz = np.array(rows["foot_z"])[sl]
    res["stance_foot_z_p10"] = float(np.percentile(fz, 10))      # how deep the feet go (sphere centre; 0.032 = on the floor)
    res["base_z"] = float(np.mean(rows["z"][sl]))
    res["mud_weight_frac"] = float(np.mean(rows["fz_mud"][sl]) / weight)
    res["floor_weight_frac"] = float(np.mean(rows["fz_floor"][sl]) / weight)
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "cfg"}))
