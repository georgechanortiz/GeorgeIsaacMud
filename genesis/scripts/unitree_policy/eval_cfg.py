"""Turning a checkpoint into its configs and a callable policy, shared by the eval scripts.

The RMA / adaptation-module code (stabilise_adaptation, et_bounds, the privileged and history
helpers) serves the Go2 RMA runs and the A2 `a2_flat_rma_*` runs; the plain A2 policies don't
use it. For an A2 RMA run, `_rma_spec` puts the dimensions phi and mu need into
`obs_cfg["rma"]`, which is the single key that makes envs/unitree_env.py emit the "rma_history"
and "env_factors" groups.
"""

import importlib
import pickle
import sys
import types
from importlib import metadata
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

__all__ = ["DEFAULT_A2_POLICY", "DEFAULT_CONFIG", "DEFAULT_POLICY", "REPO_ROOT", "load_cfgs", "load_policy", "resolve_policy"]

# configs/ module for runs whose experiment is not in CONFIG_BY_EXPERIMENT
DEFAULT_CONFIG = "go2_mud_flat"
DEFAULT_POLICY = "logs/go2_mud_flat/2026-07-15_17-09-30/model_4999.pt"
DEFAULT_A2_POLICY = "Policies/v1_rom/model_4999.pt"

# experiment name (logs/<experiment>/<run>/model_*.pt) -> configs/ module
CONFIG_BY_EXPERIMENT = {
    "unitree_go2_flat_rma": "go2_rma",
    "unitree_go2_flat_privobs": "go2_mud_flat",
    "a2_flat": "a2_flat",
    "a2_flat_erft": "a2_flat",
    "a2_flat_erft_shear": "a2_flat",
    "a2_flat_shear": "a2_flat",
    "a2_flat_nomud": "a2_flat",
    "a2_flat_rma_rom": "a2_flat",
    "a2_flat_rma_erft": "a2_flat",
}

# RMA actors are custom rsl-rl models in the mud repo, which rsl-rl imports by name from agent.yaml.
MUD_SRC = Path("/home/rml/Documents/george/mud/source/mud")

# The A2 RMA actor (RMAModel) lives in the a2_mud repo, likewise named in agent.yaml.
A2_MUD_SRC = Path("/home/rml/Documents/george/a2_mud/source/a2_mud")
A2_MUD_AGENTS_PKG = "a2_mud.tasks.manager_based.a2_mud.agents"


def _ensure_mud_models_importable():
    """Make `mud.learning.models.*` importable without importing IsaacLab.

    mud's top-level __init__ needs isaaclab_tasks, so the parent packages are registered as
    stubs that skip it.
    """
    if "mud.learning.models" in sys.modules or not MUD_SRC.is_dir():
        return
    for name, path in (
        ("mud", MUD_SRC / "mud"),
        ("mud.learning", MUD_SRC / "mud" / "learning"),
        ("mud.learning.models", MUD_SRC / "mud" / "learning" / "models"),
    ):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [str(path)]
            sys.modules[name] = stub


def _ensure_a2_mud_models_importable():
    """Make `a2_mud.tasks.manager_based.a2_mud.agents.*` importable without importing IsaacLab.

    Same trick as _ensure_mud_models_importable: a2_mud's top-level __init__ pulls in
    isaaclab_tasks, which Genesis does not have, so every package on the way to `agents` is
    registered as a stub carrying only a __path__. The real modules under it are still found by
    the normal path finder, and rma_model imports only torch, tensordict and rsl_rl.models.
    """
    if A2_MUD_AGENTS_PKG in sys.modules or not A2_MUD_SRC.is_dir():
        return
    path = A2_MUD_SRC
    name = ""
    for part in A2_MUD_AGENTS_PKG.split("."):
        path = path / part
        name = f"{name}.{part}" if name else part
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [str(path)]
            sys.modules[name] = stub


def resolve_policy(policy):
    """Resolve a checkpoint path (absolute or relative to the repo root) to an existing file."""
    path = Path(policy).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path

    if path.is_dir():
        available = sorted(p.name for p in path.glob("model_*.pt"))
        raise IsADirectoryError(
            f"{path} is a directory, not a checkpoint. "
            + (f"Available: {', '.join(available)}" if available else "It contains no model_*.pt files.")
        )
    if not path.is_file():
        available = sorted(p.name for p in path.parent.glob("model_*.pt")) if path.parent.is_dir() else []
        raise FileNotFoundError(
            f"No checkpoint at {path}. "
            + (f"Available in {path.parent}: {', '.join(available)}" if available else f"{path.parent} has no model_*.pt files.")
        )
    return path


def _privileged_dim_from_checkpoint(log_dir):
    """Width of the actor's privileged latent, read off the checkpoint's adaptation module.

    Only RMA runs have one. For the Go2 `adaptation_module.*` layout this is informational only:
    envs/unitree_env.py does not emit those runs' privileged or history groups, so they cannot be
    replayed. a2_mud's RMAModel (`adaptation.*`) can be -- see _rma_spec.
    """
    actor_sd = _actor_state_dict(log_dir)
    if not actor_sd:
        return None
    # a2_mud's RMAModel names phi `adaptation` and ends it in a single `head` Linear, so the
    # index-based match below cannot be widened to cover it (int("embed") raises).
    head = actor_sd.get("adaptation.head.weight")
    if head is not None:
        return int(head.shape[0])
    keys = [k for k in actor_sd if k.startswith("adaptation_module.") and k.endswith(".weight")]
    if not keys:
        return None
    last = max(keys, key=lambda k: int(k.split(".")[1]))
    return int(actor_sd[last].shape[0])


def _actor_state_dict(log_dir):
    """The actor weights of the run's last checkpoint, or None if there is none to read."""
    ckpts = sorted(log_dir.glob("model_*.pt"))
    if not ckpts:
        return None
    return torch.load(ckpts[-1], map_location="cpu", weights_only=False).get("actor_state_dict")


def _rma_spec(log_dir, fallback_history_length=0):
    """The env's `obs_cfg["rma"]` for a run whose actor is a2_mud's RMAModel.

    Everything but the history length comes off the checkpoint, so the groups the env emits are
    sized by the weights that will consume them:

    * `factor_dim`    mu's input width  (`encoder.net.0.weight`, 17 for the A2 runs)
    * `latent_dim`    z's width         (`adaptation.head.weight`, 8)
    * `prop_dim`      one history frame (`adaptation.embed.0.weight`, 42)

    `history_length` is in the run's params/env.yaml; `fallback_history_length` (the actor cfg in
    agent.yaml) covers runs whose env.yaml was not saved.
    """
    actor_sd = _actor_state_dict(log_dir)
    if not actor_sd:
        raise FileNotFoundError(f"{log_dir} has no checkpoint to read the RMA dimensions from")
    missing = [k for k in ("encoder.net.0.weight", "adaptation.head.weight", "adaptation.embed.0.weight") if k not in actor_sd]
    if missing:
        raise KeyError(
            f"the actor in {log_dir} names 'rma_history' in its obs_groups but its checkpoint is missing {missing}; "
            "it is not an a2_mud RMAModel"
        )
    length = _history_spec(log_dir, "rma_history")[0] or int(fallback_history_length or 0)
    if length < 1:
        raise ValueError(
            f"could not find the rma_history length for {log_dir}: neither params/env.yaml nor the actor cfg in "
            "params/agent.yaml gives one"
        )
    return {
        "factor_dim": int(actor_sd["encoder.net.0.weight"].shape[1]),
        "latent_dim": int(actor_sd["adaptation.head.weight"].shape[0]),
        "prop_dim": int(actor_sd["adaptation.embed.0.weight"].shape[1]),
        "history_length": length,
    }


def _history_spec(log_dir, history_group):
    """Read the run's history group definition out of params/env.yaml.

    Returns (history_length, has_mud_force).
    """
    env_yaml = log_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        return 0, False
    groups = yaml.unsafe_load(env_yaml.read_text()).get("observations", {})
    group = groups.get(history_group)
    if not isinstance(group, dict):
        return 0, False
    terms = [n for n, t in group.items() if isinstance(t, dict) and t.get("func")]
    return int(group.get("history_length") or 0), "mud_force" in terms


def _obs_terms(log_dir, group_name):
    """Term names in one of the run's observation groups, in declaration order."""
    env_yaml = log_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        return []
    group = yaml.unsafe_load(env_yaml.read_text()).get("observations", {}).get(group_name)
    if not isinstance(group, dict):
        return []
    return [name for name, term in group.items() if isinstance(term, dict) and term.get("func")]


def _migrate_legacy_agent_cfg(train_cfg):
    """Rewrite a pre-5.0 rsl-rl agent cfg (one `policy: ActorCritic` block) as 5.x actor/critic cfgs.

    The A2 runs were trained on rsl-rl < 5. Only the plain MLP ActorCritic is translated; anything
    else is left to fail loudly.
    """
    policy = train_cfg.get("policy")
    if "actor" in train_cfg or not isinstance(policy, dict) or policy.get("class_name") != "ActorCritic":
        return train_cfg
    train_cfg = dict(train_cfg)
    train_cfg.pop("policy")
    activation = policy.get("activation", "elu")
    train_cfg["actor"] = {
        "class_name": "MLPModel",
        "hidden_dims": list(policy["actor_hidden_dims"]),
        "activation": activation,
        "obs_normalization": bool(policy.get("actor_obs_normalization", False)),
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": float(policy.get("init_noise_std", 1.0)),
            "std_type": policy.get("noise_std_type", "scalar"),
        },
    }
    train_cfg["critic"] = {
        "class_name": "MLPModel",
        "hidden_dims": list(policy["critic_hidden_dims"]),
        "activation": activation,
        "obs_normalization": bool(policy.get("critic_obs_normalization", False)),
    }
    return train_cfg


def _migrate_legacy_checkpoint(loaded):
    """Map a pre-5.0 ActorCritic checkpoint onto the 5.x actor MLPModel's state dict.

    Old: model_state_dict = {actor.N.*, critic.N.*, std | log_std, [actor_obs_normalizer.*]}.
    New: actor_state_dict = {mlp.N.*, distribution.std_param, [obs_normalizer.*]}.
    """
    if "actor_state_dict" in loaded or "model_state_dict" not in loaded:
        return loaded
    actor = {}
    for key, value in loaded["model_state_dict"].items():
        if key.startswith("actor."):
            actor["mlp." + key[len("actor."):]] = value
        elif key in ("std", "log_std"):
            actor["distribution.std_param"] = value
        elif key.startswith("actor_obs_normalizer."):
            actor["obs_normalizer." + key[len("actor_obs_normalizer."):]] = value
    print("[INFO] converted a pre-5.0 rsl-rl ActorCritic checkpoint to the 5.x actor layout.")
    return {**loaded, "actor_state_dict": actor}


def _agent_experiment_name(log_dir):
    """The run's experiment name from params/agent.yaml, for run directories moved out of logs/<experiment>/."""
    agent_yaml = log_dir / "params" / "agent.yaml"
    if not agent_yaml.exists():
        return None
    return yaml.unsafe_load(agent_yaml.read_text()).get("experiment_name")


def load_cfgs(log_dir, config=None):
    """Assemble (env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg) for a run.

    A run's own cfgs.pkl is used as is. Otherwise (IsaacLab runs) the env cfgs come from a
    configs/ module and the agent cfg from the run's params/agent.yaml.
    """
    cfgs_path = log_dir / "cfgs.pkl"
    if cfgs_path.exists():
        with open(cfgs_path, "rb") as f:
            return pickle.load(f)

    if config is None:
        config = CONFIG_BY_EXPERIMENT.get(log_dir.parent.name) or CONFIG_BY_EXPERIMENT.get(
            _agent_experiment_name(log_dir), DEFAULT_CONFIG
        )
    mod = importlib.import_module(f"configs.{config}")
    env_cfg, obs_cfg, reward_cfg, command_cfg = mod.get_cfgs()

    # Some Go2 runs have an extra cross_track_error observation (not supported by the env).
    obs_cfg["cross_track_error"] = "cross_track_error" in _obs_terms(log_dir, "policy")

    agent_yaml = log_dir / "params" / "agent.yaml"
    if not agent_yaml.exists():
        return env_cfg, obs_cfg, reward_cfg, command_cfg, mod.get_train_cfg(log_dir.name)

    train_cfg = _migrate_legacy_agent_cfg(yaml.unsafe_load(agent_yaml.read_text()))
    train_cfg["logger"] = "tensorboard"  # not the training run's wandb
    train_cfg["resume"] = False

    obs_groups = train_cfg.get("obs_groups") or {}
    if not obs_groups:
        # rsl-rl would infer this anyway, with a deprecation warning
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        train_cfg["obs_groups"] = obs_groups
    else:
        obs_cfg["obs_groups"] = obs_groups
        _ensure_mud_models_importable()
        _ensure_a2_mud_models_importable()
        priv_dim = _privileged_dim_from_checkpoint(log_dir)
        if priv_dim is not None:
            obs_cfg["privileged_dim"] = priv_dim
        # RMA actors read "rma_history" and "env_factors" out of the observation TensorDict by
        # name. This key is what makes the env emit them; no other run ever sets it.
        if any("rma_history" in groups for groups in obs_groups.values()):
            obs_cfg["rma"] = _rma_spec(log_dir, (train_cfg.get("actor") or {}).get("history_length"))
            # Only the actor is loaded at eval (see load_policy), but rsl-rl still builds a critic
            # and validates every observation set against the env's groups. The env has no
            # privileged "critic" group, so redirect the sets it cannot serve.
            emitted = {"policy", "rma_history", "env_factors"}
            for set_name, groups in obs_groups.items():
                missing_groups = [g for g in groups if g not in emitted]
                if missing_groups:
                    print(
                        f"[INFO] obs set '{set_name}' wants {missing_groups}, which the Genesis env does not emit; "
                        "replaying it on the policy observation (this set's model is never loaded)."
                    )
                    obs_groups[set_name] = ["policy"]
    history_group = (obs_groups.get("history") or [None])[0]
    if history_group:
        length, has_mud_force = _history_spec(log_dir, history_group)
        obs_cfg["history_length"] = length
        obs_cfg["history_has_mud_force"] = has_mud_force

    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


# Go2 base-link mass (assets/go2/urdf/go2.urdf); training randomized it by an additive offset.
_BASE_LINK_MASS_KG = 6.921


def et_bounds(log_dir, latent_dim, pad=0.15):
    """Per-dimension (lo, hi) of ``e_t`` during training, read from the run's env.yaml, widened by ``pad``.

    Terms in order: body_mass(1), body_com_pos(3), body_material_props(3), mud_coeffs(5), each
    times its observation scale. Returns (None, None) if the yaml doesn't match this term list.
    """
    env_yaml = log_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        return None, None
    try:
        y = yaml.unsafe_load(env_yaml.read_text())
        ev, priv = y["events"], y["observations"]["privileged"]

        def scale_of(term):
            s = priv.get(term, {}).get("scale", 1.0)
            return float(s) if s is not None else 1.0

        lo, hi = [], []

        # body_mass
        m_lo, m_hi = ev["randomize_rigid_body_mass_base"]["params"]["mass_distribution_params"]
        s = scale_of("body_mass")
        lo.append((_BASE_LINK_MASS_KG + m_lo) * s)
        hi.append((_BASE_LINK_MASS_KG + m_hi) * s)

        # body_com_pos
        s = scale_of("body_com_pos")
        com = ev["randomize_com_positions"]["params"]["com_range"]
        for axis in ("x", "y", "z"):
            a, b = com[axis]
            lo.append(a * s)
            hi.append(b * s)

        # body_material_props: static friction, dynamic friction, restitution
        s = scale_of("body_material_props")
        mat = ev["randomize_rigid_body_material"]["params"]
        for key in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
            a, b = mat[key]
            lo.append(a * s)
            hi.append(b * s)

        # mud_coeffs: suction, c1, c2, area, depth
        s = scale_of("mud_coeffs")
        mud = ev["mud_force_term"]["params"]
        depth = mud.get("depth_range") or ev["reset_mud_depth"]["params"]["depth_range"]
        for a, b in (mud["f_range"], mud["c1"], mud["c2"], mud["surface_area"], depth):
            lo.append(a * s)
            hi.append(b * s)
    except (KeyError, TypeError, yaml.YAMLError) as e:
        print(
            f"[WARN] could not read e_t bounds from {env_yaml} ({e!r}); not clamping. "
            "The mud force model's event params likely no longer match the term list this "
            "function assumes -- update it to the run's actual parametrization if you need "
            "--phi-clamp to work here."
        )
        return None, None

    if len(lo) != latent_dim:
        print(f"[WARN] e_t bounds came to {len(lo)} dims but the model's latent is {latent_dim}; not clamping.")
        return None, None

    out_lo, out_hi = [], []
    for a, b in zip(lo, hi):
        margin = pad * (b - a)
        out_lo.append(a - margin)
        out_hi.append(b + margin)
    return out_lo, out_hi


def stabilise_adaptation(model, bounds=(None, None), smooth=1.0, hold_steps=1, device=None):
    """Wrap ``model.adapt`` so ``phi``'s estimate of ``e_t`` is held, clamped and smoothed.

    Out of distribution, phi's estimate is jittery and the actor follows it. Options:

    * ``hold_steps``: recompute only every N steps and reuse the value in between.
    * ``bounds``: clamp to the range seen in training.
    * ``smooth``: EMA weight on each new estimate (1.0 = off).

    Also sets ``model.last_e_t`` and ``model.last_e_t_fresh`` each step for the report. The filter
    state is not reset between episodes.
    """
    lo, hi = bounds
    lo_t = torch.tensor(lo, device=device) if lo is not None else None
    hi_t = torch.tensor(hi, device=device) if hi is not None else None
    raw_adapt = model.adapt
    state = {"z": None, "n": 0}

    def adapt(obs):
        fresh = state["z"] is None or state["n"] % hold_steps == 0
        if fresh:
            z = raw_adapt(obs)
            if lo_t is not None:
                z = torch.clamp(z, lo_t, hi_t)
            prev = state["z"]
            if smooth < 1.0 and prev is not None and prev.shape == z.shape:
                z = smooth * z + (1.0 - smooth) * prev
            state["z"] = z.detach()
        state["n"] += 1
        model.last_e_t = state["z"]
        model.last_e_t_fresh = fresh
        return state["z"]

    # get_latent calls self.adapt(), so this replaces it for this instance
    model.adapt = adapt
    what = []
    if hold_steps > 1:
        what.append(f"held for {hold_steps} steps")
    if lo_t is not None:
        what.append("clamped to the training e_t range")
    if smooth < 1.0:
        what.append(f"EMA-smoothed (alpha={smooth})")
    if what:
        print(f"[INFO] phi estimate {', '.join(what)}.")


def load_policy(env, ckpt_path, train_cfg, *, phi_smooth=1.0, phi_clamp=False, phi_rate_hz=None):
    """Build a runner around `env` and return a callable `obs_dict -> actions`.

    Only the actor is loaded; asymmetric-critic checkpoints would otherwise fail to load.
    """
    runner = OnPolicyRunner(env, train_cfg, str(ckpt_path.parent), device=gs.device)
    loaded = _migrate_legacy_checkpoint(torch.load(str(ckpt_path), weights_only=False, map_location=gs.device))
    runner.alg.load(loaded, {"actor": True}, True)
    policy = runner.get_inference_policy(device=gs.device)

    # RMA actors: use the adaptation module (phi), since the env has no privileged observations.
    if hasattr(policy, "set_adaptation_mode"):
        policy.set_adaptation_mode(True)
        bounds = et_bounds(ckpt_path.parent, policy.latent_dim) if phi_clamp else (None, None)

        hold_steps = 1
        if phi_rate_hz:
            control_hz = 1.0 / env.dt
            hold_steps = max(int(round(control_hz / phi_rate_hz)), 1)
            print(
                f"[INFO] control loop {control_hz:.0f} Hz, phi requested at {phi_rate_hz:g} Hz "
                f"-> holding each estimate for {hold_steps} steps "
                f"(effective {control_hz / hold_steps:.1f} Hz)."
            )

        stabilise_adaptation(
            policy, bounds=bounds, smooth=phi_smooth, hold_steps=hold_steps, device=gs.device
        )
    elif phi_clamp or phi_smooth < 1.0 or phi_rate_hz:
        print("[WARN] --phi-clamp / --phi-smooth / --phi-rate need an actor with an adaptation module; ignored.")

    return policy
