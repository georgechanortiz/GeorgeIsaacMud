"""Config and policy loading shared by every eval entry point.

scripts/eval.py, scripts/eval_mpm.py and scripts/eval_sph.py differ only in the
scene they build; how a checkpoint becomes a callable policy lives here so that
architectural changes to loading (e.g. adding an RMA-style adaptation module)
happen in one place.
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

__all__ = ["DEFAULT_CONFIG", "DEFAULT_POLICY", "REPO_ROOT", "load_cfgs", "load_policy", "resolve_policy"]

# Env config used when a run has no cfgs.pkl of its own (i.e. anything trained outside
# this repo). Change here rather than per-script so all three entry points agree.
DEFAULT_CONFIG = "go2_mud_flat"
DEFAULT_POLICY = "logs/go2_mud_flat/2026-07-15_17-09-30/model_4999.pt"

# Genesis-side dynamics differ per experiment (base height, hip defaults), so map the
# experiment directory -- logs/<experiment>/<run>/model_*.pt -- to a configs/ module.
CONFIG_BY_EXPERIMENT = {
    "unitree_go2_flat_rma": "go2_rma",
    "unitree_go2_flat_privobs": "go2_mud_flat",
}

# RMA / privileged-obs actors are custom rsl-rl models living in the mud repo. rsl-rl
# resolves them by qualified name out of agent.yaml, so they must be importable.
MUD_SRC = Path("/home/rml/Documents/george/mud/source/mud")


def _ensure_mud_models_importable():
    """Make `mud.learning.models.*` importable without dragging in IsaacLab.

    The model classes themselves depend only on torch/tensordict/rsl_rl, but mud's
    top-level __init__ imports .tasks, which needs isaaclab_tasks. Register namespace
    stubs for the parent packages so importing the leaf modules skips that __init__.
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


def resolve_policy(policy):
    """Resolve a checkpoint path into an existing file.

    Accepts an absolute path or one relative to the repo root, so both of these work
    from any working directory:

        /home/.../genesis/logs/unitree_go2_flat/2026-05-28 - Baseline/model_4999.pt
        logs/unitree_go2_flat/2026-05-28 - Baseline/model_4999.pt
    """
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
    """Ground-truth width of the actor's privileged latent, read off a checkpoint's
    adaptation module.

    envs/go2_env.py defaults obs_cfg["privileged_dim"] to 12 and nothing in this
    fallback path (runs with no cfgs.pkl of their own) ever overrides it. That default
    happens to match some runs by coincidence, but it silently breaks the moment a
    run's privileged term set is a different width -- re-deriving it from env.yaml's
    term list would have the same problem, since that list drifts independently
    whenever a term's own width changes (e.g. mud_coeffs gaining an entry). Reading it
    off the checkpoint itself is immune to that: it's whatever the actor actually was
    trained with, always.
    """
    ckpts = sorted(log_dir.glob("model_*.pt"))
    if not ckpts:
        return None
    actor_sd = torch.load(ckpts[-1], map_location="cpu", weights_only=False).get("actor_state_dict")
    if not actor_sd:
        return None
    keys = [k for k in actor_sd if k.startswith("adaptation_module.") and k.endswith(".weight")]
    if not keys:
        return None
    last = max(keys, key=lambda k: int(k.split(".")[1]))
    return int(actor_sd[last].shape[0])


def _history_spec(log_dir, history_group):
    """Read the run's history group definition out of params/env.yaml.

    Returns (history_length, has_mud_force). The adaptation module's input width is
    history_length * sum(term dims), so both must match the trained model exactly.
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
    """Term names in one of the run's observation groups, in declaration order.

    Runs in the same experiment directory can differ in which terms they trained with,
    so the group has to be read per-run rather than assumed from the configs/ module.
    """
    env_yaml = log_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        return []
    group = yaml.unsafe_load(env_yaml.read_text()).get("observations", {}).get(group_name)
    if not isinstance(group, dict):
        return []
    return [name for name, term in group.items() if isinstance(term, dict) and term.get("func")]


def load_cfgs(log_dir, config=None):
    """Assemble (env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg) for a run.

    Runs trained in this repo carry their own cfgs.pkl and are used verbatim. For runs
    trained in IsaacLab, the env cfg comes from a configs/ module (Genesis dynamics are
    not derivable from IsaacLab's yaml) while the *agent* cfg is read straight from the
    run's params/agent.yaml -- that is what selects the actor class and its obs_groups,
    so RMA / privileged-obs architectures rebuild exactly as trained.
    """
    cfgs_path = log_dir / "cfgs.pkl"
    if cfgs_path.exists():
        with open(cfgs_path, "rb") as f:
            return pickle.load(f)

    if config is None:
        config = CONFIG_BY_EXPERIMENT.get(log_dir.parent.name, DEFAULT_CONFIG)
    mod = importlib.import_module(f"configs.{config}")
    env_cfg, obs_cfg, reward_cfg, command_cfg = mod.get_cfgs()

    # The straight-line cross_track_error term was added partway through the go2_mud
    # runs, so runs sharing an experiment directory (and hence a configs/ module) can
    # differ by one observation. Take it from the run's own env.yaml.
    obs_cfg["cross_track_error"] = "cross_track_error" in _obs_terms(log_dir, "policy")

    agent_yaml = log_dir / "params" / "agent.yaml"
    if not agent_yaml.exists():
        return env_cfg, obs_cfg, reward_cfg, command_cfg, mod.get_train_cfg(log_dir.name)

    train_cfg = yaml.unsafe_load(agent_yaml.read_text())
    train_cfg["logger"] = "tensorboard"  # never re-init the training run's wandb at eval
    train_cfg["resume"] = False

    obs_groups = train_cfg.get("obs_groups") or {}
    if not obs_groups:
        # Some runs ship an empty obs_groups; rsl-rl then infers "policy" for both roles
        # and warns. Say it explicitly instead -- same behaviour, no deprecation noise.
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        train_cfg["obs_groups"] = obs_groups
    else:
        # Tell the env which groups to emit; it produces only "policy" without this.
        obs_cfg["obs_groups"] = obs_groups
        _ensure_mud_models_importable()
        priv_dim = _privileged_dim_from_checkpoint(log_dir)
        if priv_dim is not None:
            obs_cfg["privileged_dim"] = priv_dim
    history_group = (obs_groups.get("history") or [None])[0]
    if history_group:
        length, has_mud_force = _history_spec(log_dir, history_group)
        obs_cfg["history_length"] = length
        obs_cfg["history_has_mud_force"] = has_mud_force

    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


# Nominal mass of the Go2 base link, from assets/go2/urdf/go2.urdf. The mass randomization event is
# an *additive* offset on top of this, so the absolute bound needs the nominal value.
_BASE_LINK_MASS_KG = 6.921


def et_bounds(log_dir, latent_dim, pad=0.15):
    """Per-dimension (lo, hi) that ``e_t`` actually spanned during training, read from the run's yaml.

    Order follows ``PrivilegedCfg``'s term declaration -- body_mass(1), body_com_pos(3),
    body_material_props(3), mud_coeffs(5) -- and each term's ``scale`` is applied, because the actor
    was trained on the *scaled* values. ``pad`` widens every bound by that fraction of its width so a
    legitimate estimate near the edge of the distribution is not clipped.

    Returns ``(lo, hi)`` as float lists, or ``(None, None)`` if the yaml is missing, an expected
    event/param key is absent, or the dimensions do not add up to ``latent_dim`` -- in which case
    clamping is skipped rather than guessed at. The mud force model's parametrization has already
    changed once (a Coulomb-style ``f_range``/``c1``/``c2`` law replaced by a Herschel-Bulkley-style
    ``f_range``/``k_range``/``n_range``/``tau_y_range`` one on at least one run seen so far), so this
    hardcoded term list *will* go stale again -- failing closed here is deliberate.
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

        # body_mass: nominal base-link mass plus the additive randomization offset
        m_lo, m_hi = ev["randomize_rigid_body_mass_base"]["params"]["mass_distribution_params"]
        s = scale_of("body_mass")
        lo.append((_BASE_LINK_MASS_KG + m_lo) * s)
        hi.append((_BASE_LINK_MASS_KG + m_hi) * s)

        # body_com_pos: x/y/z offsets
        s = scale_of("body_com_pos")
        com = ev["randomize_com_positions"]["params"]["com_range"]
        for axis in ("x", "y", "z"):
            a, b = com[axis]
            lo.append(a * s)
            hi.append(b * s)

        # body_material_props: static friction, dynamic friction, restitution.
        # These are averaged across all bodies before being observed, so the true spread is far
        # narrower than the per-body sampling range -- using the raw range keeps the clamp loose.
        s = scale_of("body_material_props")
        mat = ev["randomize_rigid_body_material"]["params"]
        for key in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
            a, b = mat[key]
            lo.append(a * s)
            hi.append(b * s)

        # mud_coeffs: suction, c1, c2, area, depth -- the depth comes from its own reset event
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
    """Rate-limit, clamp and EMA-smooth ``phi``'s estimate of ``e_t`` before the actor consumes it.

    Why this exists: the actor is trained on the *true* ``e_t`` every step, so it learns to trust that
    input completely. At deployment in a different simulator, ``phi`` is reading proprioceptive
    history from dynamics it has never seen and its output is both out of range and jittery. The
    actor faithfully switches behaviour along with it, and the gait never settles into a limit cycle.

    Three independent, composable damping mechanisms:

    * ``hold_steps`` -- zero-order hold. Recompute ``phi`` only every N policy steps and reuse the
      cached value in between, so the estimate refreshes at ``control_rate / N``. This is the most
      physically honest of the three: the mud a robot is standing in does not change at 50 Hz, so
      re-estimating it every control step buys nothing and only injects jitter. It also mirrors how
      a real system would run a slow estimator alongside a fast controller.
    * ``bounds`` -- clamp to values the actor was actually trained against.
    * ``smooth`` -- EMA weight on each new sample: 1.0 disables it, smaller filters harder (0.1
      keeps ~90% of the previous estimate). Applied only on refresh steps, so it composes with the
      hold rather than fighting it.

    Neither clamping nor filtering substitutes for training the actor to tolerate a wrong ``e_t`` --
    that is a training-side fix -- but all three are free at inference and testable on an existing
    checkpoint.

    Filter state is not reset on episode boundaries, which is harmless for the single continuous
    rollouts these scripts run.

    Always wraps ``model.adapt`` (even with every damping mechanism off) so that
    ``model.last_e_t`` (the value actually fed to the actor) and ``model.last_e_t_fresh``
    (whether this step recomputed it or reused the held one) are available for reporting
    -- see Reporter._plot_phi in reporting.py. With no flags set this is a pass-through:
    every step is fresh and last_e_t equals the raw estimate.
    """
    lo, hi = bounds
    lo_t = torch.tensor(lo, device=device) if lo is not None else None
    hi_t = torch.tensor(hi, device=device) if hi is not None else None
    raw_adapt = model.adapt
    state = {"z": None, "n": 0}

    def adapt(obs):
        # refresh on the first call and every hold_steps thereafter; otherwise reuse the cache
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

    # shadows the bound method on this instance; get_latent calls self.adapt(), so this intercepts it
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

    Actor-only: eval never uses the critic, and checkpoints trained with an asymmetric
    actor-critic (e.g. a privileged 48-dim critic) would otherwise fail to load here.

    This is the seam for architectural changes to inference. An RMA-style setup would
    load its adaptation module here and return a callable that runs the estimator on
    the proprioceptive history, then feeds actor(obs, latent) -- callers only depend
    on the returned callable, not on how it is assembled.
    """
    runner = OnPolicyRunner(env, train_cfg, str(ckpt_path.parent), device=gs.device)
    runner.load(str(ckpt_path), load_cfg={"actor": True})
    policy = runner.get_inference_policy(device=gs.device)

    # RMA / privileged-obs actors can source their latent either from the privileged
    # encoder mu (training) or the adaptation module phi (deployment). Only phi runs on
    # observations this env can produce, so force it -- otherwise the actor silently
    # reads the zero-filled privileged group and the latent is meaningless.
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
