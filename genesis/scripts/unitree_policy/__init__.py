"""Shared policy-loading code for the Go2 eval entry points."""

from scripts.unitree_policy.eval_cfg import (
    DEFAULT_CONFIG,
    DEFAULT_POLICY,
    REPO_ROOT,
    load_cfgs,
    load_policy,
    resolve_policy,
)

__all__ = ["DEFAULT_CONFIG", "DEFAULT_POLICY", "REPO_ROOT", "load_cfgs", "load_policy", "resolve_policy"]
