"""Shared code for the eval entry points.

    eval_cfg   resolve a checkpoint, rebuild its env/train configs, load the policy
    cli        command-line flags and setup steps every eval script shares
    reporting  Reporter (per-step recording, metrics, params.json) and run_rollout
    plots      report figures and their style
"""
