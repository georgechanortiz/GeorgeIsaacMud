"""Report figures and their shared style.

Every figure function takes the directory to write into, so the same plots serve the sim
reports (Reporter.finalize), the hardware runs (scripts/mud_calibration/plot_hw_report.py)
and the sphere drop bench (scripts/eval_mpm_1d_intrusion.py).
"""

import matplotlib

matplotlib.use("Agg")  # report files only; never open a GUI alongside the genesis viewer

import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-3 of the validated reference palette, used in fixed order.
# Three series max per axis here (hip/thigh/calf, x/y/z), so slots 4+ are never reached.
SERIES = ("#2a78d6", "#008300", "#e87ba4")
# One colour per leg (FL, RL, FR, RR) for the hardware bag plots. quad-sdk's own CMU colours put
# FL red next to FR green, which colour-blind readers cannot separate.
LEG_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e0e0dc"


def style_axes(ax, title, xlabel, ylabel):
    """Recessive grid and axes; all text in ink tokens, never a series colour."""
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---- rollout reports ---------------------------------------------------------------------------


def plot_speed(out_dir, t, lin, cmd, err_x):
    """Forward speed against its command.

    v_y is deliberately absent: the command holds it at zero, so it is tracking
    error rather than speed and lives in the lin_vel_error_y figure. err_x is the
    same series the X error figure plots, so the RMS quoted here matches it.
    """
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Command is a reference, not a peer series: muted and dashed, so the measured
    # series keeps the categorical slot.
    ax.plot(t, cmd[:, 0], color=INK_MUTED, linewidth=1.2, linestyle="--", label="commanded v_x")
    ax.plot(t, lin[:, 0], color=SERIES[0], linewidth=1.6, label="measured v_x")
    style_axes(ax, "Base linear velocity", "time (s)", "velocity (m/s)")
    # Headroom so the legend never sits on the trace.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    # Sits on the right end of the command line, labelling it the way the error
    # figures label their zero line.
    rms = float(np.sqrt((err_x**2).mean()))
    ax.annotate(
        f"RMS {rms:.3f}",
        xy=(t[-1], cmd[-1, 0]),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=2, loc="upper right")
    _save(fig, out_dir, "speed")


def plot_power(out_dir, t, p_mech, energy, cot):
    """Mechanical joint power over the rollout, with the run totals annotated.

    The dashed line is mean power -- the level whose area under it equals the total
    energy, so the annotated joules and the trace read as the same quantity.
    """
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    mean_p = float(p_mech.mean())
    ax.axhline(mean_p, color=INK_MUTED, linewidth=1.2, linestyle="--", label="mean power")
    ax.plot(t, p_mech, color=SERIES[0], linewidth=1.4, label="sum |tau * omega|")
    ax.set_ylim(0.0, float(p_mech.max()) * 1.30 or 1.0)  # power is non-negative; headroom for the totals
    style_axes(ax, "Mechanical joint power", "time (s)", "power (W)")
    ax.annotate(
        f"total {energy:.1f} J    CoT {cot:.3f}    mean {mean_p:.1f} W",
        xy=(0.995, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=2, loc="upper left")
    _save(fig, out_dir, "mech_power")


def plot_torques(out_dir, t, torque, joint_names):
    """Small multiples: one panel per leg, three joints each, shared y-scale."""
    legs = [(joint_names[i][:2], slice(i, i + 3)) for i in range(0, 12, 3)]
    lim = float(np.abs(torque).max()) * 1.08 or 1.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), facecolor=SURFACE, sharex=True, sharey=True)
    flat = axes.ravel()
    for idx, (ax, (leg, sl)) in enumerate(zip(flat, legs)):
        ax.set_facecolor(SURFACE)
        for k, joint in enumerate(("hip", "thigh", "calf")):
            ax.plot(t, torque[:, sl][:, k], color=SERIES[k], linewidth=1.4, label=joint)
        ax.set_ylim(-lim, lim)
        # Axes are shared, so label only the outer edges.
        style_axes(
            ax,
            f"{leg} leg",
            "time (s)" if idx >= 2 else "",
            "torque (N m)" if idx % 2 == 0 else "",
        )
    # One figure-level legend above the panels: identity is shared across all four,
    # and this keeps it off the data.
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        labelcolor=INK_MUTED,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.0),
    )
    fig.suptitle("Joint torque by leg", color=INK, fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "torques_by_leg.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_foot_forces(out_dir, t, ff, names):
    """Small multiples: one panel per foot, world-frame Fx/Fy/Fz, shared y-scale.

    ff: (steps, num_feet, 3) -- the MPM mud reaction on each foot (see
    envs.mud.MudFootForces). One panel per foot answers "each foot as a separate
    plot"; the three series answer "break down the xyz forces".
    """
    n = ff.shape[1]
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    lim = float(np.abs(ff).max()) * 1.08 or 1.0
    comps = ("F_x", "F_y", "F_z")
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow), facecolor=SURFACE, sharex=True, sharey=True)
    flat = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(flat):
        ax.set_facecolor(SURFACE)
        if idx >= n:
            ax.set_visible(False)  # odd foot count: hide the empty panel
            continue
        # Zero line: Fx/Fy swing about it, Fz (support) rides above it.
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
        for k in range(3):
            ax.plot(t, ff[:, idx, k], color=SERIES[k], linewidth=1.4, label=comps[k])
        ax.set_ylim(-lim, lim)
        # Shared axes: label only the outer edges.
        style_axes(
            ax,
            names[idx],
            "time (s)" if idx >= n - ncol else "",
            "force (N)" if idx % ncol == 0 else "",
        )
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        labelcolor=INK_MUTED,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.0),
    )
    fig.suptitle("Foot reaction force (world frame)", color=INK, fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "foot_forces.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_total_foot_forces(out_dir, t, ff):
    """Whole-robot reaction force: all four feet summed, one line per world-frame axis.

    ff: (steps, num_feet, 3), same data as foot_forces.png but collapsed across feet --
    this answers "what's the net push on the robot" rather than "what's each foot doing".
    Summed F_z should hover near the robot's weight (mg), since that's what holds it up;
    see mean_total_foot_fz_n in metrics for the same quantity as a scalar.
    """
    total = ff.sum(axis=1)  # (steps, 3)
    comps = ("F_x", "F_y", "F_z")
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    for k in range(3):
        ax.plot(t, total[:, k], color=SERIES[k], linewidth=1.6, label=comps[k])
    style_axes(ax, "Total foot reaction force (world frame, summed across feet)", "time (s)", "force (N)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3, loc="upper right")
    _save(fig, out_dir, "total_foot_forces")


def plot_phi(out_dir, t, phi, fresh):
    """Small multiples: phi's live estimate of each privileged latent dimension,
    one line per panel, plus a usage panel showing which steps were freshly
    recomputed vs held over (see stabilise_adaptation in eval_cfg.py, --phi-rate).

    phi: (steps, latent_dim). Dimensions mix unrelated quantities (mass, friction,
    mud coefficients, ...) on unrelated scales, so panels do NOT share a y-axis --
    unlike torques_by_leg / foot_forces, which are homogeneous units. Dimensions are
    labelled by index only: their composition (which privileged term is where) is
    training-run-specific and not reliably recoverable from the checkpoint.
    """
    n_dims = phi.shape[1]
    n = n_dims + 1  # + the usage panel
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.4 * nrow), facecolor=SURFACE, sharex=True)
    flat = np.atleast_1d(axes).ravel()

    panels = [("phi usage", fresh, True)] + [(f"e_t[{i}]", phi[:, i], False) for i in range(n_dims)]
    for idx, ax in enumerate(flat):
        ax.set_facecolor(SURFACE)
        if idx >= n:
            ax.set_visible(False)
            continue
        title, series, is_usage = panels[idx]
        if is_usage:
            ax.plot(t, series, color=SERIES[0], linewidth=1.4, drawstyle="steps-post")
            ax.set_ylim(-0.15, 1.15)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["held", "fresh"])
        else:
            ax.plot(t, series, color=SERIES[0], linewidth=1.4)
        style_axes(ax, title, "time (s)" if idx >= n - ncol else "", "")

    fig.suptitle("Adaptation estimate (phi)", color=INK, fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "phi_estimate.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_path(out_dir, pos, yaw0):
    """Top-down trace against the commanded ray, on equal axes.

    Equal aspect on purpose: it shows the path's true shape rather than magnifying a few
    centimetres of drift into a dramatic curve. The cross_track_error figure is the zoomed
    view of the same quantity.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rel = pos - pos[0]
    # reference ray, drawn to the along-track extent actually reached
    direction = np.array([np.cos(yaw0), np.sin(yaw0)])
    reach = float((rel @ direction).max())
    ax.plot([0.0, reach * direction[0]], [0.0, reach * direction[1]],
            color=INK_MUTED, linewidth=1.2, linestyle="--", label="commanded path")
    ax.plot(rel[:, 0], rel[:, 1], color=SERIES[0], linewidth=1.6, label="actual path")
    ax.plot(rel[0, 0], rel[0, 1], marker="o", markersize=5, color=INK, linestyle="none", label="start")
    ax.set_aspect("equal", adjustable="datalim")
    style_axes(ax, "Ground track", "along start heading (m)", "lateral (m)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="best")
    _save(fig, out_dir, "ground_track")


def plot_error(out_dir, t, err, title, ylabel, name, signed=False):
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    if signed:
        # Zero line is the target; make it readable without competing with the data.
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    # Single series: no legend box, the title names it.
    ax.plot(t, err, color=SERIES[0], linewidth=1.6)
    rms = float(np.sqrt((err**2).mean()))
    ax.annotate(
        f"RMS {rms:.3f}",
        xy=(0.995, 0.94),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color=INK,
    )
    style_axes(ax, title, "time (s)", ylabel)
    _save(fig, out_dir, name)


# ---- sphere drop bench ------------------------------------------------------------------------


def plot_drop_force_time(out_dir, data):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    for k, comp, color in zip(("fx", "fy", "fz"), ("F_x", "F_y", "F_z"), SERIES):
        ax.plot(data["t"], data[k], color=color, linewidth=1.6, label=comp)
    style_axes(ax, "Mud force on the sphere (world frame)", "time since release (s)", "force (N)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "force_time.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_drop_depth_time(out_dir, data):
    """Penetration (sphere bottom below the settled surface) and vertical velocity, stacked on one time axis."""
    fig, (ax_d, ax_v) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, facecolor=SURFACE)
    for ax in (ax_d, ax_v):
        ax.set_facecolor(SURFACE)
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax_d.plot(data["t"], data["depth"], color=SERIES[0], linewidth=1.6)
    ax_d.invert_yaxis()  # depth increases downward
    style_axes(ax_d, "Sphere penetration (bottom below mud surface)", "", "depth (m)")
    ax_v.plot(data["t"], data["vz"], color=SERIES[1], linewidth=1.6)
    style_axes(ax_v, "Sphere vertical velocity", "time since release (s)", "v_z (m/s)")
    fig.tight_layout()
    fig.savefig(out_dir / "depth_time.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_drop_force_depth(out_dir, data):
    """F_z vs. penetration from first contact on. Loading (sphere still moving down) and
    rebound/rest (moving up or stopped) are drawn separately; the gap between them is energy
    the mud absorbed.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    contact = data["depth"] > 0.0
    if contact.any():
        start = int(np.argmax(contact))
        depth, fz, vz = data["depth"][start:], data["fz"][start:], data["vz"][start:]
        # end of loading = the first step the sphere stops descending
        n_load = int(np.argmax(vz >= 0.0)) if (vz >= 0.0).any() else len(vz)
        ax.plot(depth[: n_load + 1], fz[: n_load + 1], color=SERIES[0], linewidth=1.8, label="loading (descending)")
        if n_load < len(depth):
            ax.plot(depth[n_load:], fz[n_load:], color=SERIES[2], linewidth=1.8, label="rebound / rest")
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    style_axes(ax, "Sphere force vs. penetration", "depth below mud surface (m)", "F_z (N)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "force_depth.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
