"""Rendering utilities for the wildfire digital twin.

"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from .digital_asset import WildfireDigitalAsset

# Colour maps and static helpers

# Categorical colour map for burn status  0=Unburned 1=Burning 2=Burned
_BURN_COLORS = ["#025D18", "#ff4d00", "#101112"]   # green / bright orange / dark grey (modified to be more accessible to the colorblind)
_BURN_CMAP = mcolors.ListedColormap(_BURN_COLORS)
_BURN_NORM = mcolors.BoundaryNorm([0, 1, 2, 3], _BURN_CMAP.N)

_BURN_LEGEND = [
    mpatches.Patch(color=_BURN_COLORS[0], label="Unburned"),
    mpatches.Patch(color=_BURN_COLORS[1], label="Burning"),
    mpatches.Patch(color=_BURN_COLORS[2], label="Burned"),
]


def _imshow(ax, data, cmap, vmin=None, vmax=None, norm=None, title=""):
    # Thin wrapper: imshow with origin='lower' and a colour-bar.
    im = ax.imshow(
        data.T,                # transpose so x→right, y→up
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _colorbar(fig, ax, im, label="", ticks=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=7)
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


# Main render function


def render_state(
    physical_state: Dict[str, np.ndarray],
    risk_map: np.ndarray,
    prior_risk_map: np.ndarray,
    qois: Optional[Dict] = None,
    latent_summary: Optional[Dict[str, Dict[str, float]]] = None,
    step: Optional[int] = None,
    action_name: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> Figure:
    # Render a single timestep as a 2x3 multi-panel figure.
    #
    # physical_state: WildfirePhysicalAsset.state_dict (fine grid).
    # risk_map/prior_risk_map: coarse-grid posterior/prior burn-probability
    #   maps, from digital_asset.risk_map(step)/prior_risk_map(step).
    # qois/latent_summary: optional dicts from get_qois()/latent_summary();
    #   add the summary text box when supplied.
    # step/action_name: shown in the figure title.
    g = physical_state["temperature"].shape[0]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.patch.set_facecolor("#f2f2f7")
    for row in axes:
        for ax in row:
            ax.set_facecolor("#16213e")

    # Row 0: physical state

    # [0,0] Temperature
    im00 = _imshow(
        axes[0, 0],
        physical_state["temperature"],
        cmap="inferno",
        vmin=0, vmax=8,
        title="Temperature (u)",
    )
    _colorbar(fig, axes[0, 0], im00, label="nondimensional")

    # [0,1] Burn Status
    # Burn status is categorical, so it gets a legend instead of a colorbar.
    _imshow(
        axes[0, 1],
        physical_state["burn_status"].astype(float),
        cmap=_BURN_CMAP,
        norm=_BURN_NORM,
        title="Burn Status",
    )
    axes[0, 1].legend(
        handles=_BURN_LEGEND,
        loc="lower right",
        fontsize=6,
        framealpha=0.5,
    )

    # [0,2] Fuel Fraction
    im02 = _imshow(
        axes[0, 2],
        physical_state["fuel_fraction"],
        cmap="YlGn",
        vmin=0, vmax=1,
        title="Fuel Fraction (β)",
    )
    _colorbar(fig, axes[0, 2], im02, label="fraction remaining")

    # Overlay asset-importance contour on fuel map
    asset = physical_state.get("asset_importance", np.zeros((g, g)))
    if np.any(asset > 0):
        axes[0, 2].contour(
            asset.T,
            levels=[0.5],
            colors=["black"],
            linewidths=[0.8],
            alpha=0.7,
        )

    # Row 1: digital / posterior state (coarse grid)

    # [1,0] Posterior risk map
    im10 = _imshow(
        axes[1, 0],
        np.asarray(risk_map),
        cmap="YlOrRd",
        vmin=0, vmax=1,
        title="Posterior Risk Map",
    )
    _colorbar(fig, axes[1, 0], im10, label="burn probability")

    # [1,1] Prior risk map: the thesis's static baseline, for comparison
    im11 = _imshow(
        axes[1, 1],
        np.asarray(prior_risk_map),
        cmap="YlOrRd",
        vmin=0, vmax=1,
        title="Prior Risk Map (baseline)",
    )
    _colorbar(fig, axes[1, 1], im11, label="burn probability")

    # [1,2] Asset Importance + burn overlay
    im12 = _imshow(
        axes[1, 2],
        asset,
        cmap="Purples",
        vmin=0, vmax=1,
        title="Asset Importance",
    )
    _colorbar(fig, axes[1, 2], im12, label="priority weight")

    # Overlay burn perimeter
    burn = physical_state.get("burn_status", np.zeros((g, g), dtype=int))
    if np.any(burn == 1):
        axes[1, 2].contour(
            (burn == 1).T.astype(float),
            levels=[0.5],
            colors=["#ff6f00"],
            linewidths=[1.0],
        )

    # Figure-level title and annotation
    title_parts = ["Wildfire Digital Twin"]
    if step is not None:
        title_parts.append(f"– step {step}")
    if action_name:
        title_parts.append(f"| action: {action_name}")
    fig.suptitle(
        "  ".join(title_parts),
        fontsize=11,
        color="black",
        y=.95,
    )

    # Row labels
    axes[0, 0].set_ylabel("Physical asset", fontsize=8, color="#aaaaaa", labelpad=4)
    axes[1, 0].set_ylabel("Digital asset", fontsize=8, color="#aaaaaa", labelpad=4)
    for ax in axes.flat:
        ax.title.set_color("black")
        ax.yaxis.label.set_color("#000000")

    # QoI / latent summary text box
    lines: List[str] = []
    if qois:
        lines += [
            f"Burnt fraction: {qois.get('burnt_fraction_mean', 0):.1%}"
            f" ± {qois.get('burnt_fraction_std', 0):.1%}",
            f"Asset damage:   {qois.get('asset_damage_mean', 0):.3f}"
            f"  [{qois.get('asset_damage_lo90', 0):.3f},"
            f" {qois.get('asset_damage_hi90', 0):.3f}]",
            f"Eff. samples:   {qois.get('effective_sample_size', 0):.1f}",
            f"Observations:   {int(qois.get('n_assimilated', 0))}",
        ]
    if latent_summary:
        x0 = latent_summary.get("x0", {})
        y0 = latent_summary.get("y0", {})
        ws = latent_summary.get("wind_speed", {})
        wd = latent_summary.get("wind_direction", {})
        lines += [
            f"Ignition (x0,y0): ({x0.get('mean', 0):.1f}, {y0.get('mean', 0):.1f})"
            f"  ±({x0.get('std', 0):.1f}, {y0.get('std', 0):.1f})",
            f"Wind: {ws.get('mean', 0):.2f} @ {wd.get('mean', 0):.0f}°"
            f"  (±{ws.get('std', 0):.2f}, ±{wd.get('std', 0):.0f}°)",
        ]
    # tight_layout doesn't know about the fig.text box below, so reserve
    # headroom for it explicitly; otherwise it overlaps the top row's
    # titles, growing worse as lines are added (qois + latent_summary).
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90 if lines else 0.96))

    if lines:
        textstr = "\n".join(lines)
        props = dict(boxstyle="round", facecolor="#0f3460", alpha=0.8)
        fig.text(
            0.99, 0.98, textstr,
            transform=fig.transFigure,
            fontsize=7.5,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
            color="white",
            family="monospace",
        )

    return fig


def render_digital_twin(
    physical_state: Dict[str, np.ndarray],
    digital_asset: WildfireDigitalAsset,
    step: int,
    action_name: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> Figure:
    # Convenience wrapper around render_state for a live digital asset.
    # step must be one of digital_asset's cached risk steps (see
    # inverse.ForwardCache.risk_steps).
    return render_state(
        physical_state,
        digital_asset.risk_map(step),
        digital_asset.prior_risk_map(step),
        qois=digital_asset.get_qois(step),
        latent_summary=digital_asset.latent_summary(),
        step=step,
        action_name=action_name,
        figsize=figsize,
    )


# Convenience: save a single frame to disk

def save_frame(
    physical_state: Dict[str, np.ndarray],
    risk_map: np.ndarray,
    prior_risk_map: np.ndarray,
    path: str,
    **kwargs,
) -> None:
    # Render and save a single frame to *path* (PNG, PDF, ...).
    fig = render_state(physical_state, risk_map, prior_risk_map, **kwargs)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# Animation

# One frame = (physical_state, risk_map, prior_risk_map), all already
# computed for that timestep. Live digital-asset state isn't reusable here
# because its posterior mutates as the run progresses; each historical
# frame needs its risk maps captured (e.g. via ``digital_asset.risk_map``)
# at the moment it was current, not recomputed from the final posterior.
Frame = Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]


def animate(
    frames: List[Frame],
    qois_list: Optional[List[Dict]] = None,
    latent_summaries: Optional[List[Dict]] = None,
    interval: int = 200,
    figsize: Tuple[float, float] = (14, 8),
) -> FuncAnimation:
    # Create a FuncAnimation from a list of per-timestep frames.
    #
    # frames: list of (physical_state, risk_map, prior_risk_map) tuples.
    # qois_list/latent_summaries: optional, one dict per frame.
    # interval: milliseconds between frames.
    #
    # Call .save("out.gif", writer="pillow") or .save("out.mp4",
    # writer="ffmpeg") on the result.
    if not frames:
        raise ValueError("frames list is empty")

    # Render the first frame to create the figure and axes
    qois0 = qois_list[0] if qois_list else None
    latent0 = latent_summaries[0] if latent_summaries else None
    fig = render_state(
        *frames[0], qois=qois0, latent_summary=latent0, step=0, figsize=figsize
    )

    def _update(idx):
        # Clear and re-draw each frame by replacing figure content.
        # This is simpler than managing individual artist references for a
        # grid of images with colour-bars.
        fig.clf()
        qois = qois_list[idx] if qois_list else None
        latent = latent_summaries[idx] if latent_summaries else None
        _draw_into(fig, frames[idx], qois=qois, latent_summary=latent, step=idx)

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=interval,
        blit=False,
    )
    return anim


def _draw_into(
    fig: Figure,
    frame: Frame,
    qois: Optional[Dict] = None,
    latent_summary: Optional[Dict] = None,
    step: Optional[int] = None,
) -> None:
    # Re-use *fig* for animation: same layout as render_state but in-place.
    # Delegate to render_state on a fresh figure, then transfer the axes;
    # this keeps all layout logic in one place.
    tmp = render_state(
        *frame, qois=qois, latent_summary=latent_summary, step=step,
        figsize=fig.get_size_inches(),
    )
    # Copy content axes into fig
    for ax in tmp.get_axes():
        ax.remove()
        ax.figure = fig
        fig.add_axes(ax)
    plt.close(tmp)
