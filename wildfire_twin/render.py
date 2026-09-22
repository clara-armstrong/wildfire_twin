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
from scipy.ndimage import distance_transform_edt

from .config import FUEL_TYPES, WildfireConfig
from .digital_asset import WildfireDigitalAsset
from .solver import Latent, asset_regions, build_fuel_map, initial_temperature

# Colour maps and static helpers

# Categorical colour map for burn status  0=Unburned 1=Burning 2=Burned
_BURN_COLORS = ["#025D18", "#ff4d00", "#101112"]   # green / bright orange / dark grey (modified to be more accessible to the colorblind)
_BURN_CMAP = mcolors.ListedColormap(_BURN_COLORS)
_BURN_NORM = mcolors.BoundaryNorm([0, 1, 2, 3], _BURN_CMAP.N)

# Flat overlay colour for suppression-treated cells (see suppression_mask).
_SUPPRESSION_CMAP = mcolors.ListedColormap(["#4fc3f7"])

_BURN_LEGEND = [
    mpatches.Patch(color=_BURN_COLORS[0], label="Unburned"),
    mpatches.Patch(color=_BURN_COLORS[1], label="Burning"),
    mpatches.Patch(color=_BURN_COLORS[2], label="Burned"),
]


def _imshow(ax, data, cmap, vmin=None, vmax=None, norm=None, title="", extent=None):
    # Thin wrapper: imshow with origin='lower' and a colour-bar.
    #
    # extent: physical (x_min, x_max, y_min, y_max). Pass it whenever
    # something else has to land on the same axes in physical units (wind
    # arrows, fuel labels, a coarse risk map beside the fine grid);
    # leave it None for a bare panel in cell indices.
    im = ax.imshow(
        data.T,                # transpose so x→right, y→up
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        interpolation="nearest",
        extent=extent,
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


# Physical-coordinate overlays
#
# The helpers below exist because the panels they decorate are drawn with
# extent= in physical units (config.domain_min..domain_max), not cell
# indices. Anything drawn on top -- arrows, labels, contours -- has to be
# in the same units or it lands in the wrong place.


def domain_extent(config: WildfireConfig) -> Tuple[float, float, float, float]:
    # (x_min, x_max, y_min, y_max) for imshow. Square domain, so both axes
    # share the same bounds. Grid-independent, which is what lets a coarse
    # (16, 16) risk map sit beside a fine (304, 304) state field and line up.
    return (config.domain_min, config.domain_max, config.domain_min, config.domain_max)


def _physical_ticks(ax, config: WildfireConfig) -> None:
    # _imshow blanks the ticks; put back a minimal x/y scale for panels
    # drawn in physical coordinates.
    ticks = [config.domain_min, 0.0, config.domain_max]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=6, colors="#444444")


def _asset_overlay(ax, asset: np.ndarray, extent, color="white", linewidth=0.9) -> None:
    # Outline the protected asset blocks. The thesis overlays a grey imshow
    # of the asset map at alpha=0.3, which tints the whole panel (the zeros
    # are drawn too) and washes out the field underneath. A contour marks
    # the same boundary without touching the colours it sits on.
    if asset is None or not np.any(asset > 0):
        return
    ax.contour(
        asset.T,
        levels=[0.5],
        colors=[color],
        linewidths=[linewidth],
        extent=extent,
        origin="lower",
    )


def _wind_quiver(
    ax,
    config: WildfireConfig,
    speed: float,
    direction_deg: float,
    extent,
    n_arrows: int = 6,
    color: str = "white",
    alpha: float = 0.55,
    label: bool = True,
) -> None:
    # Uniform wind field (thesis Eq. 5.2), so every arrow is identical --
    # the grid of them reads as direction, and the length as speed. This is
    # the honest analogue of the thesis's wind_field(X, Y, t) quiver: there
    # is no spatial variation to show until slope_speed is wired in.
    #
    # scale_units="width" with scale=10 means |v| = 1 draws as a tenth of
    # the panel width, so speeds are comparable across figures.
    x_min, x_max, y_min, y_max = extent
    xs = np.linspace(x_min, x_max, n_arrows + 2)[1:-1]
    ys = np.linspace(y_min, y_max, n_arrows + 2)[1:-1]
    X, Y = np.meshgrid(xs, ys, indexing="xy")   # "xy" here: quiver wants plot order

    theta = np.radians(direction_deg)
    u = np.full_like(X, speed * np.cos(theta))
    v = np.full_like(Y, speed * np.sin(theta))

    ax.quiver(
        X, Y, u, v,
        color=color, alpha=alpha,
        angles="xy", scale_units="width", scale=10.0,
        width=0.006,
    )
    if label:
        ax.text(
            0.02, 0.02, f"wind {speed:.2f} @ {direction_deg:.0f}°",
            transform=ax.transAxes, fontsize=6, color=color, alpha=0.9,
            family="monospace", va="bottom", ha="left",
        )


def _resolve_wind(wind) -> Optional[Tuple[float, float]]:
    # Accept a Latent, a (speed, direction_deg) pair, or a latent_summary()
    # dict, so callers can pass the true wind, a posterior mean, or nothing.
    if wind is None:
        return None
    if isinstance(wind, Latent):
        return float(wind.wind_speed), float(wind.wind_direction_deg)
    if isinstance(wind, dict):
        return (
            float(wind["wind_speed"]["mean"]),
            float(wind["wind_direction"]["mean"]),
        )
    speed, direction = wind
    return float(speed), float(direction)


def suppression_mask(
    ignition_threshold: np.ndarray, config: WildfireConfig, tol: float = 1e-3
) -> np.ndarray:
    # Cells whose ignition threshold has been raised above the baseline for
    # their fuel type -- i.e. where a water drop or firebreak has landed.
    #
    # The thesis masks on `u_pc > 4.`, an absolute cutoff. That can't work
    # here: _enforce_bounds clips u_pc to [2.5, 4.0], so nothing ever
    # exceeds 4, and the mask would always be empty. Comparing against the
    # per-cell baseline from build_fuel_map is what actually isolates the
    # treated cells, and it stays correct if FUEL_TYPES changes.
    _, _, baseline, _ = build_fuel_map(config)
    return np.asarray(ignition_threshold) > baseline + tol


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


# Setup figure: the domain before the fire runs
#
# Adapted from the thesis's plot_initial_domain. Differences that matter:
# labels are placed from the actual fuel masks rather than hard-coded
# coordinates, so they follow build_fuel_map instead of drifting from it;
# the β/u_pc values in the label text come from FUEL_TYPES for the same
# reason; and the wind is one uniform vector, not a field.


def _fuel_label_positions(config: WildfireConfig) -> List[Tuple[float, float, str]]:
    # (x, y, text) for one label per fuel type, at the centroid of that
    # type's mask in physical coordinates. Assets get one label per block,
    # since the two blocks are on opposite corners and a single centroid
    # would land between them, on grass.
    fuel_type, _, _, _ = build_fuel_map(config)
    coords = config.coords
    out: List[Tuple[float, float, str]] = []

    for k, props in FUEL_TYPES.items():
        text = (
            f"{props.name}\n"
            f"$\\beta={props.fuel_fraction}$\n"
            f"$u_{{pc}}={props.ignition_threshold}$"
        )
        if props.name == "Assets":
            for i0, i1, j0, j1 in asset_regions(config):
                out.append((
                    float(coords[(i0 + i1) // 2]),
                    float(coords[(j0 + j1) // 2]),
                    text,
                ))
            continue

        mask = fuel_type == k
        if not np.any(mask):
            continue
        i, j = _label_anchor(mask)
        out.append((float(coords[i]), float(coords[j]), text))
    return out


def _label_anchor(mask: np.ndarray) -> Tuple[int, int]:
    # Deepest interior cell of *mask*: the cell furthest from anything
    # outside it. A plain centroid isn't safe here -- an asset block sits
    # inside the shrub region, so the shrub centroid lands on top of the
    # asset block and its label collides with the asset label.
    dist = distance_transform_edt(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]
    i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(i), int(j)


def render_initial_domain(
    config: WildfireConfig,
    latent: Optional[Latent] = None,
    figsize: Tuple[float, float] = (15, 5),
) -> Figure:
    # Three-panel setup figure: ignition seed + wind, fuel fraction with
    # per-type labels, and ignition threshold. Everything is derived from
    # *config* (and *latent*, for the ignition centre and wind), so this
    # can be called before any solver exists.
    extent = domain_extent(config)
    if latent is None:
        latent = Latent(
            x0=config.ignition_center_x, y0=config.ignition_center_y,
            wind_speed=config.wind_speed,
            wind_direction_deg=config.wind_direction_deg,
        )

    _, beta, u_pc, asset = build_fuel_map(config)
    u0 = initial_temperature(config, latent.x0, latent.y0)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#f2f2f7")
    for ax in axes:
        ax.set_facecolor("#16213e")

    # [0] Ignition seed + wind
    im0 = _imshow(
        axes[0], u0, cmap="inferno", vmin=0, vmax=8,
        title="Initial Temperature and Wind Field", extent=extent,
    )
    _colorbar(fig, axes[0], im0, label="nondimensional u")
    _wind_quiver(
        axes[0], config, latent.wind_speed, latent.wind_direction_deg, extent
    )
    _asset_overlay(axes[0], asset, extent)

    # [1] Fuel fraction, labelled per fuel type
    im1 = _imshow(
        axes[1], beta, cmap="YlGn", vmin=0, vmax=1,
        title="Fuel Fraction (β) by Fuel Type", extent=extent,
    )
    _colorbar(fig, axes[1], im1, label="fraction β")
    _asset_overlay(axes[1], asset, extent, color="black")
    span = config.domain_max - config.domain_min
    for x, y, text in _fuel_label_positions(config):
        # Anchors sit at the deepest point of each region, which for a thin
        # region (the shrub strip below the asset block) is close to the
        # domain edge. Grow the text inward from the anchor rather than
        # centring it there, so it stays on the panel.
        fx = (x - config.domain_min) / span
        fy = (y - config.domain_min) / span
        ha = "left" if fx < 0.15 else "right" if fx > 0.85 else "center"
        va = "bottom" if fy < 0.15 else "top" if fy > 0.85 else "center"
        axes[1].text(
            x, y, text, color="black", fontsize=7, ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.65,
                      edgecolor="none"),
        )

    # [2] Ignition threshold
    im2 = _imshow(
        axes[2], u_pc, cmap="cividis", vmin=2.5, vmax=4.0,
        title="Ignition Threshold ($u_{pc}$)", extent=extent,
    )
    _colorbar(fig, axes[2], im2, label="$u_{pc}$")
    _asset_overlay(axes[2], asset, extent)

    for ax in axes:
        _physical_ticks(ax, config)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.title.set_color("black")

    fig.tight_layout()
    return fig


# Evolution figure: several timesteps side by side
#
# Adapted from the thesis's plot_evolution. Its bottom row is
# np.mean(temp_fields[i], axis=0) -- an ensemble mean over sampled
# trajectories. ForwardCache stores only coarse burn indicators and scalar
# QoIs (inverse.py), so there is no temperature ensemble to average here;
# the bottom row is the ground-truth field instead, and the posterior
# spread lives in the risk map above it.


def render_evolution(
    states: List[Dict[str, np.ndarray]],
    steps: List[int],
    config: WildfireConfig,
    risk_maps: Optional[List[np.ndarray]] = None,
    wind=None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    # A 2 x len(states) grid: risk (or burn status) on top, temperature
    # with suppression overlay below, one column per timestep.
    #
    # states: fine-grid state dicts, e.g. WildfirePhysicalAsset.state_dict
    #   or solver.simulate(...)["frames"], one per timestep.
    # steps: step number per column, used in the titles.
    # risk_maps: coarse posterior risk maps captured at those steps (from
    #   digital_asset.risk_map(step)). Captured as each step was current --
    #   recomputing them from the final posterior would show the same map
    #   in every column.
    # wind: a Latent, a (speed, direction_deg) pair, a latent_summary()
    #   dict, or a list of any of those (one per column).
    if not states:
        raise ValueError("states list is empty")
    if len(steps) != len(states):
        raise ValueError(f"got {len(states)} states but {len(steps)} steps")
    if risk_maps is not None and len(risk_maps) != len(states):
        raise ValueError(
            f"got {len(states)} states but {len(risk_maps)} risk maps"
        )

    n = len(states)
    extent = domain_extent(config)
    if figsize is None:
        figsize = (4.2 * n, 8.4)

    # Per-column wind: a bare pair like (1.0, 45.0) is a single wind, not a
    # two-column list, so only unwrap when the length matches the columns.
    if isinstance(wind, list) and len(wind) == n:
        winds = [_resolve_wind(w) for w in wind]
    else:
        winds = [_resolve_wind(wind)] * n

    # squeeze=False keeps axes 2-D for n == 1. Note the index order:
    # axes[row, column], so time varies along the second index -- the
    # thesis's axes[i, 0]/axes[i, 1] indexes rows by time and breaks for
    # more than two timesteps.
    fig, axes = plt.subplots(2, n, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor("#f2f2f7")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    for i, (state, step) in enumerate(zip(states, steps)):
        ax_top, ax_bot = axes[0, i], axes[1, i]
        asset = state.get("asset_importance")

        # Top: posterior risk on the coarse grid, or burn status if the
        # digital asset wasn't run. Both share the physical extent, so the
        # coarse map aligns with the fine field below it.
        if risk_maps is not None:
            im_top = _imshow(
                ax_top, np.asarray(risk_maps[i]), cmap="YlOrRd", vmin=0, vmax=1,
                title=f"Posterior Risk — step {step}", extent=extent,
            )
            wind_color = "#222222"     # dark ink on the light YlOrRd map
            if i == n - 1:
                _colorbar(fig, ax_top, im_top, label="burn probability")
        else:
            _imshow(
                ax_top, state["burn_status"].astype(float),
                cmap=_BURN_CMAP, norm=_BURN_NORM,
                title=f"Burn Status — step {step}", extent=extent,
            )
            wind_color = "white"       # the burn palette is dark throughout
            if i == n - 1:
                ax_top.legend(handles=_BURN_LEGEND, loc="lower right",
                              fontsize=6, framealpha=0.5)
        if winds[i] is not None:
            _wind_quiver(ax_top, config, winds[i][0], winds[i][1], extent,
                         color=wind_color, alpha=0.65)
        _asset_overlay(ax_top, asset, extent, color="black")

        # Bottom: ground-truth temperature, with the burning perimeter and
        # any suppression-treated cells on top.
        im_bot = _imshow(
            ax_bot, state["temperature"], cmap="inferno", vmin=0, vmax=8,
            title=f"Temperature — step {step}", extent=extent,
        )
        if i == n - 1:
            _colorbar(fig, ax_bot, im_bot, label="nondimensional u")

        # Suppression-treated cells, in a flat bright colour. A "Blues"
        # ramp reads as near-black at its top end, which is invisible on
        # the dark low end of "inferno" underneath.
        treated = suppression_mask(state["ignition_threshold"], config)
        if np.any(treated):
            ax_bot.imshow(
                np.where(treated, 1.0, np.nan).T,
                origin="lower", extent=extent,
                cmap=_SUPPRESSION_CMAP, vmin=0, vmax=1,
                alpha=0.6, interpolation="nearest",
            )

        burn = state["burn_status"]
        if np.any(burn == 1):
            ax_bot.contour(
                (burn == 1).T.astype(float), levels=[0.5],
                colors=["#ffd54a"], linewidths=[0.9],
                extent=extent, origin="lower",
            )
        _asset_overlay(ax_bot, asset, extent)

    for ax in axes.flat:
        _physical_ticks(ax, config)
        ax.title.set_color("black")
    for ax in axes[:, 0]:
        ax.set_ylabel("y", fontsize=7)
    for ax in axes[1, :]:
        ax.set_xlabel("x", fontsize=7)

    fig.tight_layout()
    return fig


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
