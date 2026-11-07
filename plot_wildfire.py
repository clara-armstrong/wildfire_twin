"""Plotting for the wildfire solver, kept out of the solver module.

Importing matplotlib costs real startup time per worker when environments are
vectorized, so the solver does not import it.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from modifiedsimplephysicalmodel import BURNED, BURNING, DESTROYED, UNBURNED, WildfireSim

state_cmap = ListedColormap(["#A8CFA5", "#E9A15B", "#9B91B8", "#55545C"])
state_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], state_cmap.N)


def burn_state_of(sim: WildfireSim, u, beta):
    # frames don't store wetness, so classify against the dry threshold
    return sim.burn_state(u, beta, u_pc=sim.u_pc0)


def plot(sim: WildfireSim, frames, filename: str = "simple_wildfire.png"):
    steps = sorted(frames)
    nrows, ncols = 3, len(steps)
    fig = plt.figure(figsize=(3.0 * ncols, 9.5))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.15)

    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            share = None if (r, c) == (0, 0) else axes[0, 0]
            axes[r, c] = fig.add_subplot(gs[r, c], sharex=share, sharey=share)

    extent = [sim.coords[0], sim.coords[-1], sim.coords[0], sim.coords[-1]]

    for col, step in enumerate(steps):
        u, beta = frames[step]
        im0 = axes[0, col].imshow(u.T, origin="lower", extent=extent,
                                  cmap="inferno", vmin=0, vmax=sim.p.u_max)
        im1 = axes[1, col].imshow(beta.T, origin="lower", extent=extent,
                                  cmap="YlGn", vmin=0, vmax=1)
        im2 = axes[2, col].imshow(burn_state_of(sim, u, beta).T, origin="lower",
                                  extent=extent, cmap=state_cmap, norm=state_norm,
                                  interpolation="nearest")
        # outline the assets so damage is readable
        for ax in axes[:, col]:
            ax.contour(sim.X, sim.Y, sim.asset_value, levels=[0.5],
                       colors="w", linewidths=0.8)
        axes[0, col].set_title(f"step {step}")

    axes[0, 0].set_ylabel("temperature")
    axes[1, 0].set_ylabel("fuel fraction")
    axes[2, 0].set_ylabel("burn state")
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([-100, 0, 100])
            ax.set_yticks([-100, 0, 100])
            ax.tick_params(labelsize=8)
            ax.tick_params(labelbottom=(r == nrows - 1), labelleft=(c == 0))

    fig.colorbar(im0, ax=list(axes[0, :]), shrink=0.85)
    fig.colorbar(im1, ax=list(axes[1, :]), shrink=0.85)
    cb2 = fig.colorbar(im2, ax=list(axes[2, :]), shrink=0.85,
                       ticks=[UNBURNED, BURNING, BURNED, DESTROYED])
    cb2.ax.set_yticklabels(["unburned", "burning", "burned", "destroyed"])
    cb2.ax.tick_params(length=0)
    cb2.outline.set_visible(False)

    p, sc = sim.p, sim.sc
    fig.text(0.5, 0.07,
             f"kappa={p.kappa}  eps={p.epsilon}  alpha={p.alpha}  q={p.q}  "
             f"wind=({sc.wind_speed}@{sc.wind_direction}deg)  "
             f"fuel={sc.background_fuel.name}",
             ha="center", fontsize=8, color="0.4")
    fig.savefig(filename, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return filename