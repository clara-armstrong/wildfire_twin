"""
Runs the plain PDE solver (solver.Wildfire) directly; no digital twin /
forward cache needed for a single ground-truth trajectory.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildfire_twin.config import WildfireConfig
from wildfire_twin.solver import Latent, Wildfire

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = WildfireConfig(grid_size=350, q=1.0, dt=0.02, max_steps=250)
LATENT = Latent(
    x0=CONFIG.ignition_center_x, y0=CONFIG.ignition_center_y,
    wind_speed=1.2, wind_direction_deg=45.0,
)

BG = "#0b0f14"
INK = "#e8ecf1"
MUTED = "#7b8794"
UNBURNED_COLOR = "#274b32"
BURNED_COLOR = "#dc551b"
PERIMETER_COLOR = "#ffd54a"
FIRE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fire_glow", ["#3b0f0f", "#d11b0e", "#ff7a1a", "#ffd54a", "#fff6d0"]
)


def render_hero(fire: Wildfire, title: str, subtitle: str, path: Path) -> None:
    g = fire.config.grid_size

    rgb = np.empty((g, g, 3))
    rgb[:] = mcolors.to_rgb(UNBURNED_COLOR)
    rgb[fire.burn_status == 2] = mcolors.to_rgb(BURNED_COLOR)
    heat = np.clip(fire.u / 8.0, 0, 1)
    rgb[fire.burn_status == 1] = FIRE_CMAP(heat)[..., :3][fire.burn_status == 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    ax.imshow(rgb.transpose(1, 0, 2), origin="lower")

    active = (fire.burn_status > 0).astype(float)
    if np.any(active):
        ax.contour(active.T, levels=[0.5], colors=[PERIMETER_COLOR], linewidths=1.5)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, color=INK, fontsize=20, fontweight="bold", loc="left", pad=12)
    ax.text(0.0, 1.03, subtitle, transform=ax.transAxes, color=MUTED, fontsize=10, family="monospace")

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path}")


def main():
    fire = Wildfire(CONFIG, LATENT)
    total = CONFIG.grid_size ** 2
    pending = {"ignition", "spread", "contained"}

    for step in range(1, CONFIG.max_steps + 1):
        fire.step()
        burnt_frac = np.count_nonzero(fire.burn_status > 0) / total
        subtitle = f"step {step}  ·  burnt {burnt_frac:.1%}"
        at_end = fire.contained or step == CONFIG.max_steps

        # Independent checks (not elif): each condition can fire on its own
        # step, and "contained"/end-of-run must still fire even if "spread"
        # never crossed its threshold.
        if "ignition" in pending and step >= 5:
            pending.discard("ignition")
            render_hero(fire, "Ignition", subtitle, OUTPUT_DIR / "hero_ignition.png")
        if "spread" in pending and step >= CONFIG.max_steps // 2:
            pending.discard("spread")
            render_hero(fire, "Active Spread", subtitle, OUTPUT_DIR / "hero_spread.png")
        if "contained" in pending and at_end:
            pending.discard("contained")
            render_hero(fire, "Contained", subtitle, OUTPUT_DIR / "hero_contained.png")

        if at_end:
            break

    if pending:
        print(f"Warning: run ended before reaching checkpoint(s): {sorted(pending)}")


if __name__ == "__main__":
    main()