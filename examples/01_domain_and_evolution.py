"""
Draws the two survey figures from render.py: the domain before ignition, and
the fire at several timesteps.

Solver-only, like 00_test_render.py -- no inference, no forward cache, so it
starts immediately. That means the evolution figure's top row is ground-truth
burn status rather than the posterior risk map
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildfire_twin.config import WildfireConfig
from wildfire_twin.render import render_evolution, render_initial_domain
from wildfire_twin.solver import Latent, simulate

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 304x304 grid + q=1
CONFIG = WildfireConfig(grid_size=304, q=1.0, dt=0.02, max_steps=600)
LATENT = Latent(
    x0=CONFIG.ignition_center_x, y0=CONFIG.ignition_center_y,
    wind_speed=3, wind_direction_deg=45.0,
)

RECORD_AT = [0, 200, 400, 600]


def main():
    fig = render_initial_domain(CONFIG, LATENT)
    path = OUTPUT_DIR / "initial_domain.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path}")

    result = simulate(CONFIG, LATENT, max(RECORD_AT), record_at=RECORD_AT)

    fig = render_evolution(result["frames"], RECORD_AT, CONFIG, wind=LATENT)
    path = OUTPUT_DIR / "evolution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path}")



if __name__ == "__main__":
    main()
