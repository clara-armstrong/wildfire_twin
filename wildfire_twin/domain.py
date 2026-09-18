"""The discrete domain of the inverse problem: theta = (x0, y0, wind_speed,
wind_direction), the unknown the digital twin infers.

"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import LatentGridConfig, WildfireConfig
from .solver import Latent

VAR_NAMES = ("x0", "y0", "wind_speed", "wind_direction")


class LatentDomain:
    # Discretised latent parameter space with a prior.
    #
    # In plain terms: this builds one big table with a row per possible
    # guess (theta). E.g. with just 2 possible x0 values, 2 y0 values, and 1
    # choice each for wind speed/direction, the table would be
    #   row 0: (x0=-40, y0=-40, speed=1.0, dir=45)
    #   row 1: (x0=-40, y0=-30, speed=1.0, dir=45)
    #   row 2: (x0=-30, y0=-40, speed=1.0, dir=45)
    #   row 3: (x0=-30, y0=-30, speed=1.0, dir=45)
    # "index" is just the row number, and `InverseSolver` keeps one belief
    # score per row.

    def __init__(self, config: WildfireConfig, grid: LatentGridConfig):
        self.config = config
        self.grid = grid

        sigma = config.ignition_sigma
        self.var2values: Dict[str, np.ndarray] = {
            "x0": np.linspace(
                config.ignition_center_x - sigma,
                config.ignition_center_x + sigma,
                grid.n_x0,
            ),
            "y0": np.linspace(
                config.ignition_center_y - sigma,
                config.ignition_center_y + sigma,
                grid.n_y0,
            ),
            "wind_speed": np.asarray(grid.wind_speeds, dtype=float),
            "wind_direction": np.asarray(grid.wind_directions_deg, dtype=float),
        }

        self._shape = tuple(len(self.var2values[v]) for v in VAR_NAMES)
        self._n = int(np.prod(self._shape))

        # Dense table of every latent vector, row i == flat index i.
        mesh = np.meshgrid(*[self.var2values[v] for v in VAR_NAMES], indexing="ij")
        self._table = np.stack([m.ravel() for m in mesh], axis=1)

    # Sizing
    def __len__(self) -> int:
        return self._n

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def table(self) -> np.ndarray:
        # (n_states, 4) array of every latent vector.
        return self._table

    # Index <-> value
    def index2values(self, index: int) -> np.ndarray:
        return self._table[index]

    def values2index(self, values: Sequence[float]) -> int:
        # Nearest grid point to an arbitrary latent vector. Wind direction
        # is compared circularly, so 350 deg and 10 deg count as close.
        values = np.asarray(values, dtype=float)
        if values.shape != (len(VAR_NAMES),):
            raise ValueError(
                f"expected a latent vector of length {len(VAR_NAMES)}, got shape {values.shape}"
            )
        sub = []
        for k, name in enumerate(VAR_NAMES):
            grid_vals = self.var2values[name]
            if name == "wind_direction":
                d = np.abs((grid_vals - values[k] + 180.0) % 360.0 - 180.0)
            else:
                d = np.abs(grid_vals - values[k])
            sub.append(int(np.argmin(d)))
        return int(np.ravel_multi_index(tuple(sub), self._shape))

    def latent(self, index: int) -> Latent:
        x0, y0, v, theta = self._table[index]
        return Latent(x0=float(x0), y0=float(y0), wind_speed=float(v),
                      wind_direction_deg=float(theta))

    def latents(self) -> List[Latent]:
        return [self.latent(i) for i in range(self._n)]

    # Prior
    def prior(self) -> np.ndarray:
        # Uniform prior: ignition location per Eq. 5.3, and wind uniform
        # too (the thesis calls wind "notoriously chaotic," so uniform is
        # the honest default rather than assuming a known distribution).
        return np.full(self._n, 1.0 / self._n)

    # Marginals
    def marginal(self, weights: np.ndarray, name: str) -> np.ndarray:
        # Marginal distribution of one latent variable, summing out the
        # other three.
        if name not in VAR_NAMES:
            raise KeyError(f"unknown latent variable '{name}'; expected one of {VAR_NAMES}")
        axis = VAR_NAMES.index(name)
        w = np.asarray(weights, dtype=float).reshape(self._shape)
        other = tuple(a for a in range(len(self._shape)) if a != axis)
        return w.sum(axis=other)

    def summarise(self, weights: np.ndarray) -> Dict[str, Dict[str, float]]:
        # Posterior mean/std per latent variable, for render.py and
        # digital_asset.py's latent_summary() to display.
        out: Dict[str, Dict[str, float]] = {}
        for name in VAR_NAMES:
            vals = self.var2values[name]
            m = self.marginal(weights, name)
            m = m / m.sum()
            if name == "wind_direction":
                # Circular mean/std (wind direction wraps at 360 degrees, a
                # plain arithmetic mean would be wrong near the wrap point).
                rad = np.radians(vals)
                c, s = float((m * np.cos(rad)).sum()), float((m * np.sin(rad)).sum())
                mean = float(np.degrees(np.arctan2(s, c)) % 360.0)
                R = float(np.hypot(c, s))
                std = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0))))
            else:
                mean = float((m * vals).sum())
                std = float(np.sqrt(max((m * (vals - mean) ** 2).sum(), 0.0)))
            out[name] = {"mean": mean, "std": std}
        return out


def build_latent_domain(
    config: WildfireConfig, grid: LatentGridConfig = None
) -> LatentDomain:
    return LatentDomain(config, grid or LatentGridConfig())
