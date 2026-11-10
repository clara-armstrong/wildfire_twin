"""
Simple Wildfire PDE model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

UNBURNED, BURNING, BURNED, DESTROYED = 0, 1, 2, 3


@dataclass(frozen=True)
class SolverParams:
    grid_size: int = 128
    domain: float = 150.0
    dt: float = 0.05

    kappa: float = 1.0
    epsilon: float = 0.3
    q: float = 4.0
    alpha: float = 0.001

    u_max: float = 8.0

    @property
    def dx(self) -> float:
        return 2 * self.domain / (self.grid_size - 1)


@dataclass(frozen=True)
class FuelType:

    name: str
    fuel_fraction: float
    ignition_threshold: float


FOREST = FuelType("forest", 0.9, 3.5)
GRASS = FuelType("grass", 0.5, 2.0)
SHRUB = FuelType("shrub", 0.7, 3.0)
STRUCTURE = FuelType("structure", 0.8, 4.5)
WATER = FuelType("water", 0, np.inf)


@dataclass(frozen=True)
class Scenario:
    wind_speed: float = 3.0
    wind_direction: float = 45.0  # degrees, from +x axis

    background_fuel: FuelType = FOREST
    # axis-aligned fuel patches over the background, painted in order:
    # (cx, cy, half_x, half_y, fuel)
    fuel_patches: Tuple[Tuple[float, float, float, float, FuelType], ...] = (
        (0.0, 150.0, np.inf, 60.0, GRASS),  # full-width strip along the top edge
        (0.0, 85.0, np.inf, 4.0, WATER),  # creek along the grass edge
        (20.0, -60.0, 45.0, 20.0, SHRUB),  # shrub belt across the south
        (-70.0, 50.0, 25.0, 20.0, SHRUB),
        (110.0, 20.0, 15.0, 35.0, SHRUB),
        (-45.0, -45.0, 25.0, 25.0, GRASS),  # meadow around the ignition zone
        (45.0, -35.0, 15.0, 12.0, SHRUB),  # brush leading into the east asset
        (5.0, 45.0, 15.0, 15.0, GRASS),  # clearing beside the north asset
        (-115.0, -10.0, 15.0, 25.0, GRASS),
        (90.0, 65.0, 6.0, 6.0, STRUCTURE),  # outbuildings: burnable, no asset value
        (-60.0, 100.0, 6.0, 6.0, STRUCTURE),
        (100.0, 120.0, 20.0, 10.0, WATER),  # pond in the meadow
        (0.0, 0.0, 20.0, 20.0, WATER),  # lake at the center, painted last
    )
    # axis-aligned asset squares: (cx, cy, half_width, value)
    asset_squares: Tuple[Tuple[float, float, float, float], ...] = (
        (40.0, 40.0, 12.0, 1.0),
        (70.0, -10.0, 12.0, 1.0),
    )
    asset_fuel: FuelType = STRUCTURE

    # burn-state thresholds
    burned_fuel_loss: float = 1e-4
    overwhelming_fuel_loss: float = 0.99


    @property
    def wind_vec(self) -> Tuple[float, float]:
        th = np.radians(self.wind_direction)
        return self.wind_speed * np.cos(th), self.wind_speed * np.sin(th)


@dataclass(frozen=True)
class DropParams:
    """Geometry and effect of one water drop.
    """

    half_length: float = 20.0
    half_width: float = 5.0
    u_damping: float = 0.5        # u <- u * u_damping inside the footprint (between 0.0 complete dampening and 1.0)
    u_pc_boost: float = 3.0
    wetness_halflife: float = 5.0  # in time units, inf for permanent


@dataclass
class Ignition:
    x: float = -35.0
    y: float = -35.0
    amplitude: float = 5.0
    decay: float = 0.005

class WildfireSim:
    """Steppable wildfire simulation on a fixed grid."""

    def __init__(
        self,
        params: Optional[SolverParams] = None,
        scenario: Optional[Scenario] = None,
        drop_params: Optional[DropParams] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.p = params or SolverParams()
        self.sc = scenario or Scenario()
        self.dp = drop_params or DropParams()
        self.rng = rng or np.random.default_rng()

        n = self.p.grid_size
        self.coords = np.linspace(-self.p.domain, self.p.domain, n)
        self.X, self.Y = np.meshgrid(self.coords, self.coords, indexing="ij")

        self.beta0, self.u_pc0, self.asset_value = self._build_maps()

        # mutable fields
        self.u = np.zeros((n, n))
        self.beta = self.beta0.copy()
        self.wetness = np.zeros((n, n))  # multiplies dp.u_pc_boost
        self.t = 0.0
        self.n_steps = 0
        self.drop_log: List[Tuple[float, float, float, float]] = []  # (t, x, y, theta)

    # maps
    def _build_maps(self):
        n = self.p.grid_size
        beta0 = np.full((n, n), self.sc.background_fuel.fuel_fraction)
        u_pc0 = np.full((n, n), self.sc.background_fuel.ignition_threshold)
        value = np.zeros((n, n))

        # patches first, so assets sit on top of any overlap
        for cx, cy, half_x, half_y, fuel in self.sc.fuel_patches:
            m = (np.abs(self.X - cx) < half_x) & (np.abs(self.Y - cy) < half_y)
            beta0[m] = fuel.fuel_fraction
            u_pc0[m] = fuel.ignition_threshold

        for cx, cy, half, val in self.sc.asset_squares:
            m = (np.abs(self.X - cx) < half) & (np.abs(self.Y - cy) < half)
            beta0[m] = self.sc.asset_fuel.fuel_fraction
            u_pc0[m] = self.sc.asset_fuel.ignition_threshold
            value[m] = val

        return beta0, u_pc0, value

    @property
    def u_pc(self) -> np.ndarray:
        """Effective ignition threshold: base map plus current wetness."""
        return self.u_pc0 + self.dp.u_pc_boost * self.wetness

    # initial conditions 
    def reset(self, ignition: Optional[Ignition] = None):
        ig = ignition or Ignition()
        self.u = ig.amplitude * np.exp(
            -ig.decay * ((self.X - ig.x) ** 2 + (self.Y - ig.y) ** 2)
        )
        self.beta = self.beta0.copy()
        self.wetness[:] = 0.0
        self.t = 0.0
        self.n_steps = 0
        self.drop_log.clear()
        return self

    def sample_ignition(self) -> Ignition:
        """Random IC, for the stochastic-initial-condition experiments."""
        return Ignition(
            x=float(self.rng.uniform(-60, -20)),
            y=float(self.rng.uniform(-60, -20)),
            amplitude=float(self.rng.uniform(4.0, 6.0)),
            decay=float(self.rng.uniform(0.004, 0.006)),
        )

    # operators 
    def _laplacian(self, u):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
        ) / self.p.dx ** 2
        return lap

    def _advection(self, u):
        vx, vy = self.sc.wind_vec
        dx = self.p.dx
        dudx = np.zeros_like(u)
        dudy = np.zeros_like(u)
        if vx > 0:
            dudx[1:, :] = (u[1:, :] - u[:-1, :]) / dx
        else:
            dudx[:-1, :] = (u[1:, :] - u[:-1, :]) / dx
        if vy > 0:
            dudy[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
        else:
            dudy[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx
        return -(vx * dudx + vy * dudy)

    def _reaction(self, u, beta, u_pc):
        p = self.p
        H = (u >= u_pc).astype(np.float64)
        arrhenius = np.exp(u / (1.0 + p.epsilon * u))
        f_u = H * beta * arrhenius - p.alpha * u
        g_beta = -H * (p.epsilon / p.q) * beta * arrhenius
        return f_u, g_beta

    def _rhs(self, u, beta, u_pc):
        f_u, g_beta = self._reaction(u, beta, u_pc)
        du = self.p.kappa * self._laplacian(u) + self._advection(u) + f_u
        # pin the outer ring
        du[0, :] = du[-1, :] = du[:, 0] = du[:, -1] = 0.0
        g_beta[0, :] = g_beta[-1, :] = g_beta[:, 0] = g_beta[:, -1] = 0.0
        return du, g_beta

    # time stepping 
    def step_once(self):
        dt = self.p.dt
        u, b = self.u, self.beta
        u_pc = self.u_pc  # frozen across the RK4 stages

        k1u, k1b = self._rhs(u, b, u_pc)
        k2u, k2b = self._rhs(u + 0.5 * dt * k1u, b + 0.5 * dt * k1b, u_pc)
        k3u, k3b = self._rhs(u + 0.5 * dt * k2u, b + 0.5 * dt * k2b, u_pc)
        k4u, k4b = self._rhs(u + dt * k3u, b + dt * k3b, u_pc)

        self.u = np.clip(u + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u), 0, self.p.u_max)
        self.beta = np.clip(b + (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b), 0, 1)

        if np.isfinite(self.dp.wetness_halflife) and self.wetness.any():
            self.wetness *= 0.5 ** (dt / self.dp.wetness_halflife)

        self.t += dt
        self.n_steps += 1
        return self

    def advance(self, n_steps: int):
        for _ in range(n_steps):
            self.step_once()
        return self

    # intervention 
    def apply_drop(self, x: float, y: float, theta: float) -> np.ndarray:
        """Rectangular drop centred at (x, y), long axis at angle theta.

        The caller decides where. This method only executes.
        """
        c, s = np.cos(theta), np.sin(theta)
        dX, dY = self.X - x, self.Y - y
        along = dX * c + dY * s
        across = -dX * s + dY * c
        mask = (np.abs(along) <= self.dp.half_length) & (
            np.abs(across) <= self.dp.half_width
        )

        self.u[mask] *= self.dp.u_damping
        self.wetness[mask] = 1.0
        self.drop_log.append((self.t, x, y, theta))
        return mask

    # quantities of interest
    def burning_mask(self, u=None, beta=None, u_pc=None) -> np.ndarray:
        """Hot enough to burn AND still has fuel. A spent cell stays hot
        for a long time (alpha is small) but is no longer burning.
        Defaults to the current fields; pass saved ones to classify a frame.
        """
        u = self.u if u is None else u
        beta = self.beta if beta is None else beta
        u_pc = self.u_pc if u_pc is None else u_pc
        has_fuel = beta >= (1.0 - self.sc.overwhelming_fuel_loss) * self.beta0
        return (u >= u_pc) & has_fuel

    def burning_fraction(self) -> float:
        return float(np.mean(self.burning_mask()))

    def burned_fraction(self) -> float:
        """Fuel consumed relative to per-cell initial fuel."""
        return float(1.0 - np.sum(self.beta) / np.sum(self.beta0))

    def asset_damage(self) -> float:
        """Value-weighted fraction of asset fuel consumed."""
        w = self.asset_value
        tot = float(np.sum(w * self.beta0))
        if tot <= 0.0:
            return 0.0
        return float(1.0 - np.sum(w * self.beta) / tot)

    def burn_state(self, u=None, beta=None, u_pc=None) -> np.ndarray:
        u = self.u if u is None else u
        beta = self.beta if beta is None else beta
        sc = self.sc
        st = np.full(u.shape, UNBURNED, dtype=np.int8)
        st[beta < (1.0 - sc.burned_fuel_loss) * self.beta0] = BURNED
        st[beta < (1.0 - sc.overwhelming_fuel_loss) * self.beta0] = DESTROYED
        st[self.burning_mask(u, beta, u_pc)] = BURNING
        return st

    def qois(self) -> Dict[str, float]:
        return {
            "t": self.t,
            "burned": self.burned_fraction(),
            "burning": self.burning_fraction(),
            "asset_damage": self.asset_damage(),
        }

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Full state, for a snapshot bank or for plotting."""
        return {
            "u": self.u.copy(),
            "beta": self.beta.copy(),
            "wetness": self.wetness.copy(),
            "t": self.t,
            "n_steps": self.n_steps,
        }

    def load_snapshot(self, snap: Dict[str, np.ndarray]):
        self.u = snap["u"].copy()
        self.beta = snap["beta"].copy()
        self.wetness = snap["wetness"].copy()
        self.t = float(snap["t"])
        self.n_steps = int(snap["n_steps"])
        return self



def run(
    steps: int = 500,
    params: Optional[SolverParams] = None,
    scenario: Optional[Scenario] = None,
    ignition: Optional[Ignition] = None,
    frame_steps: Optional[List[int]] = None,
):
    """Driver reproducing the original run(): frames, coords, history."""
    sim = WildfireSim(params=params, scenario=scenario).reset(ignition)
    if frame_steps is None:
        frame_steps = [0, steps // 3, 2 * steps // 3, steps]

    frames = {0: (sim.u.copy(), sim.beta.copy())}
    history = {"step": [0], "burning": [sim.burning_fraction()],
               "burned": [sim.burned_fraction()],
               "asset_damage": [sim.asset_damage()]}

    for step in range(1, steps + 1):
        sim.step_once()
        history["step"].append(step)
        history["burning"].append(sim.burning_fraction())
        history["burned"].append(sim.burned_fraction())
        history["asset_damage"].append(sim.asset_damage())
        if step in frame_steps:
            frames[step] = (sim.u.copy(), sim.beta.copy())

    return sim, frames, history


if __name__ == "__main__":
    from plot_wildfire import plot

    sim, frames, history = run()
    plot(sim, frames)
    print(
        f"final burning={history['burning'][-1]:.4f}  "
        f"burned={history['burned'][-1]:.4f}  "
        f"asset_damage={history['asset_damage'][-1]:.4f}"
    )