"""Core wildfire PDE solver (thesis Chapter 3, finite-difference form of Ch. 4.3).

Solves the coupled reaction-diffusion-advection system:
    du/dt     = kappa * Laplacian(u) - v . grad(u) + f(u, beta)
    dbeta/dt  = g(u, beta)
with
    f(u, beta) = H_pc * beta * exp(u / (1 + eps*u)) - alpha * u
    g(u, beta) = -H_pc * (eps/q) * beta * exp(u / (1 + eps*u))
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FUEL_TYPES, WildfireConfig

UNBURNED, BURNING, BURNED = 0, 1, 2


@dataclass(frozen=True)
class Latent:
    x0: float
    y0: float
    wind_speed: float
    wind_direction_deg: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x0, self.y0, self.wind_speed, self.wind_direction_deg], dtype=float
        )


# Static fields

def asset_regions(config: WildfireConfig) -> List[Tuple[int, int, int, int]]:
    # The two protected asset blocks (thesis Sec. 5.1.1), as (i0, i1, j0, j1)
    # index rectangles. Single source of truth: build_fuel_map paints these
    # into the fuel/asset maps, and actions.build_default_actions builds the
    # matching evacuation zones from them, so the two can't drift apart.
    g = config.grid_size
    half = g // 6
    out = []
    for ci, cj in ((3 * g // 4, g // 4), (g // 4, 3 * g // 4)):
        out.append((
            max(0, ci - half), min(g, ci + half),
            max(0, cj - half), min(g, cj + half),
        ))
    return out


def build_fuel_map(config: WildfireConfig):
    # Right now: Contiguous fuel regions + two square asset blocks from thesis.
    # Returns (fuel_type, fuel_fraction, ignition_threshold, asset_importance).
    # Fixed per trajectory, so this runs once per
    # Wildfire instance, not once per step.
    g = config.grid_size
    coords = config.coords
    X, Y = np.meshgrid(coords, coords, indexing="ij")

    fuel_type = np.empty((g, g), dtype=np.int8)
    fuel_type[:] = 0                                  # Forest, lower-left
    fuel_type[(Y <= 0) & (X > 0)] = 1                 # Shrub, lower-right
    fuel_type[Y > 0] = 2                              # Grass, upper half

    asset_importance = np.zeros((g, g), dtype=np.float32)
    for i0, i1, j0, j1 in asset_regions(config):
        asset_importance[i0:i1, j0:j1] = 1.0
        fuel_type[i0:i1, j0:j1] = 3                   # Assets

    beta = np.zeros((g, g), dtype=np.float32)
    u_pc = np.zeros((g, g), dtype=np.float32)
    for k, props in FUEL_TYPES.items():
        mask = fuel_type == k
        beta[mask] = props.fuel_fraction
        u_pc[mask] = props.ignition_threshold

    return fuel_type, beta, u_pc, asset_importance

def build_elevation_map(config: WildfireConfig) -> np.ndarray:
    # synthetic "hilly" terrain. Shared with physical_asset.py
    x = np.linspace(-2, 2, config.grid_size)
    X, Y = np.meshgrid(x, x, indexing = "ij")
    elevation= (
        np.sin(X) * np.cos(Y) * 100
        + np.sin(X*2 + Y*1.5) * 50
        + np.cos(X*1.5 - Y*2) * 30
    )
    elevation -= elevation.mean()
    elevation *= config.elevation_amplitude / np.abs(elevation).max()
    return elevation.astype(np.float32)

def apply_boundary_conditions(
    du_dt: np.ndarray, dbeta_dt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Zero the time derivative on the domain edge, holding the boundary ring
    # fixed at its initial value.
    # Returns: du_dt, dbeta_dt
    du_dt = du_dt.copy()
    dbeta_dt = dbeta_dt.copy()
    du_dt[0, :] = du_dt[-1, :] = du_dt[:, 0] = du_dt[:, -1] = 0.0
    dbeta_dt[0, :] = dbeta_dt[-1, :] = dbeta_dt[:, 0] = dbeta_dt[:, -1] = 0.0
    return du_dt, dbeta_dt


def initial_temperature(config: WildfireConfig, x0: float, y0: float) -> np.ndarray:
    # Gaussian ignition seed, kept from thesis Eq. 5.1.
    coords = config.coords
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    u = config.ignition_amplitude * np.exp(
        -config.ignition_decay * ((X - x0) ** 2 + (Y - y0) ** 2)
    )
    return u.astype(np.float32)

# Solver

class Wildfire:
    # One wildfire trajectory for a fixed latent theta.

    def __init__(self, config: WildfireConfig, latent: Latent):
        self.config = config
        self.latent = latent
        self.step_count = 0

        fuel_type, beta, u_pc, asset = build_fuel_map(config)
        self.fuel_type = fuel_type
        self.asset_importance = asset
        self.u_pc = u_pc.copy()
        self.beta = beta.copy()
        self.u = initial_temperature(config, latent.x0, latent.y0)
        self.burn_status = np.zeros((config.grid_size, config.grid_size), dtype=np.int8)
        self.burn_status[self.u >= self.u_pc] = BURNING

        self._set_wind(latent.wind_direction_deg)

    # Wind
    def _set_wind(self, direction_deg: float):
        # Wind velocity components, uniform (thesis Eq. 5.2). Can switch
        # once mid-run using config.wind_switch_step, matching the reference
        # repo's changing_wind (NW to due-West at t=30). Held constant
        # across all four RK4 stages within one step().
        theta = np.radians(direction_deg)
        self.vx = float(self.latent.wind_speed * np.cos(theta))
        self.vy = float(self.latent.wind_speed * np.sin(theta))

    # Operators

    def _laplacian(self, u: np.ndarray) -> np.ndarray:
        # kappa * Laplacian(u), central differences (diffusion term).
        # The edge ring is left at zero here; the boundary condition is
        # applied once to the assembled RHS, in apply_boundary_conditions.
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
        ) / self.config.dx ** 2
        return lap

    def _advection(self, u: np.ndarray) -> np.ndarray:
        # -v . grad(u), first-order upwind (convection term). Wind is
        # uniform, so the upwind direction is chosen once by sign, not per cell.
        dx = self.config.dx
        dudx = np.zeros_like(u)
        dudy = np.zeros_like(u)

        if self.vx > 0:
            dudx[1:, :] = (u[1:, :] - u[:-1, :]) / dx
        else:
            dudx[:-1, :] = (u[1:, :] - u[:-1, :]) / dx

        if self.vy > 0:
            dudy[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
        else:
            dudy[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx

        return -(self.vx * dudx + self.vy * dudy)

    def _reaction(self, u: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Combustion terms, thesis Eq. 3.3:
        #     f(u,beta) = H(u - u_pc) * beta * exp(u/(1+eps*u)) - alpha*u
        #     g(u,beta) = -H(u - u_pc) * (eps/q) * beta * exp(u/(1+eps*u))
        cfg = self.config
        H_pc = (u >= self.u_pc).astype(np.float32)
        arrhenius = np.exp(u / (1.0 + cfg.epsilon * u))
        f_u = H_pc * beta * arrhenius - cfg.alpha * u
        g_beta = -H_pc * (cfg.epsilon / cfg.q) * beta * arrhenius # USES Q
        return f_u, g_beta

    def _rhs(self, u: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Full right-hand side (du/dt, dbeta/dt): diffusion + advection +
        # reaction for u, reaction alone for beta. Wind is frozen for the
        # step, so this doesn't depend on time.
        #
        #  per cell:
        #   du/dt    = heat spreading in + heat blown in by wind
        #            + heat from burning fuel (if hot enough) - cooling
        #   dbeta/dt = -fuel consumed by burning (if hot enough)
        f_u, g_beta = self._reaction(u, beta)
        du_dt = self.config.kappa * self._laplacian(u) + self._advection(u) + f_u
        return apply_boundary_conditions(du_dt, g_beta)

    # Stepping
    def step(self, action: Optional[Callable[["Wildfire"], None]] = None):
        # One RK4 step:
        #     k1 = F(y_n)
        #     k2 = F(y_n + dt/2 * k1)
        #     k3 = F(y_n + dt/2 * k2)
        #     k4 = F(y_n + dt   * k3)
        #     y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        # with y = (u, beta). A single dt scales u and beta together, same
        # as the reference repo
        cfg = self.config
        self.step_count += 1

        if cfg.wind_switch_step > 0 and self.step_count == cfg.wind_switch_step:
            self._set_wind(cfg.wind_direction_deg_after)

        dt = cfg.dt
        u, beta = self.u, self.beta

        k1_u, k1_b = self._rhs(u, beta)
        k2_u, k2_b = self._rhs(u + 0.5 * dt * k1_u, beta + 0.5 * dt * k1_b)
        k3_u, k3_b = self._rhs(u + 0.5 * dt * k2_u, beta + 0.5 * dt * k2_b)
        k4_u, k4_b = self._rhs(u + dt * k3_u, beta + dt * k3_b)

        self.u = u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        self.beta = beta + (dt / 6.0) * (k1_b + 2.0 * k2_b + 2.0 * k3_b + k4_b)

        self._update_burn_status()

        if action is not None:
            action(self)

        self._enforce_bounds()

    def _update_burn_status(self):
        # Unburned -> Burning -> Burned labels for downstream use (rendering,
        # observations, QoIs)
        igniting = (self.u >= self.u_pc) & (self.beta > 0.1) & (self.burn_status == UNBURNED)
        self.burn_status[igniting] = BURNING
        spent = (self.burn_status == BURNING) & ((self.beta <= 0.1) | (self.u < 0.5))
        self.burn_status[spent] = BURNED

    def _enforce_bounds(self):
        # Clip u/beta/u_pc to valid ranges. Added as a numerical safety net
        # since the forward cache runs this loop unattended, many times.
        np.clip(self.u, 0.0, 8.0, out=self.u)
        np.clip(self.beta, 0.0, 1.0, out=self.beta)
        np.clip(self.u_pc, 2.5, 4.0, out=self.u_pc)

    @property
    def contained(self) -> bool:
        # True once no cells are still burning (burn status 1).
        return not np.any(self.burn_status == BURNING)

    def state_dict(self) -> Dict[str, np.ndarray]:
        # State fields observation.py/inverse.py/render.py need, still in
        # nondimensional units (no unit conversion, unlike the reference repo).
        return {
            "temperature": self.u,
            "fuel_fraction": self.beta,
            "ignition_threshold": self.u_pc,
            "burn_status": self.burn_status.astype(np.float32),
            "asset_importance": self.asset_importance,
        }

    # Diagnostics
    def burnt_asset_area(self) -> float:
        # Fraction of asset cells burned or burning -- a QoI tracked by
        # inverse.ForwardCache/InverseSolver.
        asset = self.asset_importance > 0
        if not np.any(asset):
            return 0.0
        return float(np.mean(self.burn_status[asset] > UNBURNED))


def simulate(
    config: WildfireConfig,
    latent: Latent,
    n_steps: int,
    record_at: Optional[Sequence[int]] = None,
    action_schedule: Optional[Dict[int, Callable[[Wildfire], None]]] = None,
) -> Dict[str, object]:
    # Run one trajectory, optionally recording state at given steps and
    # applying a step -> action schedule. This project's decisions
    # come from the Bayesian posterior (priority.py/policy.py)
    fire = Wildfire(config, latent)
    record_at = sorted(set(record_at or []))
    wanted = set(record_at)
    action_schedule = action_schedule or {}

    frames: List[Dict[str, np.ndarray]] = []
    if 0 in wanted:
        frames.append({k: v.copy() for k, v in fire.state_dict().items()})

    for step in range(1, n_steps + 1):
        fire.step(action=action_schedule.get(step))
        if step in wanted:
            frames.append({k: v.copy() for k, v in fire.state_dict().items()})

    return {"fire": fire, "frames": frames, "record_at": record_at}
