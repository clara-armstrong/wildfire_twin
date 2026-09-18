"""PDE-based wildfire physical simulation (thesis Chapter 3): the "ground
truth" simulator behind the pgmtwin BasePhysicalAsset interface.

"""
from typing import Optional

import numpy as np
from pgmtwin.core.physical_asset import BasePhysicalAsset

from .config import WildfireConfig, ObservationConfig, DEFAULT_OBS_CONFIG
from .observation import pack_state, unpack_state, observe
from .solver import apply_boundary_conditions, build_elevation_map, build_fuel_map


class WildfirePhysicalAsset(BasePhysicalAsset):
    # Ground-truth simulator; wraps pgmtwin's update()/get_observations()/
    # set_state() interface for actions.py/policy.py/digital_asset.py.

    def __init__(
        self,
        config: WildfireConfig,
        obs_config: Optional[ObservationConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.config = config
        self.obs_config = obs_config if obs_config is not None else DEFAULT_OBS_CONFIG
        self.grid_size = config.grid_size
        self.terminated = False
        self.step_count = 0

        self.dx = config.dx
        self.x_coords = config.coords
        self.y_coords = config.coords

        # rng must be set before _initialize_state, which uses it
        self.rng = rng

        self.state_dict = self._initialize_state()
        state_array = pack_state(self.state_dict, self.grid_size)

        super().__init__(state_array, rng)
        self.state_array = state_array

    def _initialize_state(self) -> dict:
        # True wind + contiguous fuel/asset regions + ignition seed (Eq.
        # 5.1/5.3, Figure 5.2). wind/fuel_type/elevation are plain
        # attributes, not in state_dict: only the 5 STATE_FIELDS round-trip
        # through pack_state/unpack_state, so anything else would be
        # silently dropped every step.
        g = self.grid_size
        self.wind_speed, self.wind_direction = self._sample_wind()

        self.elevation = build_elevation_map(self.config)
        fuel_type, fuel_fraction, ignition_threshold, asset_importance = build_fuel_map(self.config)
        self.fuel_type = fuel_type

        state = {
            "temperature": np.zeros((g, g), dtype=np.float32),
            "fuel_fraction": fuel_fraction,
            "ignition_threshold": ignition_threshold,
            "burn_status": np.zeros((g, g), dtype=int),
            "asset_importance": asset_importance,
        }

        x0, y0 = self._sample_ignition_point()
        X, Y = np.meshgrid(self.x_coords, self.y_coords, indexing="ij")
        c = self.config.ignition_decay
        A = self.config.ignition_amplitude
        state["temperature"] = (
            A * np.exp(-c * ((X - x0) ** 2 + (Y - y0) ** 2))
        ).astype(np.float32)

        i0 = int(np.argmin(np.abs(self.x_coords - x0)))
        j0 = int(np.argmin(np.abs(self.y_coords - y0)))
        state["burn_status"][i0, j0] = 1

        return state

    def _sample_ignition_point(self):
        # Ignition center with uncertainty (Eq. 5.3). A single continuous
        # draw -- this is the "real" fire, not one of LatentDomain's
        # discretized hypotheses.
        sigma = self.config.ignition_sigma
        x0 = self.config.ignition_center_x + self.rng.uniform(-sigma, sigma)
        y0 = self.config.ignition_center_y + self.rng.uniform(-sigma, sigma)
        return x0, y0

    def _sample_wind(self):
        # True wind speed/direction with uncertainty (uniform, thesis Eq.
        # 5.2). A single continuous draw around config.wind_speed/
        # wind_direction_deg -- the same relationship _sample_ignition_point
        # has to config.ignition_center_x/y -- not one of LatentDomain's
        # discretized wind hypotheses.
        speed = self.config.wind_speed + self.rng.normal(0, self.config.wind_speed_sigma)
        direction = (
            self.config.wind_direction_deg + self.rng.normal(0, self.config.wind_direction_sigma)
        ) % 360
        return float(max(speed, 0.0)), float(direction)

    def _set_wind(self, direction_deg: float):
        self.wind_direction = float(direction_deg % 360)

    def _sync_state(self):
        # Refresh state_dict from state_array (pgmtwin may mutate the array
        # directly, so state_dict can go stale between calls).
        self.state_dict = unpack_state(self.state_array, self.grid_size)

    def get_observations(self, n_observations: int = 1) -> np.ndarray:
        # n_observations independent noisy readings (Eq. 5.1.1's sensor
        # model), for WildfireDigitalAsset.get_assimilation.
        self._sync_state()
        return np.stack([
            observe(self.state_dict, self.obs_config, rng=self.rng)
            for _ in range(n_observations)
        ])

    def set_state(self, state):
        # Set the physical state from either a state dict or a flat array.
        if isinstance(state, dict):
            self.state_dict = state
            self.state_array = pack_state(state, self.grid_size)
        else:
            self.state_array = np.asarray(state)
            self.state_dict = unpack_state(self.state_array, self.grid_size)
        self.terminated = False
        self.step_count = 0

    def _rhs(self, u, beta):
        # Mirrors solver.Wildfire._rhs, boundary condition included: both
        # integrate the same RHS, so they must treat the edge ring the same
        # way or they diverge as soon as the fire reaches it.
        f_u, g_beta = self._compute_reaction_terms(u, beta)
        du_dt = self._diffusion_term(u) + self._advection_term(u) + f_u
        return apply_boundary_conditions(du_dt, g_beta)

    def update(self, action):
        # One RK4 step, matching solver.Wildfire.step.
        self.step_count += 1

        # Optional one-time wind-direction switch, same as solver.py's
        # Wildfire.step (config.wind_switch_step / wind_direction_deg_after).

        cfg = self.config
        if cfg.wind_switch_step > 0 and self.step_count == cfg.wind_switch_step:
            self._set_wind(cfg.wind_direction_deg_after)

        self._sync_state()
        dt = cfg.dt
        u = self.state_dict["temperature"]
        beta = self.state_dict["fuel_fraction"]

        k1_u, k1_b = self._rhs(u, beta)
        k2_u, k2_b = self._rhs(u + 0.5 * dt * k1_u, beta + 0.5 * dt * k1_b)
        k3_u, k3_b = self._rhs(u + 0.5 * dt * k2_u, beta + 0.5 * dt * k2_b)
        k4_u, k4_b = self._rhs(u + dt * k3_u, beta + dt * k3_b)
 
        self.state_dict["temperature"] = u + (dt / 6.0) * (
            k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u
        )
        self.state_dict["fuel_fraction"] = beta + (dt / 6.0) * (
            k1_b + 2.0 * k2_b + 2.0 * k3_b + k4_b
        )

        self._update_burn_status()
        self._apply_action(action)
        self._enforce_bounds()

        self.state_array = pack_state(self.state_dict, self.grid_size)

        if self._fire_contained() or self.step_count > self.config.max_steps:
            self.terminated = True

    def _compute_reaction_terms(self, u, beta):
        # Combustion terms, same formula as solver.py's _reaction
        #     f(u,beta) =  H_pc * beta * exp(u/(1+eps*u)) - alpha*u
        #     g(u,beta) = -H_pc * (eps/q) * beta * exp(u/(1+eps*u))
        u_pc = self.state_dict["ignition_threshold"]

        H_pc = np.where(u >= u_pc, 1.0, 0.0)
        arrhenius = np.exp(u / (1 + self.config.epsilon * u))

        f_u = H_pc * beta * arrhenius - self.config.alpha * u
        g_beta = -H_pc * (self.config.epsilon / self.config.q) * beta * arrhenius # USES Q HERE
        return f_u, g_beta

    def _diffusion_term(self, u: np.ndarray) -> np.ndarray:
        # kappa*Laplacian(u), central differences
        # RK4 stages in update() apply dt.
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
        ) / self.dx**2
        return self.config.kappa * lap

    def _advection_term(self, u: np.ndarray) -> np.ndarray:
        # -v.grad(u), upwind scheme  uniform wind, same model as solver.py's _advection
        theta = np.radians(self.wind_direction)
        vx = self.wind_speed * np.cos(theta)
        vy = self.wind_speed * np.sin(theta)

        dudx = np.zeros_like(u)
        dudy = np.zeros_like(u)
        if vx > 0:
            dudx[1:, :] = (u[1:, :] - u[:-1, :]) / self.dx
        else:
            dudx[:-1, :] = (u[1:, :] - u[:-1, :]) / self.dx
        if vy > 0:
            dudy[:, 1:] = (u[:, 1:] - u[:, :-1]) / self.dx
        else:
            dudy[:, :-1] = (u[:, 1:] - u[:, :-1]) / self.dx

        return -(vx * dudx + vy * dudy)

    def _update_burn_status(self):
        # Unburned -> Burning -> Burned transitions (diagnostic label only,
        # same logic as solver.py's Wildfire._update_burn_status).
        u = self.state_dict["temperature"]
        beta = self.state_dict["fuel_fraction"]
        u_pc = self.state_dict["ignition_threshold"]

        unburned = self.state_dict["burn_status"] == 0
        burning = (u >= u_pc) & (beta > 0.1) & unburned
        self.state_dict["burn_status"][burning] = 1

        depleted = (self.state_dict["burn_status"] == 1) & (beta <= 0.1)
        self.state_dict["burn_status"][depleted] = 2

        extinguished = (self.state_dict["burn_status"] == 1) & (u < 0.5)
        self.state_dict["burn_status"][extinguished] = 2

    def _apply_action(self, action):
        # Apply a management action (see actions.py) to the physical state.
        if hasattr(action, "apply"):
            action.apply(self.state_dict)

    def _enforce_bounds(self):
        # Clip all state variables to their physically valid ranges. The
        # boundary condition is applied to the RHS in _rhs, not here.
        sd = self.state_dict
        sd["temperature"] = np.clip(sd["temperature"], 0, 8)
        sd["fuel_fraction"] = np.clip(sd["fuel_fraction"], 0, 1)
        sd["ignition_threshold"] = np.clip(sd["ignition_threshold"], 2.5, 4.0)
        sd["asset_importance"] = np.clip(sd["asset_importance"], 0, 1)
        self.wind_speed = float(np.clip(self.wind_speed, 0, 3))
        self.wind_direction = float(self.wind_direction % 360)

    def _fire_contained(self) -> bool:
        return np.sum(self.state_dict["burn_status"] == 1) == 0
