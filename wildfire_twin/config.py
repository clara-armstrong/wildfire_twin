"""Configuration for the wildfire digital twin.

"""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# Fuel properties (thesis Figure 5.2 / Appendix B.1)

@dataclass(frozen=True)
class FuelProperties:
    # One fuel type's properties: beta = fuel fraction, u_pc = ignition threshold.
    name: str
    fuel_fraction: float   # beta
    ignition_threshold: float  # u_pc


# These are the values taken from the appendix, stored in config for referrance
FUEL_TYPES: Dict[int, FuelProperties] = {
    0: FuelProperties(name="Forest", fuel_fraction=0.9, ignition_threshold=3.5),
    1: FuelProperties(name="Shrub", fuel_fraction=0.45, ignition_threshold=3.0),
    2: FuelProperties(name="Grass", fuel_fraction=0.2, ignition_threshold=2.5),
    3: FuelProperties(name="Assets", fuel_fraction=0.7, ignition_threshold=3.0),
}


# Physical / model parameters

@dataclass(frozen=True)
class WildfireConfig:
    # Defaults = parameters taken from Table 5.1: Values used for physical and model 
    # parameters in all experiments

    grid_size: int = 128
    domain_min: float = -100.0
    domain_max: float = 100.0

    # PDE constants, kept from Table 5.1.
    kappa: float = 1.0        # diffusion constant
    epsilon: float = 0.3      # inverse of activation energy
    alpha: float = 0.001      # radiative constant

    # Table 5.1's literal value, also the reference repo's. At grid_size=128
    # this leaves the combustion front narrower than one grid cell --
    # front_width()/resolution_report() below give delta/dx < 1. 
    # This issue is being tabled for now
    q: float = 1.0            # nondimensional heat of combustion

    # Ignition Gaussian, kept from thesis Eq. 5.1. 
    ignition_amplitude: float = 6.0       # A
    ignition_decay: float = 0.005         # c
    ignition_center_x: float = -35.0      # x_bar_0
    ignition_center_y: float = -35.0      # y_bar_0
    ignition_sigma: float = 5.0           # sigma, Eq. 5.3

    dt: float = 0.02
    max_steps: int = 400

    # Wind, kept from thesis Eq. 5.2 (uniform, not per-cell). The optional
    # one-time direction switch matches the reference repo's changing_wind
    # (NW to due-West at t=30).
    # Wind will be edited to be more realistic in the future
    wind_speed: float = 1.0
    wind_direction_deg: float = 45.0      # v1 = v/sqrt(2) * [1, 1]
    wind_switch_step: int = -1            # >0 reproduces the v2 changing field
    wind_direction_deg_after: float = 0.0
    elevation_amplitude: float = 100.0
    slope_speed: float = 0.0

    # Used by physical_asset.py to draw the "true" ground-truth wind around
    # wind_speed/wind_direction_deg (uniform, still Eq. 5.2 -- just an
    # unknown draw instead of a fixed given, the same relationship
    # ignition_sigma has to ignition_center_x/y).
    wind_speed_sigma: float = 0.3
    wind_direction_sigma: float = 20.0

    @property
    def dx(self) -> float:
        return (self.domain_max - self.domain_min) / (self.grid_size - 1)

    @property
    def coords(self) -> np.ndarray:
        return np.linspace(self.domain_min, self.domain_max, self.grid_size)

    def front_width(self, u_ref: float = 5.0) -> float:
        # TODO: Right now, keeping with q value from the thesis, this is not a satisfactory value
        # Combustion-front width, in the same units as dx. The front needs
        # to span more than ~1 dx to be resolved by the finite-difference grid.
        # Since q = 1, the front does not span more than ~1 dx
        arrhenius = float(np.exp(u_ref / (1.0 + self.epsilon * u_ref)))
        return float(np.sqrt(self.kappa * self.q / (self.epsilon * arrhenius)))

    def resolution_report(self, u_ref: float = 5.0) -> Dict[str, float]:
        # Quick check: is this grid fine enough to resolve the front at this q?
        # For q = 1, we expect this to not work
        delta = self.front_width(u_ref)
        return {"front_width": delta, "dx": self.dx, "delta_over_dx": delta / self.dx}

    def cfl_report(self) -> Dict[str, float]:
        # Stability check for the explicit scheme: both *_ratio values must
        # stay below 1, or diffusion/advection blow up numerically.
        dx = self.dx
        diff_limit = dx ** 2 / (4.0 * self.kappa)
        adv_limit = dx / max(self.wind_speed + self.slope_speed, 1e-12)
        return {
            "dt": self.dt,
            "diffusion_limit": diff_limit,
            "diffusion_ratio": self.dt / diff_limit,
            "advection_limit": adv_limit,
            "advection_ratio": self.dt / adv_limit,
        }


# State serialisation

# The 5 fields the forward cache and latent domain reason about. Wind,
# elevation and fuel_type are excluded: wind is latent, and elevation and
# fuel_type are fixed and not cache-relevant.
STATE_FIELDS = (
    "temperature",
    "fuel_fraction",
    "ignition_threshold",
    "burn_status",
    "asset_importance",
)

# Divisors used to map raw state onto a roughly [0, 1] observation scale.
OBSERVATION_SCALE: Dict[str, float] = {
    "temperature": 8.0,
    "fuel_fraction": 1.0,
    "ignition_threshold": 4.0,
    "burn_status": 2.0,
    "asset_importance": 1.0,
}

# Observation operator (thesis Section 5.1.1: the coarse grid Omega-tilde)
@dataclass(frozen=True)
class ObservationConfig:
    # Sensor model, kept from thesis Sec 5.1.1 (coarse, noisy observation
    # of the fine PDE grid). Noise added to make inverse problem interesting

    coarse_size: int = 16
    channels: Tuple[str, ...] = ("temperature", "burn_status")
    noise_std: float = 0.05          # additive Gaussian noise on the [0,1] scale
    # Noising the generated data to make inverse problem interesting
    observe_every: int = 25          # assimilate an observation every N steps
    n_observations: int = 6          # number of assimilation times

# Latent parameters: the unknown of the inverse problem

@dataclass(frozen=True)
class LatentGridConfig:


    n_x0: int = 7
    n_y0: int = 7
    wind_speeds: Tuple[float, ...] = (0.5, 1.0, 1.5)
    wind_directions_deg: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)

    def __len__(self) -> int:
        return self.n_x0 * self.n_y0 * len(self.wind_speeds) * len(self.wind_directions_deg)


DEFAULT_CONFIG = WildfireConfig()
DEFAULT_OBS_CONFIG = ObservationConfig()
DEFAULT_LATENT_CONFIG = LatentGridConfig()
