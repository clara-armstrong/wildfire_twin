"""Digital twin of the wildfire, backed by a Bayesian inverse solver.

"""
from typing import Dict, List, Optional

import numpy as np
from pgmtwin.core.digital_asset import BaseDigitalAsset as _Base

from .config import (
    DEFAULT_LIKELIHOOD_TEMPERATURE,
    DEFAULT_MODEL_ERROR_STD,
    ObservationConfig,
    WildfireConfig,
)
from .domain import LatentDomain
from .inverse import ForwardCache, InverseSolver
from .priority import optimal_drop
from .solver import Latent


class WildfireDigitalAsset(_Base):
    # Digital twin state = a posterior over the latent parameters, not a
    # copy of the physical grid.

    def __init__(
        self,
        config: WildfireConfig,
        state_domain: LatentDomain,
        cache: ForwardCache,
        obs_config: ObservationConfig,
        model_error_std: float = DEFAULT_MODEL_ERROR_STD,
        temperature: float = DEFAULT_LIKELIHOOD_TEMPERATURE,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(state_domain, rng=rng)
        self.config = config
        self.obs_config = obs_config
        self.solver = InverseSolver(
            state_domain, cache, obs_config,
            model_error_std=model_error_std, temperature=temperature,
        )
        self.observation_history: List[np.ndarray] = []
        self.posterior_history: List[np.ndarray] = []

    # Assimilation
    def reset(self) -> None:
        self.solver.reset()
        self.observation_history.clear()
        self.posterior_history.clear()

    def get_assimilation(self, observations: np.ndarray) -> np.ndarray:
        # Assimilate one or more observations; return the latent estimate.
        # Raises on a shape mismatch instead of silently substituting a
        # default state (a loud failure beats a silently wrong digital twin).
        observations = np.atleast_2d(observations)
        expected = self.solver.cache.obs.shape[2]
        if observations.shape[1] != expected:
            raise ValueError(
                f"observation has {observations.shape[1]} components, expected "
                f"{expected}. This usually means a different coarse_size or "
                "channel set than the cache was built with."
            )

        for obs in observations:
            self.solver.update(obs)
            self.observation_history.append(np.asarray(obs))
            self.posterior_history.append(self.solver.posterior.copy())

        summary = self.solver.posterior_mean_latent()
        return np.array([summary[v]["mean"] for v in
                         ("x0", "y0", "wind_speed", "wind_direction")])

    def get_assimilation_distribution(self, observations: np.ndarray) -> np.ndarray:
        # Posterior over the latent domain: the actual inverse solve.
        self.get_assimilation(observations)
        return self.solver.posterior

    # Estimates
    @property
    def posterior(self) -> np.ndarray:
        return self.solver.posterior

    def map_latent(self) -> Latent:
        # Maximum a posteriori point estimate, if a single "best guess"
        # theta is needed rather than the full distribution.
        return self.solver.map_latent()

    def latent_summary(self) -> Dict[str, Dict[str, float]]:
        # Posterior mean/std per latent variable, for render.py's
        # ignition/wind text box.
        return self.solver.posterior_mean_latent()

    # Risk
    def risk_map(self, step: int) -> np.ndarray:
        # Posterior burn probability on the coarse grid.
        return self.solver.risk_map(step)

    def prior_risk_map(self, step: int) -> np.ndarray:
        # The thesis's unconditioned Monte Carlo risk map: never sharpens,
        # since it doesn't depend on any observation. Kept for comparison.
        return self.solver.prior_risk_map(step)

    def get_qois(self, step: int) -> Dict[str, float]:
        # Quantities of interest as posterior expectations, with spread.
        return self.solver.qois(step)

    # Decision
    def recommend_drop(
        self,
        step: int,
        temperature_map: np.ndarray,
        asset_map: np.ndarray,
        use_posterior: bool = True,
        lam: float = 0.1,
    ) -> Dict[str, object]:
        # Optimal water-drop location from the priority index (Eqs. 5.4-5.8).
        # use_posterior=False reproduces the thesis's own behavior (its
        # Agent.optimize() always uses the prior risk map, having no
        # posterior to switch to) -- the direct comparison for what the
        # inverse solve buys.
        risk = self.risk_map(step) if use_posterior else self.prior_risk_map(step)
        summary = self.latent_summary()
        theta = np.radians(summary["wind_direction"]["mean"])
        speed = summary["wind_speed"]["mean"]
        wind = (float(speed * np.cos(theta)), float(speed * np.sin(theta)))
        return optimal_drop(risk, temperature_map, asset_map, wind, lam=lam)
