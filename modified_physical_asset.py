from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from pgmtwin.core.action import BaseAction
from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.physical_asset import BasePhysicalAsset

"""grabbing from the physical model I made"""
from modifiedsimplephysicalmodel import (
    DropParams,
    Scenario,
    SolverParams,
    WildfireSim,
)

QOI_LABELS = ("burned", "burning", "asset_damage")


# actions 
class NoOpAction(BaseAction):
    """Let it burn for one decision interval."""

    def __init__(self, state_domain: DiscreteDomain):
        super().__init__(name="noop", state_domain=state_domain)


class DropAction(BaseAction):
    """A drop with a fixed footprint at (x, y), long axis at theta (radians).

    For a fixed-location drop, add DropAction("drop_0", dom, x, y, theta)
    to the action list; the policy then chooses the location by index.
    """

    def __init__(self, name: str, state_domain: DiscreteDomain,
                 x: float, y: float, theta: float):
        super().__init__(name=name, state_domain=state_domain)
        self.x, self.y, self.theta = x, y, theta


class PriorityDropAction(DropAction):
    """A drop placed by the digital asset's priority index.

    The policy only decides WHEN to drop; the env fills in x, y, theta from
    digital_asset.recommend_drop() just before the drop is applied.
    """

    def __init__(self, state_domain: DiscreteDomain):
        super().__init__("priority_drop", state_domain, x=0.0, y=0.0, theta=0.0)


# asset 
class WildfireAsset(BasePhysicalAsset):
    def __init__(
        self,
        state_domain: DiscreteDomain,
        steps_per_action: int = 20,
        sensor_coords: Optional[np.ndarray] = None,
        obs_noise_std: float = 0.15,
        burn_flip_prob: float = 0.0,
        snapshot_bank: Optional[Dict[int, List[dict]]] = None,
        solver: Optional[SolverParams] = None,
        scenario: Optional[Scenario] = None,
        drop_params: Optional[DropParams] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        rng = rng or np.random.default_rng()
        self.state_domain = state_domain
        self.steps_per_action = steps_per_action
        self.obs_noise_std = obs_noise_std
        self.burn_flip_prob = burn_flip_prob
        self.snapshot_bank = snapshot_bank or {}

        self.sim = WildfireSim(solver, scenario, drop_params, rng=rng)
        self.sim.reset()

        # fixed sensor sites: local temperature readings, NOT the QoIs.
        # The inverse problem is then a real one.
        if sensor_coords is None:
            g = np.linspace(-120, 120, 5)
            sensor_coords = np.array([[a, b] for a in g for b in g])
        self.sensor_coords = np.asarray(sensor_coords, dtype=float)
        self._sensor_idx = self._locate(self.sensor_coords)

        super().__init__(initial_state=self._qoi(), rng=rng)

    # helpers
    def _locate(self, pts: np.ndarray) -> np.ndarray:
        c = self.sim.coords
        return np.stack([np.argmin(np.abs(c - p[0])) for p in pts]), np.stack(
            [np.argmin(np.abs(c - p[1])) for p in pts]
        )

    def _qoi(self) -> np.ndarray:
        """Continuous QoIs. NOT snapped: nothing in the env indexes this,
        and the digital asset adds its noise to the full-resolution values. Arity must match the domain.
        """
        return np.array([
            self.sim.burned_fraction(),
            self.sim.burning_fraction(),
            self.sim.asset_damage(),
        ])

    # pgmtwin contract
    def update(self, action: BaseAction) -> np.ndarray:
        self.sim.rng = self.rng  # the env rebinds self.rng on every reset
        if isinstance(action, DropAction):
            self.sim.apply_drop(action.x, action.y, action.theta)
        self.sim.advance(self.steps_per_action)
        self.state = self._qoi()
        return self.state

    def get_observations(self, n_observations: int = 1) -> np.ndarray: # these need to be made compatible with the inverse problem
        """Noisy local temperature and burning flag at (25) fixed sensor sites.

        Returns (n_observations, 2 * n_sensors): the first n_sensors columns
        are temperatures, the last n_sensors are 0/1 burning flags, each
        flipped with probability burn_flip_prob.
        """
        ix, iy = self._sensor_idx
        truth = self.sim.u[ix, iy]
        noise = self.rng.normal(0.0, self.obs_noise_std,
                                size=(n_observations, truth.size))
        temps = np.clip(truth[None, :] + noise, 0.0, self.sim.p.u_max)

        burning = self.sim.burning_mask()[ix, iy]
        flips = self.rng.random((n_observations, burning.size)) < self.burn_flip_prob
        flags = (burning[None, :] ^ flips).astype(float)
        return np.concatenate([temps, flags], axis=1)

    def set_state(self, state: np.ndarray):
        state = np.asarray(state).ravel()
        idx = int(self.state_domain.values2index(state))

        bank = self.snapshot_bank.get(idx)
        if bank:
            self.sim.load_snapshot(bank[self.rng.integers(len(bank))])
        else:
            self.sim.reset(self.sim.sample_ignition())
            target = float(state[0])
            while self.sim.burned_fraction() < target and self.sim.n_steps < 4000:
                self.sim.step_once()

        self.state = self._qoi()