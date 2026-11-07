from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from pgmtwin.core.action import BaseAction
from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.digital_asset import BaseDigitalAsset as _Base

from modified_physical_asset import QOI_LABELS, WildfireAsset
from modified_priority import optimal_drop
from modifiedsimplephysicalmodel import UNBURNED


def _coarsen(field: np.ndarray, factor: int) -> np.ndarray:
    """Block-average a fine (n, m) field onto an (n/factor, m/factor) grid."""
    n, m = field.shape
    return field.reshape(n // factor, factor, m // factor, factor).mean(axis=(1, 3))


class WildfireDigitalAsset(_Base):
    """Digital asset that imports observations from the physical asset at
    each time step.

    NO INVERSE PROBLEM YET: the sensor readings are stored but not inverted.
    Assimilation uses the ground truth: it takes the true QoI record
    (burned, burning, asset_damage), adds Gaussian noise, and snaps it onto
    the state domain. Swap get_assimilation for a real inverse solver later;
    the stored sensor history is what it will consume.

    Drop placement uses the priority index on coarse risk/temperature maps.
    These are also ground truth for now (the latest true fields); the
    inverse problem should eventually provide them.
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        state_distro: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        asset_map: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        wind: Optional[Sequence[float]] = None,
        coarse_factor: int = 4,
        qoi_noise_std: float = 0.0,
        n_qoi_samples: int = 1,
    ):
        super().__init__(state_domain, state_distro=state_distro, rng=rng)
        # Gaussian noise on the true QoIs: each step assimilates
        # n_qoi_samples noisy copies, so state_distro spreads over bins
        self.qoi_noise_std = qoi_noise_std
        self.n_qoi_samples = n_qoi_samples
        self.history: List[Dict[str, np.ndarray]] = []

        # site information the twin is assumed to know: asset values, grid
        # coordinates (axis 0 is x) and wind
        self.asset_map = asset_map
        self.coords = coords
        self.wind = wind
        self.coarse_factor = coarse_factor
        self.fields: Optional[Dict[str, np.ndarray]] = None  # latest maps only

    # pgmtwin contract
    def get_assimilation(self, observations: np.ndarray) -> np.ndarray:
        """Ground-truth placeholder: observations ARE QoI records, so the
        "inverse solution" is just the nearest domain value."""
        return self.state_domain.values2values(observations)

    def get_assimilation_distribution(self, observations: np.ndarray) -> np.ndarray:
        observations = np.atleast_2d(observations)

        ret = np.zeros(len(self.state_domain))
        for o in observations:
            ret[self.state_domain.values2index(self.get_assimilation(o))] += 1

        return ret / np.sum(ret)

    def noisy_qois(self, qois: np.ndarray) -> np.ndarray:
        """(n_qoi_samples, n_qois) copies of qois + N(0, qoi_noise_std),
        clipped to [0, 1] since every QoI is a fraction."""
        noise = self.rng.normal(0.0, self.qoi_noise_std, size=(self.n_qoi_samples, qois.size))
        return np.clip(qois[None, :] + noise, 0.0, 1.0)

    # time-step import
    def reset(self) -> None:
        self.reset_state_distribution()
        self.history.clear()
        self.fields = None

    def import_step(
        self,
        t: float,
        sensor_obs: np.ndarray,
        qois: np.ndarray,
        action: Optional[BaseAction] = None,
        fields: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Record one time step and update the state distribution.

        Args:
            t: simulation time of the step
            sensor_obs: (n_observations, 2 * n_sensors) noisy temperatures
                followed by 0/1 burning flags
            qois: true (burned, burning, asset_damage), assimilated after
                adding Gaussian noise (see noisy_qois)
            action: the action that led to this step, if any
            fields: true "u" (temperature) and "affected" (burning or burned)
                maps, used for drop placement
        """
        qois = np.asarray(qois, dtype=float).ravel()
        self.assimilate(self.noisy_qois(qois))
        if fields is not None:
            self.fields = fields

        self.history.append(
            dict(
                t=float(t),
                action=None if action is None else action.name,
                sensor_obs=np.atleast_2d(np.array(sensor_obs, dtype=float)),
                qois=qois.copy(),
                state_distro=self.state_distro.copy(),
            )
        )
        return self.state_distro

    def import_from(
        self,
        physical: WildfireAsset,
        n_observations: int = 1,
        action: Optional[BaseAction] = None,
    ) -> np.ndarray:
        """Pull the current time step straight from a physical asset."""
        return self.import_step(
            t=physical.sim.t,
            sensor_obs=physical.get_observations(n_observations),
            qois=physical.state,
            action=action,
            fields=true_fields(physical),
        )

    # drop placement
    def recommend_drop(self) -> Optional[Dict[str, object]]:
        """Priority-index drop on the coarse grid, in physical coordinates.

        Returns x, y, theta (radians, long axis of the drop) plus the coarse
        priority fields, or None if there is nothing worth protecting yet.
        """
        if self.fields is None or self.asset_map is None:
            raise RuntimeError("recommend_drop needs fields and site information")

        c = self.coarse_factor
        risk = _coarsen(self.fields["affected"].astype(float), c)
        temperature = _coarsen(self.fields["u"], c)
        assets = _coarsen(self.asset_map, c)

        rec = optimal_drop(risk, temperature, assets, self.wind)
        if rec is None:
            return None

        centers = self.coords.reshape(-1, c).mean(axis=1)  # coarse cell centres
        i, j = rec["location"]
        rec.update(x=float(centers[i]), y=float(centers[j]), theta=rec["angle"])
        return rec

    # views over the imported history 
    @property
    def times(self) -> np.ndarray:
        return np.array([h["t"] for h in self.history])

    def sensor_history(self) -> np.ndarray:
        """(n_steps, n_observations, 2 * n_sensors): temperatures, then burning flags"""
        return np.stack([h["sensor_obs"] for h in self.history])

    def qoi_history(self) -> np.ndarray:
        """(n_steps, len(QOI_LABELS))"""
        return np.stack([h["qois"] for h in self.history])

    def map_state_history(self) -> np.ndarray:
        """Most likely domain value per step, (n_steps, n_vars)."""
        idx = [int(np.argmax(h["state_distro"])) for h in self.history]
        return self.state_domain.index2values(np.array(idx))

    def get_qois(self) -> Dict[str, float]:
        """Latest imported QoIs, by label."""
        return dict(zip(QOI_LABELS, self.history[-1]["qois"]))


def true_fields(physical: WildfireAsset) -> Dict[str, np.ndarray]:
    """True maps handed to the digital asset until the inverse problem
    provides them: temperature, and where the fire has reached."""
    return {
        "u": physical.sim.u.copy(),
        "affected": physical.sim.burn_state() != UNBURNED,
    }


def run_and_import(
    physical: WildfireAsset,
    digital: WildfireDigitalAsset,
    actions: Sequence[BaseAction],
    n_observations: int = 1,
) -> WildfireDigitalAsset:
    """Import the initial step, then step the physical asset through
    `actions` and import the observations after each one."""
    digital.import_from(physical, n_observations)
    for a in actions:
        physical.update(a)
        digital.import_from(physical, n_observations, action=a)
    return digital
