from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
# using gymnasium like the examples from pgmtwin
import numpy as np

from pgmtwin.core.action import BaseAction
from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.env import BaseDigitalTwinEnv

from modified_digital_asset import WildfireDigitalAsset, true_fields
from modified_physical_asset import PriorityDropAction, WildfireAsset


class WildfireEnv(BaseDigitalTwinEnv):
    """The twin loop as a gym env: act -> sense -> assimilate -> reward.

    What the policy sees is the digital asset's state_distro, never the
    physical state. One env step is one physical-asset update
    (steps_per_action solver steps).
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        actions: List[BaseAction],
        physical_asset: WildfireAsset,
        digital_asset: WildfireDigitalAsset,
        reward: Callable[[BaseAction, np.ndarray], float],
        n_samples_assimilation: int = 1,
        max_episode_steps: int = 50,
        burnout_threshold: float = 1e-4,
        min_drop_interval: float = 5.0,
        # min_drop_interval forces a "5 minute" wait time between drops
    ):
        super().__init__(
            state_domain=state_domain,
            actions=actions,
            physical_asset=physical_asset,
            digital_asset=digital_asset,
            reward=reward,
            n_samples_assimilation=n_samples_assimilation,
        )
        self.max_episode_steps = max_episode_steps
        self.burnout_threshold = burnout_threshold
        # minimum sim time between drops (1 time unit = 1 minute); a drop
        # requested sooner is turned into a no-op
        self.min_drop_interval = min_drop_interval
        self._n_steps = 0

    # episode
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        gym.Env.reset(self, seed=seed)

        physical: WildfireAsset = self._physical_asset
        physical.rng = self.np_random
        physical.sim.rng = self.np_random
        self._digital_asset.rng = self.np_random

        options = options or {}
        ignition = options.get("ignition") or physical.sim.sample_ignition()
        physical.sim.reset(ignition)
        physical.state = physical._qoi()
        if hasattr(self._reward, "reset"):  # rewards that track change since the last step
            self._reward.reset(physical.state)

        self._digital_asset.reset()
        self._n_steps = 0

        sensor_readings = self._step_sensor_readings(None)
        self._step_assimilate(None, sensor_readings)

        return self._get_obs(), self._get_info()

    # assimilation
    def _step_assimilate(self, action: Optional[int], sensor_readings: np.ndarray):
        # TODO: once the inverse solver exists, the digital asset should
        # assimilate sensor_readings alone. Until then it also gets the
        # true QoIs (ground truth).
        physical: WildfireAsset = self._physical_asset
        self._digital_asset.import_step(
            t=physical.sim.t,
            sensor_obs=sensor_readings,
            qois=physical.state,
            action=None if action is None else self._actions[action],
            fields=true_fields(physical),
        )

    # episode end
    def _terminated(self) -> bool:
        burning = self._physical_asset.state[1]
        return bool(burning < self.burnout_threshold)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, np.ndarray]]:
        action = int(action)
        drop = self._actions[action]
        if isinstance(drop, PriorityDropAction):
            sim = self._physical_asset.sim
            last_drop_t = sim.drop_log[-1][0] if sim.drop_log else -np.inf
            # the digital asset decides where; with nothing to protect, don't drop
            rec = self._digital_asset.recommend_drop()
            if sim.t - last_drop_t < self.min_drop_interval - 1e-9 or rec is None:
                # if last drop has been more recent than the minimum drop interval, there is no drop
                action = 0
            else:
                drop.x, drop.y, drop.theta = rec["x"], rec["y"], rec["theta"]

        obs, reward, terminated, _, info = super().step(action)
        self._n_steps += 1
        truncated = self._n_steps >= self.max_episode_steps
        return obs, float(reward), terminated, truncated, info

    # log/plot info
    def _get_qois(self) -> np.ndarray:
        return self._physical_asset.state.copy()

    def _get_info(self) -> Dict[str, np.ndarray]:
        info = super()._get_info()
        info["t"] = self._physical_asset.sim.t
        return info


if __name__ == "__main__":
    # check the env against the gymnasium API, on a small grid for speed
    from gymnasium.utils.env_checker import check_env

    from modified_config import make_env
    from modifiedsimplephysicalmodel import SolverParams

    check_env(make_env(seed=0, solver=SolverParams(grid_size=64)), skip_render_check=True)
    print("env check passed")
