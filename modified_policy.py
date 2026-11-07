"""Rule-based policies for the wildfire twin.

Actions follow modified_config: 0 = no-op, 1 = priority drop.
"""

import csv
from typing import Optional

import numpy as np
from pgmtwin.core.tabular_policy import TabularPolicy
# the policy_table lives in the modified_twin folder
NOOP, DROP = 0, 1


def _batch(observation):
    obs = np.asarray(observation)
    return np.atleast_2d(obs), obs.ndim == 1


class NoDropPolicy:
    """Baseline policy, never drop."""

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs, single = _batch(observation)
        actions = np.full(len(obs), NOOP)
        return (int(actions[0]) if single else actions), state


class IntervalDropPolicy:
    """Priority drop every `interval` steps while the fire is burning.

    "Burning" is read from the digital state: the most likely burning
    fraction must be at least `min_burning`. The steps since the last drop
    are carried in the predict() `state`, one counter per env.
    """

    def __init__(self, state_domain, interval: int = 3, min_burning: float = 0.01):
        self.state_domain = state_domain
        self.interval = interval
        self.min_burning = min_burning
        self._burning = list(state_domain.var2values).index("burning")

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs, single = _batch(observation)

        # steps since the last drop; a new episode starts "due"
        since = np.full(len(obs), self.interval) if state is None else np.array(state)
        if episode_start is not None:
            since[np.asarray(episode_start, dtype=bool)] = self.interval

        burning = self.state_domain.index2values(np.argmax(obs, axis=1))[:, self._burning]
        drop = (burning >= self.min_burning) & (since >= self.interval)

        since = np.where(drop, 1, since + 1)
        actions = np.where(drop, DROP, NOOP)
        return (int(actions[0]) if single else actions), since


def threshold_table(state_domain, burning_min: Optional[float] = 0.02,
                    damage_min: Optional[float] = None, burned_min: Optional[float] = None):
    """Policy table, shape (n_actions, n_states): one column per QoI bin
    combination, 1 in the row of the chosen action.

    Each minimum triggers a drop on its own: drop iff
    burning >= burning_min or asset_damage >= damage_min or burned >= burned_min.
    A minimum of None is switched off.

    The table has no memory, so the time between drops is enforced by the
    env (WildfireEnv.min_drop_interval), not here.
    """
    values = state_domain.index2values(np.arange(len(state_domain)))
    labels = list(state_domain.var2values)

    drop = np.zeros(len(state_domain), dtype=bool)
    for label, vmin in (("burning", burning_min), ("asset_damage", damage_min),
                        ("burned", burned_min)):
        if vmin is not None:
            drop |= values[:, labels.index(label)] >= vmin

    table = np.zeros((2, len(state_domain)))
    table[DROP, drop] = 1.0
    table[NOOP, ~drop] = 1.0
    return table


class QoITablePolicy(TabularPolicy):
    """pgmtwin TabularPolicy with the table written from threshold_table.

    With use_map_state=False the observed state_distro is weighted against
    the table, so it drops when most of the probability mass sits in drop
    states; with use_map_state=True it reads the most likely state's column.
    """

    def __init__(self, state_domain, actions, burning_min: Optional[float] = 0.02,
                 damage_min: Optional[float] = None, burned_min: Optional[float] = None,
                 use_map_state: bool = False, rng=None):
        super().__init__(state_domain, actions, reward=None,
                         use_map_state=use_map_state, rng=rng)
        self.policy = threshold_table(state_domain, burning_min, damage_min, burned_min)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        action, _ = super().predict(np.asarray(observation), state, episode_start, deterministic)
        return (int(action) if np.ndim(action) == 0 else action), state


def write_policy_table(policy: TabularPolicy, path: str):
    dom = policy.state_domain
    values = dom.index2values(np.arange(len(dom)))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", *dom.var2values, "action"])
        for s, v in enumerate(values):
            w.writerow([s, *v, policy.actions[int(np.argmax(policy.policy[:, s]))].name])


if __name__ == "__main__":
    from modified_config import actions, dom

    policy = QoITablePolicy(dom, actions)
    write_policy_table(policy, "policy_table.csv")
    print(f"wrote policy_table.csv: {int(policy.policy[DROP].sum())} of "
          f"{len(dom)} states drop")
