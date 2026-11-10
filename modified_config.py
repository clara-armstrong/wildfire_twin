import numpy as np
from pgmtwin.core.domain import DiscreteDomain
from modified_physical_asset import QOI_LABELS, NoOpAction, PriorityDropAction, WildfireAsset
from modified_digital_asset import WildfireDigitalAsset
from modified_env import WildfireEnv

"""Shared setup between multiple files"""
# digital state
bins = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0])
dom = DiscreteDomain({label: bins for label in QOI_LABELS})

# actions: the policy picks when to drop; the priority index picks where
actions = [NoOpAction(dom), PriorityDropAction(dom)]

def reward(action, state):
    """Reward = -(asset damage in the landing state).

    Right now, reward only depends on the asset damage
    """
    return -float(state[QOI_LABELS.index("asset_damage")])


# Gaussian noise on the QoIs the digital asset sees
qoi_noise_std = 0.01
n_qoi_samples = 20


def make_env(seed=None, qoi_noise_std=qoi_noise_std, n_qoi_samples=n_qoi_samples,
             **physical_kwargs):
    rng = np.random.default_rng(seed)
    physical = WildfireAsset(dom, rng=rng, **physical_kwargs)
    sim = physical.sim
    digital = WildfireDigitalAsset(
        dom, rng=rng, asset_map=sim.asset_value, coords=sim.coords, wind=sim.sc.wind_vec,
        qoi_noise_std=qoi_noise_std, n_qoi_samples=n_qoi_samples,
    )
    return WildfireEnv(dom, actions, physical, digital, reward, max_episode_steps=200)
