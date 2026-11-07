from modified_config import make_env
from policy_eval import run_episode

env = make_env(seed=0)


def sensor_rule(env, obs):
    """Priority drop once any sensor reads hotter than forest ignition."""
    obs = env._digital_asset.history[-1]["sensor_obs"].mean(axis=0)   # what the twin sensed
    temps = obs[: obs.size // 2]   # second half is burning flags
    return 1 if temps.max() > 3.5 else 0


run_episode(env, sensor_rule, seed=0)
