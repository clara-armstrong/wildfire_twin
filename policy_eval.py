import numpy as np


def run_episode(env, decide, seed=None, verbose=True):
    """Run one episode, choosing each action with either a policy that has
    predict(obs, state, episode_start) (as in modified_policy and sb3), or a
    function decide(env, obs) -> action index.

    Returns the total reward.
    """
    actions = env._actions

    obs, info = env.reset(seed=seed)
    done, total = False, 0.0
    state, start = None, True
    while not done:
        if hasattr(decide, "predict"):
            a, state = decide.predict(obs, state=state, episode_start=[start])
            start = False
        else:
            a = decide(env, obs)
        a = int(a)
        obs, r, terminated, truncated, info = env.step(a)
        total += r
        done = terminated or truncated
        if verbose:
            print(f"t={info['t']:5.1f}  {actions[a].name:8s}  qois={np.round(info['qois'], 3)}")
    if verbose:
        print("total reward", round(total, 3))
    return total
