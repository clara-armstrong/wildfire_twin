import csv

from modified_config import make_env
from policy_eval import run_episode


def export_sensors(env, path="sensor_readings.csv", every=1):
    """Write the sensor readings from the env's last episode to a CSV,
    one row per (step, observation, sensor), keeping only every
    `every`-th step (0, every, 2*every, ...)."""
    coords = env._physical_asset.sensor_coords
    n_sensors = len(coords)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t", "action", "observation", "sensor", "x", "y",
                    "temperature", "burning"])
        for step, h in enumerate(env._digital_asset.history):
            if step % every:
                continue
            for k, obs in enumerate(h["sensor_obs"]):
                temps, flags = obs[:n_sensors], obs[n_sensors:]
                for s, (x, y) in enumerate(coords):
                    w.writerow([step, round(h["t"], 4), h["action"] or "", k, s, x, y,
                                round(float(temps[s]), 4), int(flags[s])])
    return path


if __name__ == "__main__":
    env = make_env(seed=0)
    run_episode(env, lambda env, obs: 0, seed=0, verbose=False)   # no drops
    print("wrote", export_sensors(env, every=3))
