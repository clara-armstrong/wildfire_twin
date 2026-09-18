"""The inverse problem: recover theta = (x0, y0, wind_speed, wind_direction)
from coarse, noisy observations of the fire.

Because the latent space is small (588 states by default), the posterior
is computed exactly by enumeration:

    p(theta | y_1:t)  proportional to  p(theta) * prod_k p(y_k | theta)

theta is static, so there's no predict step, just repeated multiplication
of likelihoods. H(theta, t) doesn't depend on the data, so it's
precomputed once offline (ForwardCache) and reused; assimilation is then
a table lookup and a vectorised norm.

"""
from dataclasses import dataclass, fields
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .config import ObservationConfig, WildfireConfig
from .domain import LatentDomain
from .observation import coarsen, observation_dim, observation_times, observe
from .solver import UNBURNED, Latent, Wildfire


@dataclass
class ForwardCache:
    # obs : (n_latent, n_obs_times, obs_dim) -- noiseless observations.
    # burn : (n_latent, n_risk_steps, C, C) -- coarse burn indicator.
    # asset_burnt, burnt_fraction : (n_latent, n_risk_steps) -- scalar QoIs.
    obs: np.ndarray
    burn: np.ndarray
    asset_burnt: np.ndarray
    burnt_fraction: np.ndarray
    obs_steps: np.ndarray
    risk_steps: np.ndarray
    grid_size: int
    coarse_size: int

    def save(self, path) -> None:
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path) -> "ForwardCache":
        d = np.load(path)
        ints = {"grid_size", "coarse_size"}
        return cls(**{f.name: (int(d[f.name]) if f.name in ints else d[f.name]) for f in fields(cls)})


_W: Dict[str, object] = {}  # per-worker globals, set by _init_worker


def _init_worker(config, obs_config, obs_steps, risk_steps):
    _W["config"] = config
    _W["obs_config"] = obs_config
    _W["obs_steps"] = set(int(s) for s in obs_steps)
    _W["risk_steps"] = set(int(s) for s in risk_steps)
    _W["n_steps"] = int(max(list(obs_steps) + list(risk_steps)))


def _run_one(args):
    idx, latent_tuple = args
    config: WildfireConfig = _W["config"]
    obs_config: ObservationConfig = _W["obs_config"]
    fire = Wildfire(config, Latent(*latent_tuple))
    obs_rows, burn_rows, asset_rows, frac_rows = [], [], [], []

    for step in range(1, _W["n_steps"] + 1):
        fire.step()
        if step in _W["obs_steps"]:
            obs_rows.append(observe(fire.state_dict(), obs_config, noiseless=True))
        if step in _W["risk_steps"]:
            burned = (fire.burn_status > UNBURNED).astype(np.float32)
            burn_rows.append(coarsen(burned, obs_config.coarse_size).astype(np.float32))
            asset_rows.append(np.float32(fire.burnt_asset_area()))
            frac_rows.append(np.float32(burned.mean()))

    return (idx, np.stack(obs_rows), np.stack(burn_rows),
            np.asarray(asset_rows, dtype=np.float32), np.asarray(frac_rows, dtype=np.float32))


def _with_progress(results: Iterable, total: int, progress: bool) -> Iterable:
    for done, item in enumerate(results, start=1):
        if progress and done % 50 == 0:
            print(f"  forward cache {done}/{total}", flush=True)
        yield item


def build_forward_cache(
    config: WildfireConfig,
    domain: LatentDomain,
    obs_config: ObservationConfig,
    risk_steps: Optional[Sequence[int]] = None,
    n_workers: int = 1,
    progress: bool = True,
) -> ForwardCache:
    obs_steps = observation_times(obs_config)
    risk_steps = np.asarray(sorted(set(int(s) for s in (risk_steps if risk_steps is not None else obs_steps))))
    n = len(domain)
    tasks = [(i, (l.x0, l.y0, l.wind_speed, l.wind_direction_deg)) for i, l in enumerate(domain.latents())]

    obs = np.zeros((n, len(obs_steps), observation_dim(obs_config)), dtype=np.float32)
    burn = np.zeros((n, len(risk_steps), obs_config.coarse_size, obs_config.coarse_size), dtype=np.float32)
    asset = np.zeros((n, len(risk_steps)), dtype=np.float32)
    frac = np.zeros((n, len(risk_steps)), dtype=np.float32)

    def scatter(results):
        # By index, not arrival: imap_unordered returns rows out of order.
        # Consumed as results arrive so only one row is held at a time.
        for i, o, b, a, f in results:
            obs[i], burn[i], asset[i], frac[i] = o, b, a, f

    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers, initializer=_init_worker,
                      initargs=(config, obs_config, obs_steps, risk_steps)) as pool:
            # Must drain inside the with block: the pool shuts down on exit.
            scatter(_with_progress(pool.imap_unordered(_run_one, tasks, chunksize=4), n, progress))
    else:
        _init_worker(config, obs_config, obs_steps, risk_steps)
        scatter(_with_progress(map(_run_one, tasks), n, progress))

    return ForwardCache(obs=obs, burn=burn, asset_burnt=asset, burnt_fraction=frac,
                         obs_steps=np.asarray(obs_steps), risk_steps=risk_steps,
                         grid_size=config.grid_size, coarse_size=obs_config.coarse_size)


class InverseSolver:
    # Bayesian inversion by enumeration: every theta on the grid is scored,
    # so there is no sampling error, but the likelihood is tempered, so the
    # result is not the literal Bayes posterior. The likelihood is Gaussian
    # with sigma_total^2 = noise_std^2 + model_error_std^2. model_error_std
    # covers the cache not being truth; temperature flattens the likelihood,
    # since the coarse observation's components are correlated rather than
    # independent. Both are set empirically in validation.py.

    def __init__(
        self,
        domain: LatentDomain,
        cache: ForwardCache,
        obs_config: ObservationConfig,
        model_error_std: float = 0.06,
        temperature: float = 60.0,
        discrepancy: Optional[np.ndarray] = None,
    ):
        self.domain = domain
        self.cache = cache
        self.obs_config = obs_config
        self.model_error_std = float(model_error_std)
        self.temperature = float(temperature)

        if discrepancy is not None:
            discrepancy = np.asarray(discrepancy, dtype=np.float32)
            if discrepancy.shape != cache.obs.shape[1:]:
                raise ValueError(
                    f"discrepancy has shape {discrepancy.shape}, expected {cache.obs.shape[1:]}"
                )
        self.discrepancy = discrepancy
        self.reset()

    def reset(self) -> None:
        self._log_w = np.log(self.domain.prior())
        self.n_assimilated = 0

    @property
    def sigma_total(self) -> float:
        return float(np.hypot(self.obs_config.noise_std, self.model_error_std))

    @property
    def posterior(self) -> np.ndarray:
        w = np.exp(self._log_w - self._log_w.max())
        return w / w.sum()

    @property
    def effective_sample_size(self) -> float:
        w = self.posterior
        return float(1.0 / np.sum(w ** 2))

    def log_likelihood(self, y: np.ndarray, obs_index: int) -> np.ndarray:
        pred = self.cache.obs[:, obs_index, :]
        if self.discrepancy is not None:
            pred = pred + self.discrepancy[obs_index][None, :]
        resid = pred - np.asarray(y, dtype=np.float32)[None, :]
        sse = np.einsum("ij,ij->i", resid, resid)
        return -0.5 * sse / (self.sigma_total ** 2 * self.temperature)

    def update(self, y: np.ndarray, obs_index: Optional[int] = None) -> np.ndarray:
        if obs_index is None:
            obs_index = self.n_assimilated
        if obs_index >= self.cache.obs.shape[1]:
            raise IndexError(
                f"obs_index {obs_index} exceeds the {self.cache.obs.shape[1]} observation times in the cache"
            )
        self._log_w = self._log_w + self.log_likelihood(y, obs_index)
        self._log_w -= self._log_w.max()
        self.n_assimilated += 1
        return self.posterior

    def assimilate_sequence(self, ys: Sequence[np.ndarray]) -> np.ndarray:
        self.reset()
        for k, y in enumerate(ys):
            self.update(y, obs_index=k)
        return self.posterior

    def map_latent(self) -> Latent:
        return self.domain.latent(int(np.argmax(self.posterior)))

    def posterior_mean_latent(self) -> Dict[str, Dict[str, float]]:
        return self.domain.summarise(self.posterior)

    def _risk_index(self, step: int) -> int:
        matches = np.where(self.cache.risk_steps == step)[0]
        if matches.size == 0:
            raise KeyError(f"step {step} is not a cached risk step; available: {self.cache.risk_steps.tolist()}")
        return int(matches[0])

    def risk_map(self, step: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
        w = self.posterior if weights is None else np.asarray(weights, dtype=float)
        return np.tensordot(w, self.cache.burn[:, self._risk_index(step)], axes=(0, 0))

    def prior_risk_map(self, step: int) -> np.ndarray:
        return self.risk_map(step, weights=self.domain.prior())

    @staticmethod
    def _mean_std(w: np.ndarray, vals: np.ndarray) -> Tuple[float, float]:
        mean = float(w @ vals)
        return mean, float(np.sqrt(max(w @ (vals - mean) ** 2, 0.0)))

    def expected_asset_damage(self, step: int) -> Tuple[float, float]:
        return self._mean_std(self.posterior, self.cache.asset_burnt[:, self._risk_index(step)])

    def credible_interval(self, step: int, level: float = 0.9) -> Tuple[float, float]:
        w = self.posterior
        vals = self.cache.asset_burnt[:, self._risk_index(step)]
        order = np.argsort(vals)
        v, cw = vals[order], np.cumsum(w[order])
        lo = float(v[np.searchsorted(cw, (1 - level) / 2)])
        hi = float(v[min(np.searchsorted(cw, 1 - (1 - level) / 2), len(v) - 1)])
        return lo, hi

    def qois(self, step: int) -> Dict[str, float]:
        mean, std = self.expected_asset_damage(step)
        lo, hi = self.credible_interval(step)
        frac_mean, frac_std = self._mean_std(self.posterior, self.cache.burnt_fraction[:, self._risk_index(step)])
        return {
            "asset_damage_mean": mean, "asset_damage_std": std,
            "asset_damage_lo90": lo, "asset_damage_hi90": hi,
            "burnt_fraction_mean": frac_mean, "burnt_fraction_std": frac_std,
            "effective_sample_size": self.effective_sample_size,
            "n_assimilated": float(self.n_assimilated),
        }
