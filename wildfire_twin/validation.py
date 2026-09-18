"""Validation for the inverse solver.

"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .config import ObservationConfig, WildfireConfig
from .domain import LatentDomain
from .inverse import ForwardCache, InverseSolver
from .observation import observe
from .solver import UNBURNED, Latent, Wildfire


@dataclass
class TruthRun:
    # One ground-truth trajectory and the observations taken from it.
    latent: Latent
    observations: List[np.ndarray]
    asset_burnt: Dict[int, float] = field(default_factory=dict)
    burnt_fraction: Dict[int, float] = field(default_factory=dict)


def sample_prior_latent(
    config: WildfireConfig, domain: LatentDomain, rng: np.random.Generator
) -> Latent:
    # Draw theta* continuously: Eq. 5.3 for location, grid range for wind.
    s = config.ignition_sigma
    speeds = domain.var2values["wind_speed"]
    dirs = domain.var2values["wind_direction"]
    return Latent(
        x0=float(config.ignition_center_x + rng.uniform(-s, s)),
        y0=float(config.ignition_center_y + rng.uniform(-s, s)),
        wind_speed=float(rng.uniform(speeds.min(), speeds.max())),
        wind_direction_deg=float(rng.uniform(dirs.min(), dirs.max())),
    )


def generate_truth(
    truth_config: WildfireConfig,
    latent: Latent,
    obs_config: ObservationConfig,
    obs_steps: Sequence[int],
    risk_steps: Sequence[int],
    rng: np.random.Generator,
) -> TruthRun:
    # Run one ground-truth trajectory and observe it.
    obs_steps = set(int(s) for s in obs_steps)
    risk_steps = set(int(s) for s in risk_steps)
    n_steps = max(obs_steps | risk_steps)

    fire = Wildfire(truth_config, latent)
    run = TruthRun(latent=latent, observations=[])
    for step in range(1, n_steps + 1):
        fire.step()
        if step in obs_steps:
            run.observations.append(observe(fire.state_dict(), obs_config, rng=rng))
        if step in risk_steps:
            run.asset_burnt[step] = fire.burnt_asset_area()
            run.burnt_fraction[step] = float((fire.burn_status > UNBURNED).mean())
    return run


def generate_truth_set(
    truth_config: WildfireConfig,
    domain: LatentDomain,
    obs_config: ObservationConfig,
    cache: ForwardCache,
    n_truths: int = 24,
    seed: int = 0,
) -> List[TruthRun]:
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_truths):
        theta = sample_prior_latent(domain.config, domain, rng)
        runs.append(
            generate_truth(truth_config, theta, obs_config,
                           cache.obs_steps, cache.risk_steps, rng)
        )
    return runs


# Scoring

def _angular_error(a: float, b: float) -> float:
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def score_runs(
    runs: Sequence[TruthRun],
    domain: LatentDomain,
    cache: ForwardCache,
    obs_config: ObservationConfig,
    model_error_std: float,
    temperature: float,
    risk_step: int,
    level: float = 0.9,
) -> Dict[str, float]:
    # Assimilate each truth run and score accuracy and calibration.
    solver = InverseSolver(domain, cache, obs_config,
                           model_error_std=model_error_std, temperature=temperature)

    err_x, err_y, err_dir, ess = [], [], [], []
    covered, interval_width, damage_err = [], [], []
    prior_damage_err = []

    prior = domain.prior()
    k = int(np.where(cache.risk_steps == risk_step)[0][0])
    prior_damage = float(prior @ cache.asset_burnt[:, k])

    for run in runs:
        solver.assimilate_sequence(run.observations)
        s = solver.posterior_mean_latent()

        err_x.append(s["x0"]["mean"] - run.latent.x0)
        err_y.append(s["y0"]["mean"] - run.latent.y0)
        err_dir.append(_angular_error(s["wind_direction"]["mean"],
                                      run.latent.wind_direction_deg))
        ess.append(solver.effective_sample_size)

        lo, hi = solver.credible_interval(risk_step, level=level)
        truth_damage = run.asset_burnt[risk_step]
        covered.append(lo <= truth_damage <= hi)
        interval_width.append(hi - lo)

        mean, _ = solver.expected_asset_damage(risk_step)
        damage_err.append(mean - truth_damage)
        prior_damage_err.append(prior_damage - truth_damage)

    err_x, err_y = np.asarray(err_x), np.asarray(err_y)
    damage_err = np.asarray(damage_err)
    prior_damage_err = np.asarray(prior_damage_err)

    return {
        "temperature": temperature,
        "model_error_std": model_error_std,
        "rmse_x0": float(np.sqrt((err_x ** 2).mean())),
        "rmse_y0": float(np.sqrt((err_y ** 2).mean())),
        "bias_x0": float(err_x.mean()),
        "bias_y0": float(err_y.mean()),
        "mae_wind_dir": float(np.mean(err_dir)),
        "mean_ess": float(np.mean(ess)),
        "coverage": float(np.mean(covered)),
        "nominal": level,
        "mean_interval_width": float(np.mean(interval_width)),
        "rmse_damage_posterior": float(np.sqrt((damage_err ** 2).mean())),
        "rmse_damage_prior": float(np.sqrt((prior_damage_err ** 2).mean())),
    }


def calibrate(
    runs: Sequence[TruthRun],
    domain: LatentDomain,
    cache: ForwardCache,
    obs_config: ObservationConfig,
    risk_step: int,
    temperatures: Sequence[float],
    model_error_std: float = 0.11,
    level: float = 0.9,
) -> List[Dict[str, float]]:
    # Sweep the likelihood temperature and report calibration at each value.
    #
    # The truth runs are simulated once and reused, so the sweep is nearly
    # free. Pick the smallest temperature whose coverage reaches the nominal
    # level: larger temperatures are safe but throw away information.
    return [
        score_runs(runs, domain, cache, obs_config, model_error_std, T, risk_step, level)
        for T in temperatures
    ]


def sharpening_curve(
    runs: Sequence[TruthRun],
    domain: LatentDomain,
    cache: ForwardCache,
    obs_config: ObservationConfig,
    model_error_std: float,
    temperature: float,
) -> List[Dict[str, float]]:
    # Posterior spread vs. number of observations assimilated. Should fall
    # monotonically as data accumulates -- the checkable version of "the
    # posterior sharpens" claimed in digital_asset.py/priority.py.
    n_obs = cache.obs.shape[1]
    out = []
    for n in range(0, n_obs + 1):
        solver = InverseSolver(domain, cache, obs_config,
                               model_error_std=model_error_std, temperature=temperature)
        stds_x, stds_y, errs, esss = [], [], [], []
        for run in runs:
            solver.assimilate_sequence(run.observations[:n])
            s = solver.posterior_mean_latent()
            stds_x.append(s["x0"]["std"])
            stds_y.append(s["y0"]["std"])
            errs.append(np.hypot(s["x0"]["mean"] - run.latent.x0,
                                 s["y0"]["mean"] - run.latent.y0))
            esss.append(solver.effective_sample_size)
        out.append({
            "n_observations": n,
            "posterior_std_x0": float(np.mean(stds_x)),
            "posterior_std_y0": float(np.mean(stds_y)),
            "rmse_location": float(np.sqrt(np.mean(np.square(errs)))),
            "mean_ess": float(np.mean(esss)),
        })
    return out


def grid_resolution_report(domain: LatentDomain, cache: ForwardCache) -> Dict[str, float]:
    # RMS change in H per one-step move along each latent axis: the
    # diagnostic for whether the latent grid is fine enough. If one axis's
    # half-step error in H exceeds another axis's per-step signal, that
    # axis's discretisation error gets absorbed as bias in the other.
    #
    # Rule of thumb: for every axis pair (a, b), 0.5*signal[a] < signal[b]
    # should hold both ways, or the coarser axis needs refining.
    from .domain import VAR_NAMES

    O = cache.obs
    Or = O.reshape(*domain.shape, O.shape[1], O.shape[2])
    out: Dict[str, float] = {}
    for axis, name in enumerate(VAR_NAMES):
        if Or.shape[axis] < 2:
            out[name] = float("nan")
            continue
        d = np.diff(Or, axis=axis)
        out[name] = float(np.sqrt((d ** 2).mean()))
    return out
