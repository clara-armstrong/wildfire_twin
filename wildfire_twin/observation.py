"""The observation operator H, and named (de)serialisation of state.

"""
from typing import Dict, Optional

import numpy as np

from .config import STATE_FIELDS, OBSERVATION_SCALE, ObservationConfig

_INT_FIELDS = {"burn_status"}


# Coarse graining

def coarsen(field: np.ndarray, coarse_size: int) -> np.ndarray:
    # Block-average a (g, g) field down to (coarse_size, coarse_size). Same
    # technique as the reference repo's probabilisticMap.compute_probabilites.
    #
    # In plain terms: chop the grid into equal square blocks and replace each
    # block with its average. E.g. a 4x4 grid coarsened to 2x2 averages each
    # 2x2 sub-block into one number:
    #   [[1,1,2,2],        [[1.0, 2.0],
    #    [1,1,2,2],   ->     [3.0, 4.0]]
    #    [3,3,4,4],
    #    [3,3,4,4]]
    g = field.shape[0]
    if g % coarse_size != 0:
        raise ValueError(
            f"grid_size={g} is not divisible by coarse_size={coarse_size}; "
            "the observation operator needs an integer block size"
        )
    block = g // coarse_size
    return field.reshape(coarse_size, block, coarse_size, block).mean(axis=(1, 3))


def upsample(field: np.ndarray, grid_size: int) -> np.ndarray:
    # Inverse of coarsen: broadcast (c, c) back up to (grid_size, grid_size)
    # by repeating each block. Used to compare a coarse-grid quantity (e.g.
    # the posterior risk map) against fine-grid fields, as policy.py's
    # RuleBasedPolicy does.
    c = field.shape[0]
    if grid_size % c != 0:
        raise ValueError(
            f"grid_size={grid_size} is not divisible by coarse_size={c}; "
            "cannot upsample without a fractional block size"
        )
    block = grid_size // c
    return np.repeat(np.repeat(field, block, axis=0), block, axis=1)


def observation_channels(
    state: Dict[str, np.ndarray], obs_config: ObservationConfig
) -> Dict[str, np.ndarray]:
    # Map a full state dict onto the observable channels, on the coarse grid.
    #
    # - temperature  -> block-mean temperature, scaled to ~[0, 1]
    # - burn_status  -> block fraction of cells burned or burning (what a
    #   satellite burn-scar product returns, not the raw three-level code)
    out: Dict[str, np.ndarray] = {}
    for name in obs_config.channels:
        if name == "burn_status":
            field = (np.asarray(state["burn_status"]) > 0).astype(np.float32)
        else:
            field = np.asarray(state[name], dtype=np.float32) / OBSERVATION_SCALE[name]
        out[name] = coarsen(field, obs_config.coarse_size).astype(np.float32)
    return out


def observe(
    state: Dict[str, np.ndarray],
    obs_config: ObservationConfig,
    rng: Optional[np.random.Generator] = None,
    noiseless: bool = False,
) -> np.ndarray:
    # Apply H to a state -> flat observation vector. noiseless=True is used
    # only when building the forward cache (inverse.py): the cache stores
    # H(theta) itself, and noise is added fresh at assimilation time.
    channels = observation_channels(state, obs_config)
    obs = np.concatenate([channels[name].ravel() for name in obs_config.channels])
    if not noiseless and obs_config.noise_std > 0:
        rng = rng or np.random.default_rng()
        obs = obs + rng.normal(0.0, obs_config.noise_std, size=obs.shape)
    return obs.astype(np.float32)


def observation_dim(obs_config: ObservationConfig) -> int:
    return obs_config.coarse_size ** 2 * len(obs_config.channels)


def observation_times(obs_config: ObservationConfig) -> np.ndarray:
    # Step indices at which observations are assimilated.
    return np.arange(1, obs_config.n_observations + 1) * obs_config.observe_every


# Flat state serialisation (raw, for pgmtwin's internal state array)

def _pack(state: Dict[str, np.ndarray], grid_size: int, scale: Dict[str, float]) -> np.ndarray:
    # Flatten STATE_FIELDS into one 1D array, each field divided by `scale`.
    # Raises loudly on a missing field or wrong shape, instead of silently
    # producing a malformed array.
    arrays = []
    for name in STATE_FIELDS:
        if name not in state:
            raise KeyError(f"state is missing required field '{name}'")
        arr = np.asarray(state[name], dtype=np.float32)
        if arr.shape != (grid_size, grid_size):
            raise ValueError(
                f"state['{name}'] has shape {arr.shape}, expected ({grid_size}, {grid_size})"
            )
        arrays.append((arr / scale[name]).ravel())
    return np.concatenate(arrays)


def _unpack(flat: np.ndarray, grid_size: int, scale: Dict[str, float]) -> Dict[str, np.ndarray]:
    # Inverse of _pack. Raises loudly on a length mismatch (usually a
    # different grid_size) instead of reshaping into garbage.
    flat = np.asarray(flat).reshape(-1)
    cell = grid_size * grid_size
    expected = cell * len(STATE_FIELDS)
    if flat.shape[0] != expected:
        raise ValueError(
            f"Got a flat array of length {flat.shape[0]}, expected {expected} "
            f"({len(STATE_FIELDS)} fields x {cell} cells for grid_size={grid_size}). "
            "This usually means the array was produced with a different grid_size."
        )
    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(STATE_FIELDS):
        chunk = flat[i * cell:(i + 1) * cell].reshape(grid_size, grid_size) * scale[name]
        result[name] = chunk.astype(int) if name in _INT_FIELDS else chunk
    return result


_RAW_SCALE = {name: 1.0 for name in STATE_FIELDS}


def pack_state(state: Dict[str, np.ndarray], grid_size: int) -> np.ndarray:
    # Flatten a {field: (g, g) array} dict into pgmtwin's raw internal
    # layout (unscaled -- for state storage, not observe()'s [0,1] scaling).
    return _pack(state, grid_size, _RAW_SCALE)


def unpack_state(flat: np.ndarray, grid_size: int) -> Dict[str, np.ndarray]:
    # Inverse of pack_state.
    return _unpack(flat, grid_size, _RAW_SCALE)
