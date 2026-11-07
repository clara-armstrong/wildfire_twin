"""Spatial priority index for water-drop placement based off of Krish's thesis 
and my original draft
"""

from typing import Dict, Optional, Tuple

import numpy as np


def _asset_positions(asset_map: np.ndarray) -> np.ndarray:
    # Coordinates of every cell with nonzero asset value.
    return np.argwhere(asset_map > 0).astype(float)


def gradient_field(risk: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    # G(x) = -(1/2) (grad R + grad T), thesis Eq. 5.4. Points from
    # high-risk/hot areas toward low-risk/cool ones, i.e. the way heat flows.
    # Returns an (n, m, 2) array.
    gr = np.gradient(risk)
    gt = np.gradient(temperature)
    gx = -0.5 * (gr[0] + gt[0])
    gy = -0.5 * (gr[1] + gt[1])
    return np.stack([gx, gy], axis=-1)


def alignment(
    G: np.ndarray, wind: Tuple[float, float], asset_map: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    # L(x), thesis Eq. 5.5. For each asset point a, the mean of the cosines
    # between (a - x) and the wind and between (a - x) and G(x), divided by
    # |a - x|^2, summed over assets. High where the fire is headed through x
    # toward something valuable.
    n, m, _ = G.shape
    assets = _asset_positions(asset_map)
    if assets.shape[0] == 0:
        return np.zeros((n, m))

    ii, jj = np.indices((n, m))
    pos = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)  # (P, 2)

    d = assets[None, :, :] - pos[:, None, :]  # (P, K, 2)
    dist = np.linalg.norm(d, axis=-1)  # (P, K)
    safe = np.maximum(dist, eps)

    w = np.asarray(wind, dtype=float)
    w = w / max(np.linalg.norm(w), eps)

    cos_w = (d @ w) / safe

    g_flat = G.reshape(-1, 2)  # (P, 2)
    g_norm = np.maximum(np.linalg.norm(g_flat, axis=1), eps)
    cos_g = np.einsum("pkc,pc->pk", d, g_flat) / (safe * g_norm[:, None])

    contrib = (cos_w + cos_g) / (2.0 * safe**2)
    contrib[dist < eps] = 0.0  # x is itself an asset
    return contrib.sum(axis=1).reshape(n, m)


def asset_importance_term(
    risk: np.ndarray, asset_map: np.ndarray, p: int = 20, eps: float = 1e-9
) -> np.ndarray:
    # zeta(x) = A(x) R(x)^2 / mu(x), thesis Eqs. 5.6-5.7, where mu(x) is the
    # mean squared distance to the p nearest asset points. Large at valuable,
    # at-risk, clustered asset cells.
    n, m = risk.shape
    assets = _asset_positions(asset_map)
    if assets.shape[0] == 0:
        return np.zeros((n, m))

    ii, jj = np.indices((n, m))
    pos = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)

    d2 = ((assets[None, :, :] - pos[:, None, :]) ** 2).sum(axis=-1)  # (P, K)
    p_eff = min(p, d2.shape[1])
    nearest = np.sort(d2, axis=1)[:, :p_eff]
    mu = np.maximum(nearest.mean(axis=1), eps).reshape(n, m)  # Eq. 5.6

    return asset_map * risk**2 / mu  # Eq. 5.7


def priority_index(
    risk: np.ndarray,
    temperature: np.ndarray,
    asset_map: np.ndarray,
    wind: Tuple[float, float],
    lam: float = 0.1,
    p: int = 20,
) -> Dict[str, np.ndarray]:
    # P(x) = ||G(x)|| L(x) + lambda * zeta(x) where ||G|| > 0, else 0
    # (Eq. 5.8). lam=0.1 and p=20 are the thesis's values.
    G = gradient_field(risk, temperature)
    g_norm = np.linalg.norm(G, axis=-1)
    L = alignment(G, wind, asset_map)
    zeta = asset_importance_term(risk, asset_map, p=p)

    P = np.where(g_norm > 0, g_norm * L + lam * zeta, 0.0)
    return {"priority": P, "gradient": G, "gradient_norm": g_norm,
            "alignment": L, "zeta": zeta}


def shift_out_of_risk(
    loc: Tuple[int, int], direction: np.ndarray, risk: np.ndarray, threshold: float = 1e-3
) -> Tuple[int, int]:
    # Step from loc along direction until risk drops below threshold, so the
    # drop lands ahead of the fire instead of on it. Stops at the grid edge.
    norm = np.linalg.norm(direction)
    if norm == 0:
        return loc
    step = direction / norm

    n, m = risk.shape
    pos = np.array(loc, dtype=float)
    i, j = loc
    while risk[i, j] >= threshold:
        pos += step
        ni, nj = int(round(pos[0])), int(round(pos[1]))
        if not (0 <= ni < n and 0 <= nj < m):
            break
        i, j = ni, nj
    return int(i), int(j)


def optimal_drop(
    risk: np.ndarray,
    temperature: np.ndarray,
    asset_map: np.ndarray,
    wind: Tuple[float, float],
    lam: float = 0.1,
    p: int = 20,
    shift_threshold: Optional[float] = 1e-3,
) -> Optional[Dict[str, object]]:
    # Drop at argmax P, shifted out of the risky area, with the angle
    # bisecting the wind and G at the argmax (thesis get_initial_guess).
    # Returns None when no cell has positive priority.
    fields = priority_index(risk, temperature, asset_map, wind, lam=lam, p=p)
    P = fields["priority"]
    if P.max() <= 0:
        return None

    peak = np.unravel_index(int(np.argmax(P)), P.shape)
    G = fields["gradient"][peak]

    loc = peak
    if shift_threshold is not None:
        loc = shift_out_of_risk(peak, G, risk, threshold=shift_threshold)

    theta_w = np.arctan2(wind[1], wind[0])
    theta_g = np.arctan2(G[1], G[0])
    diff = (theta_g - theta_w + np.pi) % (2 * np.pi) - np.pi  # circular mean
    angle = (theta_w + 0.5 * diff) % (2 * np.pi)

    return {"location": (int(loc[0]), int(loc[1])),
            "peak": (int(peak[0]), int(peak[1])),
            "angle": float(angle),  # radians from +x
            "priority_value": float(P[peak]),
            "fields": fields}
