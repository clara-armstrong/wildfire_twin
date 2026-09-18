"""Spatial priority index for water-drop placement (thesis Section 5.2).

Kept from thesis: direct implementation of Eqs. 5.4-5.8, on the coarse
grid Omega-tilde (same grid the observation operator produces).

The priority index consumes a risk map R (Eq. 5.4 uses grad(R), Eq. 5.7
uses R^2). In the thesis R is a prior Monte Carlo average that never
sharpens no matter how long the fire is observed; digital_asset.
recommend_drop can feed it the posterior map instead.

"""
from typing import Dict, Tuple

import numpy as np


def _asset_positions(asset_map: np.ndarray) -> np.ndarray:
    # Coordinates of every cell with nonzero asset value.
    return np.argwhere(asset_map > 0).astype(float)


def gradient_field(risk: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    # G(x) = -(1/2) (grad R + grad T), thesis Eq. 5.4.
    # Returns an (n, m, 2) array.
    #
    # In plain terms: G(x) is a compass at every cell x, pointing from
    # high-risk/hot areas toward low-risk/cool ones, i.e. "which way does
    # a water drop need to push to move the fire toward safety".
    gr = np.gradient(risk)
    gt = np.gradient(temperature)
    gx = -0.5 * (gr[0] + gt[0])
    gy = -0.5 * (gr[1] + gt[1])
    return np.stack([gx, gy], axis=-1)


def alignment(
    G: np.ndarray, wind: Tuple[float, float], asset_map: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    # L(x), thesis Eq. 5.5. Average alignment between the gradient field, the
    # wind, and the directions to the asset points, inversely weighted by
    # squared distance.
    #
    # In plain terms, for ONE asset at position a and ONE candidate cell x
    # (the vectorised code below does this for every (x, a) pair at once):
    #
    #   to_asset = (a - x) / distance(a, x)                  # unit vector x -> a
    #   score = (to_asset . wind_direction + to_asset . G(x)) / (2 * distance(a,x)^2)
    #
    # summed over every asset a. A cell scores high when it is close to an
    # asset AND the wind and the risk/heat gradient both happen to point
    # toward that asset, i.e. "the fire is already headed toward something
    # valuable, right through here".
    n, m, _ = G.shape
    assets = _asset_positions(asset_map)
    if assets.shape[0] == 0:
        return np.zeros((n, m))

    ii, jj = np.indices((n, m))
    pos = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)   # (P, 2)

    d = assets[None, :, :] - pos[:, None, :]                          # (P, K, 2)
    dist = np.linalg.norm(d, axis=-1)                                 # (P, K)
    safe = np.maximum(dist, eps)

    w = np.asarray(wind, dtype=float)
    w = w / max(np.linalg.norm(w), eps)

    cos_w = (d @ w) / safe

    g_flat = G.reshape(-1, 2)                                         # (P, 2)
    g_norm = np.maximum(np.linalg.norm(g_flat, axis=1), eps)
    cos_g = np.einsum("pkc,pc->pk", d, g_flat) / (safe * g_norm[:, None])

    contrib = (cos_w + cos_g) / (2.0 * safe ** 2)
    contrib[dist < eps] = 0.0                                         # x is itself an asset
    return contrib.sum(axis=1).reshape(n, m)


def asset_importance_term(
    risk: np.ndarray, asset_map: np.ndarray, p: int = 20, eps: float = 1e-9
) -> np.ndarray:
    # zeta(x) = A(x) R(x)^2 / mu(x), thesis Eqs. 5.6-5.7.
    #
    # In plain terms: zeta(x) is large when x is (a) itself a valuable asset
    # cell (A(x) > 0), (b) at high risk of burning (R(x)^2), and (c) part of
    # a tight cluster of assets rather than an isolated one (mu(x), the mean
    # distance to the p nearest assets, is small). It's a "protect the
    # valuable, at-risk, clustered stuff first" score.
    n, m = risk.shape
    assets = _asset_positions(asset_map)
    if assets.shape[0] == 0:
        return np.zeros((n, m))

    ii, jj = np.indices((n, m))
    pos = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)

    d2 = ((assets[None, :, :] - pos[:, None, :]) ** 2).sum(axis=-1)   # (P, K)
    p_eff = min(p, d2.shape[1])
    nearest = np.sort(d2, axis=1)[:, :p_eff]
    mu = np.maximum(nearest.mean(axis=1), eps).reshape(n, m)          # Eq. 5.6

    return asset_map * risk ** 2 / mu                                 # Eq. 5.7


def priority_index(
    risk: np.ndarray,
    temperature: np.ndarray,
    asset_map: np.ndarray,
    wind: Tuple[float, float],
    lam: float = 0.1,
    p: int = 20,
) -> Dict[str, np.ndarray]:
    # P(x) = ||G(x)|| L(x) + lambda * zeta(x) where ||G|| > 0, else 0 (Eq.
    # 5.8). lam=0.1 and p=20 are the thesis's experimental values.
    G = gradient_field(risk, temperature)
    g_norm = np.linalg.norm(G, axis=-1)
    L = alignment(G, wind, asset_map)
    zeta = asset_importance_term(risk, asset_map, p=p)

    P = np.where(g_norm > 0, g_norm * L + lam * zeta, 0.0)
    return {"priority": P, "gradient": G, "gradient_norm": g_norm,
            "alignment": L, "zeta": zeta}


def optimal_drop(
    risk: np.ndarray,
    temperature: np.ndarray,
    asset_map: np.ndarray,
    wind: Tuple[float, float],
    lam: float = 0.1,
    p: int = 20,
) -> Dict[str, object]:
    # Optimal drop location x_s = argmax P(x), and the drop angle.
    #
    # The drop angle bisects the wind direction and the direction of -G at
    # the drop point, following the thesis's construction in Section 5.2.
    fields = priority_index(risk, temperature, asset_map, wind, lam=lam, p=p)
    P = fields["priority"]
    idx = int(np.argmax(P))
    loc = np.unravel_index(idx, P.shape)

    G = fields["gradient"][loc[0], loc[1]]
    theta_w = float(np.degrees(np.arctan2(wind[1], wind[0])) % 360.0)
    theta_g = float(np.degrees(np.arctan2(-G[1], -G[0])) % 360.0)

    diff = (theta_g - theta_w + 180.0) % 360.0 - 180.0
    theta_s = (theta_w + 0.5 * diff) % 360.0

    return {"location": (int(loc[0]), int(loc[1])),
            "angle_deg": theta_s,
            "priority_value": float(P[loc]),
            "fields": fields}
