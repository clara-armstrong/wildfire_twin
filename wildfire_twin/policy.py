"""Wildfire management policies.

Three policies, from trivial to practical: ``DoNothingPolicy`` (baseline),
``RandomPolicy`` (lower-bound baseline), and ``RuleBasedPolicy``. All
expose ``select_action(physical_state, digital_asset, step, actions)``.

"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
from pgmtwin.core.action import BaseAction

from .actions import (
    BackburnAction,
    DoNothingAction,
    EvacuationAction,
    FirebreakAction,
    WaterDropAction,
)
from .digital_asset import WildfireDigitalAsset
from .observation import upsample


# Base class

class BasePolicy:
    # Interface every policy must implement.

    def select_action(
        self,
        physical_state: Dict[str, np.ndarray],
        digital_asset: WildfireDigitalAsset,
        step: int,
        actions: List[BaseAction],
    ) -> BaseAction:
        # Return the action to apply this timestep.
        #
        # physical_state: the fine-grid, directly-observable fields (see
        #   WildfirePhysicalAsset.state_dict).
        # digital_asset: the current digital twin; its posterior risk map is
        #   available via digital_asset.risk_map(step).
        # step: current simulation step. Must be one of the digital asset's
        #   cached risk steps.
        # actions: the discrete action space built by build_default_actions.
        raise NotImplementedError


# Trivial policies


class DoNothingPolicy(BasePolicy):
    # Always return the do-nothing action: baseline / null policy.

    def select_action(
        self,
        physical_state: Dict[str, np.ndarray],
        digital_asset: WildfireDigitalAsset,
        step: int,
        actions: List[BaseAction],
    ) -> BaseAction:
        for a in actions:
            if isinstance(a, DoNothingAction):
                return a
        # Fall back to the first action if somehow DoNothingAction is absent.
        return actions[0]


class RandomPolicy(BasePolicy):
    # Sample uniformly from the action space: a random baseline.

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def select_action(
        self,
        physical_state: Dict[str, np.ndarray],
        digital_asset: WildfireDigitalAsset,
        step: int,
        actions: List[BaseAction],
    ) -> BaseAction:
        return self._rng.choice(actions)


# Rule-based policy


class RuleBasedPolicy(BasePolicy):
    # Priority-ordered rule-based policy for wildfire management.
    #
    # Rules (highest to lowest priority):
    # 1. Evacuate any asset region whose mean posterior risk exceeds
    #    evacuation_risk_threshold.
    # 2. Water drop on the hottest actively-burning cell.
    # 3. Firebreak on the highest-risk unburned cell within
    #    perimeter_buffer cells of the fire perimeter.
    # 4. Backburn on the highest-risk unburned cell farther from the
    #    perimeter (create a fuel break ahead of the fire).
    # 5. Do nothing: no urgent condition detected.
    #
    # evacuation_risk_threshold: posterior burn probability (0-1) in an
    #   asset cell that triggers evacuation.
    # perimeter_buffer: number of cells from the burning front within which
    #   a firebreak is preferred over a backburn.

    def __init__(
        self,
        evacuation_risk_threshold: float = 0.6,
        perimeter_buffer: int = 5,
    ):
        self.evacuation_risk_threshold = evacuation_risk_threshold
        self.perimeter_buffer = perimeter_buffer

    # Public interface

    def select_action(
        self,
        physical_state: Dict[str, np.ndarray],
        digital_asset: WildfireDigitalAsset,
        step: int,
        actions: List[BaseAction],
    ) -> BaseAction:
        g = physical_state["temperature"].shape[0]

        burn_status = physical_state.get("burn_status", np.zeros((g, g), dtype=int))
        temperature = physical_state.get("temperature", np.zeros((g, g)))
        fuel_fraction = physical_state.get("fuel_fraction", np.ones((g, g)))
        asset_importance = physical_state.get("asset_importance", np.zeros((g, g)))
        risk_level = upsample(digital_asset.risk_map(step), g)

        # Rule 1: evacuate high-risk asset zones
        evac = self._evacuation_action(
            burn_status, asset_importance, risk_level, actions, g
        )
        if evac is not None:
            return evac

        # Rule 2: suppress the hottest burning cell
        drop = self._water_drop_action(burn_status, temperature, actions)
        if drop is not None:
            return drop

        # Rules 3 & 4: firebreak / backburn
        contain = self._containment_action(
            burn_status, fuel_fraction, risk_level, actions
        )
        if contain is not None:
            return contain

        # Rule 5: do nothing
        return self._do_nothing(actions)

    # Rule helpers

    def _evacuation_action(
        self,
        burn_status: np.ndarray,
        asset_importance: np.ndarray,
        risk_level: np.ndarray,
        actions: List[BaseAction],
        g: int,
    ) -> Optional[BaseAction]:
        # Return an EvacuationAction for the highest-risk asset block, or
        # None. Find high-importance cells with risk above threshold first.
        asset_mask = asset_importance > 0.5
        if not np.any(asset_mask):
            return None
        risk_in_assets = risk_level * asset_mask
        if risk_in_assets.max() < self.evacuation_risk_threshold:
            return None

        # Best existing evacuation action (pick the one whose area
        # contains the cell with highest combined risk×importance)
        best_score = -1.0
        best_action = None
        for a in actions:
            if not isinstance(a, EvacuationAction):
                continue
            x1, y1, x2, y2 = a.area
            if x2 > x1 and y2 > y1:
                region_score = float(np.mean(risk_in_assets[x1:x2, y1:y2]))
            else:
                region_score = 0.0
            if region_score > best_score:
                best_score = region_score
                best_action = a
        return best_action  # may be None if no EvacuationActions in space

    def _water_drop_action(
        self,
        burn_status: np.ndarray,
        temperature: np.ndarray,
        actions: List[BaseAction],
    ) -> Optional[BaseAction]:
        # Return a WaterDropAction on the hottest burning cell, or None.
        burning = burn_status == 1
        if not np.any(burning):
            return None

        masked_temp = np.where(burning, temperature, -1.0)
        target = np.unravel_index(masked_temp.argmax(), masked_temp.shape)

        best_action = None
        best_dist = float("inf")
        for a in actions:
            if not isinstance(a, WaterDropAction):
                continue
            dist = np.sqrt(
                (a.location[0] - target[0]) ** 2
                + (a.location[1] - target[1]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_action = a
        return best_action

    def _containment_action(
        self,
        burn_status: np.ndarray,
        fuel_fraction: np.ndarray,
        risk_level: np.ndarray,
        actions: List[BaseAction],
    ) -> Optional[BaseAction]:
        # Return a FirebreakAction near the perimeter or BackburnAction
        # farther out.
        burning = burn_status == 1
        unburned = burn_status == 0

        if not np.any(burning) or not np.any(unburned):
            return None

        # Distance from each cell to the nearest burning cell. The EDT is
        # O(cells); the obvious pairwise version allocates
        # (grid x grid x n_burning) floats, which is ~60MB partway into a
        # 128^2 run and keeps growing as the fire does.
        dist_to_fire = distance_transform_edt(~burning)

        # Score = risk × fuel (protect high-value fuelled cells)
        score = risk_level * fuel_fraction * unburned.astype(float)

        near_perimeter = dist_to_fire <= self.perimeter_buffer
        near_score = score * near_perimeter
        far_score = score * ~near_perimeter

        # Try firebreak near perimeter first
        if near_score.max() > 0:
            target = np.unravel_index(near_score.argmax(), near_score.shape)
            action = self._closest_action_of_type(
                FirebreakAction, target, actions
            )
            if action is not None:
                return action

        # Else try backburn farther out
        if far_score.max() > 0:
            target = np.unravel_index(far_score.argmax(), far_score.shape)
            action = self._closest_action_of_type(
                BackburnAction, target, actions
            )
            if action is not None:
                return action

        return None

    @staticmethod
    def _closest_action_of_type(
        action_type,
        target,
        actions: List[BaseAction],
    ) -> Optional[BaseAction]:
        best_action = None
        best_dist = float("inf")
        for a in actions:
            if not isinstance(a, action_type):
                continue
            dist = np.sqrt(
                (a.location[0] - target[0]) ** 2
                + (a.location[1] - target[1]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_action = a
        return best_action

    @staticmethod
    def _do_nothing(actions: List[BaseAction]) -> BaseAction:
        for a in actions:
            if isinstance(a, DoNothingAction):
                return a
        return actions[0]
