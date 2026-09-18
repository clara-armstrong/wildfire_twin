"""Wildfire management actions.

Includes DoNothing, FirebreakAction, BackburnAction, WaterDropAction, EvacuationAction
"""
from typing import List, Tuple

import numpy as np
from pgmtwin.core.action import BaseAction
from pgmtwin.core.domain import DiscreteDomain

from .config import WildfireConfig
from .solver import asset_regions


class DoNothingAction(BaseAction):
    # Let the fire progress naturally.

    def __init__(self, state_domain: DiscreteDomain):
        super().__init__("do_nothing", state_domain)
        self.type = "do_nothing"

    def apply(self, physical_state):
        pass


class FirebreakAction(BaseAction):
    # Create a firebreak at a location (thesis Section 5.3, water-drop-derived).

    def __init__(self, location: Tuple[int, int], state_domain: DiscreteDomain):
        super().__init__(f"firebreak_{location[0]}_{location[1]}", state_domain)
        self.location = location
        self.type = "firebreak"
        self.temperature_reduction = 1.0
        self.fuel_fraction_removal = 0.5

    def apply(self, physical_state):
        i, j = self.location
        if not (0 <= i < physical_state["temperature"].shape[0] and
                0 <= j < physical_state["temperature"].shape[1]):
            return

        physical_state["temperature"][i, j] = max(
            0, physical_state["temperature"][i, j] - self.temperature_reduction
        )
        physical_state["fuel_fraction"][i, j] = max(
            0, physical_state["fuel_fraction"][i, j] * (1 - self.fuel_fraction_removal)
        )
        physical_state["ignition_threshold"][i, j] = min(
            4.0, physical_state["ignition_threshold"][i, j] + 0.5
        )
        if physical_state["fuel_fraction"][i, j] <= 0.05:
            physical_state["burn_status"][i, j] = 2


class BackburnAction(BaseAction):
    # Controlled burn ahead of the fire to consume fuel and create a fuel break.

    def __init__(self, location: Tuple[int, int], state_domain: DiscreteDomain):
        super().__init__(f"backburn_{location[0]}_{location[1]}", state_domain)
        self.location = location
        self.type = "backburn"
        self.ignition_temperature = 2.0

    def apply(self, physical_state):
        i, j = self.location
        if not (0 <= i < physical_state["temperature"].shape[0] and
                0 <= j < physical_state["temperature"].shape[1]):
            return
        if physical_state["burn_status"][i, j] != 0 or physical_state["fuel_fraction"][i, j] <= 0.1:
            return

        physical_state["temperature"][i, j] = max(
            physical_state["temperature"][i, j], self.ignition_temperature
        )
        physical_state["burn_status"][i, j] = 1
        physical_state["fuel_fraction"][i, j] = max(
            0, physical_state["fuel_fraction"][i, j] - 0.3
        )
        physical_state["ignition_threshold"][i, j] = min(
            4.0, physical_state["ignition_threshold"][i, j] + 0.2
        )


class WaterDropAction(BaseAction):
    # Water drop with a spatial falloff (thesis Appendix B.2).

    def __init__(self, location: Tuple[int, int], state_domain: DiscreteDomain, intensity: float = 1.0, radius: int = 2):
        super().__init__(f"water_drop_{location[0]}_{location[1]}", state_domain)
        self.location = location
        self.type = "water_drop"
        self.intensity = intensity
        self.radius = radius

    def apply(self, physical_state):
        i, j = self.location
        h, w = physical_state["temperature"].shape
        for di in range(-self.radius, self.radius + 1):
            for dj in range(-self.radius, self.radius + 1):
                ni, nj = i + di, j + dj
                if not (0 <= ni < h and 0 <= nj < w):
                    continue
                dist = np.sqrt(di**2 + dj**2)
                if dist > self.radius:
                    continue

                local_intensity = self.intensity * (1 - dist / self.radius)
                physical_state["temperature"][ni, nj] = max(
                    0, physical_state["temperature"][ni, nj] - 1.0 * local_intensity
                )
                physical_state["ignition_threshold"][ni, nj] = min(
                    4.0, physical_state["ignition_threshold"][ni, nj] + 0.3 * local_intensity
                )
                physical_state["fuel_fraction"][ni, nj] = max(
                    0, physical_state["fuel_fraction"][ni, nj] - 0.2 * local_intensity
                )
                if (physical_state["temperature"][ni, nj] < 0.5 and
                        physical_state["fuel_fraction"][ni, nj] < 0.1):
                    physical_state["burn_status"][ni, nj] = 2


class EvacuationAction(BaseAction):
    # Evacuate a rectangular area, reducing its asset importance (people are safe).

    def __init__(self, area: Tuple[int, int, int, int], state_domain: DiscreteDomain):
        super().__init__(f"evacuate_{area[0]}_{area[1]}_{area[2]}_{area[3]}", state_domain)
        self.area = area
        self.type = "evacuation"

    def apply(self, physical_state):
        x1, y1, x2, y2 = self.area
        h, w = physical_state["asset_importance"].shape
        x1, x2 = max(0, min(x1, h - 1)), max(0, min(x2, h - 1))
        y1, y2 = max(0, min(y1, w - 1)), max(0, min(y2, w - 1))

        physical_state["asset_importance"][x1:x2, y1:y2] *= 0.5
        if "evacuated" not in physical_state:
            physical_state["evacuated"] = np.zeros_like(
                physical_state["asset_importance"], dtype=bool
            )
        physical_state["evacuated"][x1:x2, y1:y2] = True


def build_default_actions(config: WildfireConfig, state_domain: DiscreteDomain) -> List[BaseAction]:
    # Build the default discrete action space: do-nothing, a firebreak grid,
    # a few backburn points, a few water-drop points, and one evacuation
    # action per asset region.
    #
    # Kept out of the environment's __init__ so the environment doesn't
    # need to know how the action space is populated.
    g = config.grid_size
    actions: List[BaseAction] = [DoNothingAction(state_domain)]

    # One evacuation zone per protected asset block. Without these,
    # RuleBasedPolicy's highest-priority rule (evacuate at-risk assets) can
    # never fire, because it has no EvacuationAction to select.
    for i0, i1, j0, j1 in asset_regions(config):
        actions.append(EvacuationAction((i0, j0, i1, j1), state_domain))

    step = max(1, g // 5)
    for i in range(step, g, step):
        for j in range(step, g, step):
            actions.append(FirebreakAction((i, j), state_domain))

    for i in [g // 4, g // 2, 3 * g // 4]:
        for j in [g // 4, g // 2, 3 * g // 4]:
            actions.append(BackburnAction((i, j), state_domain))

    for i in range(g // 4, g * 3 // 4, max(1, g // 4)):
        for j in range(g // 4, g * 3 // 4, max(1, g // 4)):
            actions.append(WaterDropAction((i, j), state_domain, intensity=1.0))

    return actions
