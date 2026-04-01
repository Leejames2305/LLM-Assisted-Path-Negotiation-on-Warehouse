"""SIPP-style low-level planner hook for LNS2.

This module currently uses the existing time-constraint search engine while
exposing a dedicated low-level interface for future full SIPP expansion.
"""

from typing import List, Set, Tuple

from . import SimplePathfinder
from .path_table import PathTable

Location = Tuple[int, int]


class SIPPLowLevelSolver:
    """SIPP-compatible adapter over the current time-aware grid search."""

    def __init__(self, pathfinder: SimplePathfinder):
        self.pathfinder = pathfinder

    def find_path(
        self,
        start: Location,
        goal: Location,
        walls: Set[Location],
        path_table: PathTable,
        max_time_steps: int,
    ) -> List[Location]:
        # First pass with the requested horizon.
        path = self.pathfinder.find_path_with_time_constraints(
            start=start,
            goal=goal,
            walls=walls,
            max_time_steps=max_time_steps,
            path_table=path_table,
        )
        if path:
            return path

        # Second pass with longer horizon to tolerate longer waits.
        extended_horizon = max_time_steps * 2
        return self.pathfinder.find_path_with_time_constraints(
            start=start,
            goal=goal,
            walls=walls,
            max_time_steps=extended_horizon,
            path_table=path_table,
        )
