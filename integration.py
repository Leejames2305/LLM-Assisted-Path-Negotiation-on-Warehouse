"""
POGEMA Port — pogema-toolbox Algorithm Interface.

Wraps LLMNegotiationController to conform to the pogema-toolbox algorithm API.
Note: pogema-toolbox >= 0.2.0 is required but may not be installable on Python 3.12
due to a transitive dependency on `pathtools` which uses the removed `imp` module.
"""

from typing import List, Optional
from pogema import GridConfig

from agent_controller import LLMNegotiationController


class LLMNegotiationAlgorithm:
    """
    pogema-toolbox compatible algorithm interface for LLM-Assisted Path Negotiation.

    Implements:
    - act(observations) -> List[int]
    - reset(observations)
    """

    def __init__(
        self,
        enable_negotiation: bool = True,
        enable_spatial_hints: bool = True,
    ):
        self.enable_negotiation = enable_negotiation
        self.enable_spatial_hints = enable_spatial_hints
        self._controller: Optional[LLMNegotiationController] = None
        self._step = 0
        self._terminated: List[bool] = []

    def reset(self, observations, grid_config: Optional[GridConfig] = None):
        """
        Reset internal state for a new episode.
        Called by pogema-toolbox before each evaluation run.
        """
        self._step = 0

        if grid_config is not None:
            self._controller = LLMNegotiationController(
                grid_config=grid_config,
                enable_negotiation=self.enable_negotiation,
                enable_spatial_hints=self.enable_spatial_hints,
            )
            self._terminated = [False] * grid_config.num_agents
        elif observations is not None:
            # Can't fully initialize without grid_config; just clear queues
            if self._controller is not None:
                for q in self._controller.action_queues.values():
                    q.clear()
                self._controller.agent_done = [False] * self._controller.num_agents
            self._terminated = [False] * len(observations)

    def act(self, observations) -> List[int]:
        """
        Return one POGEMA action per agent for the current step.
        Called by pogema-toolbox on each env step.
        """
        if self._controller is None:
            return [0] * len(observations)

        actions = self._controller.get_actions(
            observations,
            self._terminated,
            self._step,
        )
        self._step += 1
        return actions

    def after_step(self, terminated: List[bool], env=None, actions: Optional[List[int]] = None):
        """
        Update internal state after a POGEMA step.
        Call this after env.step() with the returned terminated flags.
        """
        self._terminated = terminated
        if self._controller is not None:
            self._controller.update_positions(env, actions)

    def get_llm_stats(self) -> dict:
        """Expose LLM stats for benchmarking."""
        if self._controller is not None:
            return self._controller.get_llm_stats()
        return {}


# --- Registration helper (requires pogema-toolbox) ---

def register_with_toolbox():
    """
    Register LLMNegotiationAlgorithm with pogema-toolbox ToolboxRegistry.
    Requires pogema-toolbox >= 0.2.0.
    """
    try:
        from pogema_toolbox.registry import ToolboxRegistry  # type: ignore
        ToolboxRegistry.register_algorithm('LLMNegotiation', LLMNegotiationAlgorithm)
        print("✅ LLMNegotiation registered with pogema-toolbox ToolboxRegistry")
    except ImportError:
        print("⚠️  pogema-toolbox not installed — skipping algorithm registration")
    except Exception as e:
        print(f"⚠️  Failed to register with toolbox: {e}")
