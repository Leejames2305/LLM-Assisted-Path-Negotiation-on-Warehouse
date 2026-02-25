"""
POGEMA GridConfig builder, map save/load, and coordinate utilities.
"""

import json
import os
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv
from pogema import GridConfig

load_dotenv()


# --- Coordinate conversion helpers ---

def xy_to_rc(x: int, y: int) -> Tuple[int, int]:
    """Convert legacy (x, y) coordinates to POGEMA (row, col)."""
    return (y, x)


def rc_to_xy(row: int, col: int) -> Tuple[int, int]:
    """Convert POGEMA (row, col) to legacy (x, y) coordinates."""
    return (col, row)


# --- Environment defaults from .env ---

def load_env_defaults() -> Dict:
    """Load POGEMA and OpenRouter configuration from environment variables."""
    return {
        # POGEMA settings
        'obs_radius': int(os.getenv('OBS_RADIUS', '5')),
        'max_episode_steps': int(os.getenv('MAX_EPISODE_STEPS', '256')),
        'seed': int(os.getenv('POGEMA_SEED', '42')),
        # OpenRouter settings
        'openrouter_api_key': os.getenv('OPENROUTER_API_KEY'),
        'central_llm_model': os.getenv('CENTRAL_LLM_MODEL', 'zai/glm-4.5-air:free'),
        'agent_llm_model': os.getenv('AGENT_LLM_MODEL', 'nvidia/nemotron-3-nano-30b-a3b:free'),
        'openrouter_provider_order': os.getenv('OPENROUTER_PROVIDER_ORDER', ''),
        'openrouter_reasoning_enabled': os.getenv('OPENROUTER_REASONING_ENABLED', 'false'),
        'openrouter_reasoning_exclude': os.getenv('OPENROUTER_REASONING_EXCLUDE', 'false'),
    }


# --- GridConfig builder ---

def create_grid_config(
    map_grid: List[str],
    agents_xy: List[Tuple[int, int]],
    targets_xy: List[Tuple[int, int]],
    obs_radius: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
) -> GridConfig:
    """
    Build a POGEMA GridConfig from a map definition and agent/target positions.

    map_grid: list of strings, '.' = free, '#' = obstacle (POGEMA format)
    agents_xy: list of (row, col) start positions
    targets_xy: list of (row, col) goal positions
    """
    defaults = load_env_defaults()

    obs_radius = obs_radius if obs_radius is not None else defaults['obs_radius']
    max_episode_steps = max_episode_steps if max_episode_steps is not None else defaults['max_episode_steps']
    seed = seed if seed is not None else defaults['seed']

    # POGEMA GridConfig expects the map as a single newline-joined string
    map_str = '\n'.join(map_grid) if isinstance(map_grid, list) else map_grid

    return GridConfig(
        map=map_str,
        agents_xy=list(agents_xy),
        targets_xy=list(targets_xy),
        obs_radius=obs_radius,
        max_episode_steps=max_episode_steps,
        seed=seed,
        **kwargs
    )


# --- Map JSON save/load ---

def save_grid_config(config_dict: Dict, filepath: str):
    """Save a map configuration dict as a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_grid_config(filepath: str) -> GridConfig:
    """Load a map config JSON file and return a GridConfig instance."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    return create_grid_config(
        map_grid=data['map'],
        agents_xy=[tuple(pos) for pos in data['agents_xy']],
        targets_xy=[tuple(pos) for pos in data['targets_xy']],
        obs_radius=data.get('obs_radius'),
        max_episode_steps=data.get('max_episode_steps'),
        seed=data.get('seed'),
    )


def extract_walls(grid_config: GridConfig) -> set:
    """Extract wall positions as a set of (row, col) tuples from a GridConfig."""
    walls = set()
    for row_idx, row in enumerate(grid_config.map):
        for col_idx, cell in enumerate(row):
            # POGEMA stores map as list of lists of ints: 0=free, 1=obstacle
            if cell == grid_config.OBSTACLE:
                walls.add((row_idx, col_idx))
    return walls
