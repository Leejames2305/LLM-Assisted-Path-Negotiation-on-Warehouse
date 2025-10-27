"""
Constants for Layout Management and Validation
"""

from enum import Enum
from typing import Dict, Any

# ==================== CELL TYPES ====================

class CellType(Enum):
    """Grid cell types for warehouse layout"""
    EMPTY = '.'
    WALL = '#'
    AGENT = 'A'
    AGENT_WITH_BOX = '@'
    BOX = 'B'
    TARGET = 'T'


# ==================== LAYOUT JSON SCHEMA ====================

LAYOUT_SCHEMA_VERSION = 1

LAYOUT_JSON_STRUCTURE = {
    "version": "int (currently 1)",
    "name": "str (layout name)",
    "description": "str (optional layout description)",
    "dimensions": {
        "width": "int (must be >= 5)",
        "height": "int (must be >= 5)"
    },
    "grid": ["array of strings representing rows", "each row must be a string of length = width"],
    "agents": [
        {
            "id": "int (unique, starting from 0)",
            "x": "int (0 <= x < width)",
            "y": "int (0 <= y < height)"
        }
    ],
    "boxes": [
        {
            "id": "int (unique, starting from 0)",
            "x": "int (0 <= x < width)",
            "y": "int (0 <= y < height)"
        }
    ],
    "targets": [
        {
            "id": "int (unique, starting from 0)",
            "x": "int (0 <= x < width)",
            "y": "int (0 <= y < height)"
        }
    ],
    "agent_goals": {
        "agent_id": "int (target_id to deliver to)"
    }
}

# ==================== VALIDATION RULES ====================

MIN_WIDTH = 5
MIN_HEIGHT = 5
MAX_WIDTH = 50
MAX_HEIGHT = 50

MIN_AGENTS = 1
MAX_AGENTS = 10

MIN_BOXES = 1
MAX_BOXES = 20

MIN_TARGETS = 1
MAX_TARGETS = 20

VALID_CELL_CHARACTERS = {'.', '#', 'A', '@', 'B', 'T'}

# ==================== VALIDATION ERROR MESSAGES ====================

class ValidationError(Exception):
    """Base exception for layout validation errors"""
    pass


VALIDATION_ERRORS: Dict[str, str] = {
    # Dimension errors
    'INVALID_WIDTH': 'Invalid width: must be between {min} and {max}',
    'INVALID_HEIGHT': 'Invalid height: must be between {min} and {max}',
    'DIMENSION_MISMATCH': 'Grid rows do not match specified dimensions (width={width}, height={height})',
    
    # Cell errors
    'INVALID_CELL_CHARACTER': 'Invalid character in grid at ({x}, {y}): "{char}". Valid: {valid}',
    'INVALID_GRID_FORMAT': 'Grid is not a list or rows have inconsistent lengths',
    
    # Agent errors
    'NO_AGENTS': 'Layout must have at least {min} agent(s)',
    'TOO_MANY_AGENTS': 'Layout cannot have more than {max} agents',
    'DUPLICATE_AGENT_ID': 'Duplicate agent ID: {id}',
    'AGENT_OUT_OF_BOUNDS': 'Agent {id} position ({x}, {y}) is out of bounds',
    'AGENT_ON_WALL': 'Agent {id} cannot be placed on wall at ({x}, {y})',
    'AGENT_OVERLAP': 'Multiple agents at position ({x}, {y}): agents {ids}',
    
    # Box errors
    'NO_BOXES': 'Layout must have at least {min} box(es)',
    'TOO_MANY_BOXES': 'Layout cannot have more than {max} boxes',
    'DUPLICATE_BOX_ID': 'Duplicate box ID: {id}',
    'BOX_OUT_OF_BOUNDS': 'Box {id} position ({x}, {y}) is out of bounds',
    'BOX_ON_WALL': 'Box {id} cannot be placed on wall at ({x}, {y})',
    'BOX_ON_AGENT': 'Box {id} overlaps with agent at ({x}, {y})',
    
    # Target errors
    'NO_TARGETS': 'Layout must have at least {min} target(s)',
    'TOO_MANY_TARGETS': 'Layout cannot have more than {max} targets',
    'DUPLICATE_TARGET_ID': 'Duplicate target ID: {id}',
    'TARGET_OUT_OF_BOUNDS': 'Target {id} position ({x}, {y}) is out of bounds',
    'TARGET_ON_WALL': 'Target {id} cannot be placed on wall at ({x}, {y})',
    'TARGET_ON_AGENT': 'Target {id} overlaps with agent at ({x}, {y})',
    
    # Goal errors
    'INVALID_AGENT_GOAL': 'Agent {agent_id} goal references non-existent target {target_id}',
    'AGENT_WITHOUT_GOAL': 'Agent {agent_id} has no goal assigned',
    'UNUSED_BOX': 'Box {box_id} is not assigned to any agent goal',
    'UNUSED_TARGET': 'Target {target_id} is not assigned to any agent goal',
    
    # Reachability errors
    'UNREACHABLE_BOX': 'Agent {agent_id} cannot reach assigned box {box_id} at ({box_x}, {box_y})',
    'UNREACHABLE_TARGET': 'Box/target goal unreachable: Agent {agent_id} cannot reach target {target_id} at ({target_x}, {target_y})',
    'ISOLATED_REGION': 'Some walkable cells are isolated - agents/targets in disconnected regions',
    
    # General errors
    'MISSING_FIELD': 'Missing required field: {field}',
    'INVALID_JSON': 'Invalid JSON format: {error}',
    'VERSION_MISMATCH': 'Layout version {version} not supported (current: {current})',
}

# ==================== SUCCESS MESSAGES ====================

VALIDATION_SUCCESS: Dict[str, str] = {
    'VALID_LAYOUT': '✅ Layout is valid and ready to use',
    'REACHABILITY_OK': '✅ All agents can reach their targets',
    'STRUCTURE_OK': '✅ Layout structure is valid',
    'GOALS_OK': '✅ All agent goals are valid',
}

# ==================== LAYOUT TEMPLATES ====================

EMPTY_LAYOUT_TEMPLATE: Dict[str, Any] = {
    "version": 1,
    "name": "Untitled Layout",
    "description": "",
    "dimensions": {
        "width": 8,
        "height": 6
    },
    "grid": [
        "########",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "########"
    ],
    "agents": [],
    "boxes": [],
    "targets": [],
    "agent_goals": {}
}

# ==================== PREBUILT LAYOUT NAMES ====================

PREBUILT_LAYOUTS = ['s_shaped', 'tunnel', 'bridge', 'empty']
CUSTOM_LAYOUT_DIR = 'layouts/custom'
PREBUILT_LAYOUT_DIR = 'layouts/prebuilt'
