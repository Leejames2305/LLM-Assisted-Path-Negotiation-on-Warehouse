"""
Layout Validation System
Validates layouts before loading/saving to ensure game compatibility
"""

from typing import Dict, List, Tuple, Set, Optional, Any, cast
from collections import deque

from .constants import (
    VALIDATION_ERRORS,
    VALIDATION_SUCCESS,
    VALID_CELL_CHARACTERS,
    MIN_WIDTH,
    MAX_WIDTH,
    MIN_HEIGHT,
    MAX_HEIGHT,
    MIN_AGENTS,
    MAX_AGENTS,
    MIN_BOXES,
    MAX_BOXES,
    MIN_TARGETS,
    MAX_TARGETS,
    LAYOUT_SCHEMA_VERSION,
    ValidationError,
)


class LayoutValidator:
    """Validates warehouse layouts for correctness and playability"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.is_valid = False

    def validate(self, layout: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete layout dictionary
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Phase 1: Basic structure validation
        self._validate_structure(layout)
        if self.errors:
            self.is_valid = False
            return False, self.errors, self.warnings

        # Phase 2: Dimension validation
        dimensions = layout.get('dimensions', {})
        width = dimensions.get('width')
        height = dimensions.get('height')
        self._validate_dimensions(width, height)
        if self.errors:
            self.is_valid = False
            return False, self.errors, self.warnings

        # Phase 3: Grid validation
        grid = layout.get('grid', [])
        self._validate_grid(grid, width, height)
        if self.errors:
            self.is_valid = False
            return False, self.errors, self.warnings

        # Phase 4: Entity validation (agents, boxes, targets)
        agents = layout.get('agents', [])
        boxes = layout.get('boxes', [])
        targets = layout.get('targets', [])

        self._validate_entities(agents, boxes, targets, grid, width, height)
        if self.errors:
            self.is_valid = False
            return False, self.errors, self.warnings

        # Phase 5: Goal validation
        agent_goals = layout.get('agent_goals', {})
        self._validate_goals(agents, boxes, targets, agent_goals)
        if self.errors:
            self.is_valid = False
            return False, self.errors, self.warnings

        # Phase 6: Reachability validation (comprehensive pathfinding check)
        self._validate_reachability(agents, boxes, targets, agent_goals, grid, width, height)
        if self.errors:
            self.is_valid = False
            return False, self.errors, self.warnings

        self.is_valid = True
        return True, self.errors, self.warnings

    # ==================== VALIDATION PHASES ====================

    def _validate_structure(self, layout: Dict):
        """Validate basic layout structure"""
        required_fields = ['version', 'name', 'dimensions', 'grid', 'agents', 'boxes', 'targets', 'agent_goals']

        for field in required_fields:
            if field not in layout:
                error = VALIDATION_ERRORS['MISSING_FIELD'].format(field=field)
                self.errors.append(error)

        # Check version
        if 'version' in layout and layout['version'] != LAYOUT_SCHEMA_VERSION:
            error = VALIDATION_ERRORS['VERSION_MISMATCH'].format(
                version=layout['version'],
                current=LAYOUT_SCHEMA_VERSION
            )
            self.errors.append(error)

    def _validate_dimensions(self, width: Optional[int], height: Optional[int]):
        """Validate width and height"""
        if not isinstance(width, int):
            self.errors.append(VALIDATION_ERRORS['INVALID_WIDTH'].format(min=MIN_WIDTH, max=MAX_WIDTH))
            return

        if not isinstance(height, int):
            self.errors.append(VALIDATION_ERRORS['INVALID_HEIGHT'].format(min=MIN_HEIGHT, max=MAX_HEIGHT))
            return

        if not (MIN_WIDTH <= width <= MAX_WIDTH):
            error = VALIDATION_ERRORS['INVALID_WIDTH'].format(min=MIN_WIDTH, max=MAX_WIDTH)
            self.errors.append(error)

        if not (MIN_HEIGHT <= height <= MAX_HEIGHT):
            error = VALIDATION_ERRORS['INVALID_HEIGHT'].format(min=MIN_HEIGHT, max=MAX_HEIGHT)
            self.errors.append(error)

    def _validate_grid(self, grid: List[str], width: int, height: int):
        """Validate grid structure and content"""
        if not isinstance(grid, list):
            self.errors.append(VALIDATION_ERRORS['INVALID_GRID_FORMAT'])
            return

        if len(grid) != height:
            error = VALIDATION_ERRORS['DIMENSION_MISMATCH'].format(width=width, height=height)
            self.errors.append(error)
            return

        for y, row in enumerate(grid):
            if not isinstance(row, str):
                error = VALIDATION_ERRORS['INVALID_GRID_FORMAT']
                self.errors.append(error)
                return

            if len(row) != width:
                error = VALIDATION_ERRORS['DIMENSION_MISMATCH'].format(width=width, height=height)
                self.errors.append(error)
                return

            for x, cell in enumerate(row):
                if cell not in VALID_CELL_CHARACTERS:
                    error = VALIDATION_ERRORS['INVALID_CELL_CHARACTER'].format(
                        x=x, y=y, char=cell, valid=VALID_CELL_CHARACTERS
                    )
                    self.errors.append(error)

    def _validate_entities(
        self,
        agents: List[Dict],
        boxes: List[Dict],
        targets: List[Dict],
        grid: List[str],
        width: int,
        height: int
    ):
        """Validate agents, boxes, and targets"""
        # Check counts
        if len(agents) < MIN_AGENTS:
            error = VALIDATION_ERRORS['NO_AGENTS'].format(min=MIN_AGENTS)
            self.errors.append(error)

        if len(agents) > MAX_AGENTS:
            error = VALIDATION_ERRORS['TOO_MANY_AGENTS'].format(max=MAX_AGENTS)
            self.errors.append(error)

        if len(boxes) < MIN_BOXES:
            error = VALIDATION_ERRORS['NO_BOXES'].format(min=MIN_BOXES)
            self.errors.append(error)

        if len(boxes) > MAX_BOXES:
            error = VALIDATION_ERRORS['TOO_MANY_BOXES'].format(max=MAX_BOXES)
            self.errors.append(error)

        if len(targets) < MIN_TARGETS:
            error = VALIDATION_ERRORS['NO_TARGETS'].format(min=MIN_TARGETS)
            self.errors.append(error)

        if len(targets) > MAX_TARGETS:
            error = VALIDATION_ERRORS['TOO_MANY_TARGETS'].format(max=MAX_TARGETS)
            self.errors.append(error)

        # Validate agents
        agent_ids = set()
        agent_positions: Dict[Tuple[int, int], List[Any]] = {}
        for agent in agents:
            agent_id: Any = agent.get('id')
            x: Any = agent.get('x')
            y: Any = agent.get('y')

            # Check ID uniqueness
            if agent_id in agent_ids:
                error = VALIDATION_ERRORS['DUPLICATE_AGENT_ID'].format(id=agent_id)
                self.errors.append(error)
            agent_ids.add(agent_id)

            # Check bounds
            if not isinstance(x, int) or not isinstance(y, int) or not (0 <= x < width and 0 <= y < height):
                error = VALIDATION_ERRORS['AGENT_OUT_OF_BOUNDS'].format(id=agent_id, x=x, y=y)
                self.errors.append(error)
                continue

            # Check not on wall
            if grid[y][x] == '#':
                error = VALIDATION_ERRORS['AGENT_ON_WALL'].format(id=agent_id, x=x, y=y)
                self.errors.append(error)
                continue

            # Track position
            if (x, y) not in agent_positions:
                agent_positions[(x, y)] = []
            agent_positions[(x, y)].append(agent_id)

        # Check overlaps
        for (x, y), agent_list in agent_positions.items():
            if len(agent_list) > 1:
                error = VALIDATION_ERRORS['AGENT_OVERLAP'].format(x=x, y=y, ids=agent_list)
                self.errors.append(error)

        # Validate boxes
        box_ids = set()
        box_positions: Dict[Any, Tuple[int, int]] = {}
        for box in boxes:
            box_id: Any = box.get('id')
            x: Any = box.get('x')
            y: Any = box.get('y')

            # Check ID uniqueness
            if box_id in box_ids:
                error = VALIDATION_ERRORS['DUPLICATE_BOX_ID'].format(id=box_id)
                self.errors.append(error)
            box_ids.add(box_id)

            # Check bounds
            if not isinstance(x, int) or not isinstance(y, int) or not (0 <= x < width and 0 <= y < height):
                error = VALIDATION_ERRORS['BOX_OUT_OF_BOUNDS'].format(id=box_id, x=x, y=y)
                self.errors.append(error)
                continue

            # Check not on wall
            if grid[y][x] == '#':
                error = VALIDATION_ERRORS['BOX_ON_WALL'].format(id=box_id, x=x, y=y)
                self.errors.append(error)
                continue

            # Check not on agent
            if (x, y) in agent_positions:
                error = VALIDATION_ERRORS['BOX_ON_AGENT'].format(id=box_id, x=x, y=y)
                self.errors.append(error)
                continue

            box_positions[box_id] = (x, y)

        # Validate targets
        target_ids = set()
        target_positions: Dict[Any, Tuple[int, int]] = {}
        for target in targets:
            target_id: Any = target.get('id')
            x: Any = target.get('x')
            y: Any = target.get('y')

            # Check ID uniqueness
            if target_id in target_ids:
                error = VALIDATION_ERRORS['DUPLICATE_TARGET_ID'].format(id=target_id)
                self.errors.append(error)
            target_ids.add(target_id)

            # Check bounds
            if not isinstance(x, int) or not isinstance(y, int) or not (0 <= x < width and 0 <= y < height):
                error = VALIDATION_ERRORS['TARGET_OUT_OF_BOUNDS'].format(id=target_id, x=x, y=y)
                self.errors.append(error)
                continue

            # Check not on wall
            if grid[y][x] == '#':
                error = VALIDATION_ERRORS['TARGET_ON_WALL'].format(id=target_id, x=x, y=y)
                self.errors.append(error)
                continue

            # Check not on agent
            if (x, y) in agent_positions:
                error = VALIDATION_ERRORS['TARGET_ON_AGENT'].format(id=target_id, x=x, y=y)
                self.errors.append(error)
                continue

            target_positions[target_id] = (x, y)

    def _validate_goals(
        self,
        agents: List[Dict],
        boxes: List[Dict],
        targets: List[Dict],
        agent_goals: Dict[str, int]
    ):
        """Validate that all goals are valid and complete"""
        box_ids = {box['id'] for box in boxes}
        target_ids = {target['id'] for target in targets}
        agent_ids = {agent['id'] for agent in agents}

        # Track which boxes and targets are used
        used_boxes = set()
        used_targets = set()

        for agent in agents:
            agent_id = agent['id']
            goal_id = agent_goals.get(str(agent_id))

            # Check agent has a goal
            if goal_id is None:
                error = VALIDATION_ERRORS['AGENT_WITHOUT_GOAL'].format(agent_id=agent_id)
                self.errors.append(error)
                continue

            # Check goal references valid target
            if goal_id not in target_ids:
                error = VALIDATION_ERRORS['INVALID_AGENT_GOAL'].format(
                    agent_id=agent_id,
                    target_id=goal_id
                )
                self.errors.append(error)
                continue

            used_targets.add(goal_id)

            # Implicitly: goal_id also implies a box with same ID should exist
            if goal_id not in box_ids:
                error = VALIDATION_ERRORS['INVALID_AGENT_GOAL'].format(
                    agent_id=agent_id,
                    target_id=goal_id
                )
                self.errors.append(error)
                continue

            used_boxes.add(goal_id)

        # Warn about unused boxes (if any)
        for box_id in box_ids:
            if box_id not in used_boxes:
                warning = VALIDATION_ERRORS['UNUSED_BOX'].format(box_id=box_id)
                self.warnings.append(warning)

        # Warn about unused targets (if any)
        for target_id in target_ids:
            if target_id not in used_targets:
                warning = VALIDATION_ERRORS['UNUSED_TARGET'].format(target_id=target_id)
                self.warnings.append(warning)

    def _validate_reachability(
        self,
        agents: List[Dict],
        boxes: List[Dict],
        targets: List[Dict],
        agent_goals: Dict[str, int],
        grid: List[str],
        width: int,
        height: int
    ):
        """Validate that all agents can reach their targets (reachability check)"""
        # Build wall positions
        walls = set()
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == '#':
                    walls.add((x, y))

        # Build entity maps
        boxes_dict = {box['id']: (box['x'], box['y']) for box in boxes}
        targets_dict = {target['id']: (target['x'], target['y']) for target in targets}

        # Check reachability for each agent
        for agent in agents:
            agent_id = agent['id']
            agent_pos = (agent['x'], agent['y'])
            goal_id = agent_goals.get(str(agent_id))

            if goal_id is None:
                continue

            # Check if agent can reach box first
            box_pos = boxes_dict.get(goal_id)
            if box_pos:
                if not self._can_reach(agent_pos, box_pos, walls, width, height):
                    error = VALIDATION_ERRORS['UNREACHABLE_BOX'].format(
                        agent_id=agent_id,
                        box_id=goal_id,
                        box_x=box_pos[0],
                        box_y=box_pos[1]
                    )
                    self.errors.append(error)

            # Check if agent can reach target
            target_pos = targets_dict.get(goal_id)
            if target_pos:
                if not self._can_reach(agent_pos, target_pos, walls, width, height):
                    error = VALIDATION_ERRORS['UNREACHABLE_TARGET'].format(
                        agent_id=agent_id,
                        target_id=goal_id,
                        target_x=target_pos[0],
                        target_y=target_pos[1]
                    )
                    self.errors.append(error)

    def _can_reach(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        width: int,
        height: int
    ) -> bool:
        """BFS to check if goal is reachable from start"""
        if start == goal:
            return True

        visited = {start}
        queue = deque([start])

        while queue:
            x, y = queue.popleft()

            # Check all 4 directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if (nx, ny) == goal:
                    return True

                if (0 <= nx < width and 0 <= ny < height and
                    (nx, ny) not in walls and
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return False

    def get_error_summary(self) -> str:
        """Get a formatted summary of all errors"""
        if not self.errors:
            return VALIDATION_SUCCESS['VALID_LAYOUT']

        summary = f"❌ Validation failed with {len(self.errors)} error(s):\n"
        for i, error in enumerate(self.errors, 1):
            summary += f"  {i}. {error}\n"

        if self.warnings:
            summary += f"\n⚠️  {len(self.warnings)} warning(s):\n"
            for i, warning in enumerate(self.warnings, 1):
                summary += f"  {i}. {warning}\n"

        return summary
