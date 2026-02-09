"""
Benchmark Tool for LLM-Assisted Multi-Robot Navigation System

Runs configurable benchmark tests on warehouse layouts with:
- Random agent/box/target position generation (seeded for reproducibility)
- Multiple simulation rounds with per-round time limits
- Comprehensive performance metrics collection
- CSV and JSON output for analysis
"""

import os
import sys
import csv
import json
import copy
import random
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simulation.game_engine import GameEngine
from src.map_generator import WarehouseMap
from src.map_generator.layout_manager import LayoutManager
from src.map_generator.layout_selector import select_layout_interactive
from src.agents import RobotAgent
from src.logging import UnifiedLogger

# Initialize colorama and load environment
init(autoreset=True)
load_dotenv()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    num_agents: int
    num_rounds: int
    time_limit_seconds: int
    seed: int
    spatial_hints_enabled: bool
    
    @classmethod
    def from_env(cls) -> 'BenchmarkConfig':
        """Load configuration from environment variables"""
        return cls(
            num_agents=int(os.getenv('BENCHMARK_NUM_AGENTS', '2')),
            num_rounds=int(os.getenv('BENCHMARK_NUM_ROUNDS', '5')),
            time_limit_seconds=int(os.getenv('BENCHMARK_TIME_LIMIT_SECONDS', '300')),
            seed=int(os.getenv('BENCHMARK_SEED', '42')),
            spatial_hints_enabled=os.getenv('BENCHMARK_SPATIAL_HINTS_ENABLED', 'true').lower() == 'true'
        )


@dataclass
class RoundResult:
    """Results from a single benchmark round"""
    round_num: int
    status: str  # 'success', 'timeout', 'failed'
    cooperative_success_rate: float
    makespan_seconds: float
    collision_rate: float
    path_efficiency: float
    total_tokens_used: int
    avg_conflict_resolution_time_ms: float
    total_turns: int
    total_negotiations: int
    total_collisions: int


def load_benchmark_config() -> BenchmarkConfig:
    """Load and display benchmark configuration from .env"""
    config = BenchmarkConfig.from_env()
    
    print(f"\n{Fore.CYAN}📊 Benchmark Configuration:{Style.RESET_ALL}")
    print(f"   Number of Agents: {config.num_agents}")
    print(f"   Number of Rounds: {config.num_rounds}")
    print(f"   Time Limit/Round: {config.time_limit_seconds}s")
    print(f"   Random Seed: {config.seed}")
    print(f"   Spatial Hints: {'Enabled' if config.spatial_hints_enabled else 'Disabled'}")
    
    return config


def get_walkable_cells(grid: List[str]) -> List[Tuple[int, int]]:
    """
    Extract all walkable cell positions from the grid.
    
    Args:
        grid: List of strings representing the warehouse grid
        
    Returns:
        List of (x, y) positions that are walkable (not walls)
    """
    walkable = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell in '.@ABT':  # Walkable cells (floor, agent, box, target markers)
                walkable.append((x, y))
    return walkable


def bfs_reachable(grid: List[str], start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """
    Check if goal is reachable from start using BFS pathfinding.
    
    Args:
        grid: List of strings representing the warehouse grid
        start: Starting position (x, y)
        goal: Target position (x, y)
        
    Returns:
        True if goal is reachable from start
    """
    if start == goal:
        return True
    
    height = len(grid)
    width = len(grid[0]) if grid else 0
    
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
    
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in visited and grid[ny][nx] != '#':
                    if (nx, ny) == goal:
                        return True
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    return False


def get_adjacent_walkable_cells(grid: List[str], pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get all walkable cells adjacent to the given position (including the position itself).
    
    Args:
        grid: List of strings representing the warehouse grid
        pos: Position (x, y) to check adjacents for
        
    Returns:
        List of adjacent walkable positions including the position itself
    """
    x, y = pos
    height = len(grid)
    width = len(grid[0]) if grid else 0
    
    adjacents = []
    
    # Include the position itself and all 4 cardinal directions
    directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] != '#':
            adjacents.append((nx, ny))
    
    return adjacents


def bfs_all_reachable(grid: List[str], start: Tuple[int, int]) -> set:
    """
    Get all cells reachable from start using BFS.
    
    Args:
        grid: List of strings representing the warehouse grid
        start: Starting position (x, y)
        
    Returns:
        Set of (x, y) positions reachable from start
    """
    height = len(grid)
    width = len(grid[0]) if grid else 0
    
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in visited and grid[ny][nx] != '#':
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    return visited


def generate_random_grid(width: int, height: int, wall_count: int, seed: int) -> Optional[List[str]]:
    """
    Generate a random grid with walls ensuring connectivity.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        wall_count: Number of internal walls to place
        seed: Random seed for reproducibility
        
    Returns:
        Grid as list of strings, or None if generation fails
    """
    rng = random.Random(seed)
    
    # Initialize grid with borders as walls and interior as floor
    grid = []
    for y in range(height):
        if y == 0 or y == height - 1:
            grid.append('#' * width)
        else:
            grid.append('#' + '.' * (width - 2) + '#')
    
    # Calculate available internal cells
    internal_cells = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            internal_cells.append((x, y))
    
    max_walls = len(internal_cells)
    
    # Clamp wall_count to available space
    if wall_count > max_walls:
        wall_count = max_walls
    
    # Try to place walls while maintaining connectivity
    # Maximum attempts = wall_count * 10 to allow retries when placement fails
    max_attempts = wall_count * 10
    walls_placed = 0
    
    for attempt in range(max_attempts):
        if walls_placed >= wall_count:
            break
        
        # Pick random internal cell
        if not internal_cells:
            break
        
        wall_pos = rng.choice(internal_cells)
        x, y = wall_pos
        
        # Convert grid to mutable
        grid_list = [list(row) for row in grid]
        grid_list[y][x] = '#'
        test_grid = [''.join(row) for row in grid_list]
        
        # Check if all walkable cells are still connected
        walkable = get_walkable_cells(test_grid)
        
        if len(walkable) < 2:
            # Not enough walkable cells
            continue
        
        # Pick any walkable cell as start and check if all others are reachable
        start = walkable[0]
        reachable = bfs_all_reachable(test_grid, start)
        
        if len(reachable) == len(walkable):
            # All walkable cells are connected, accept this wall
            grid = test_grid
            internal_cells.remove(wall_pos)
            walls_placed += 1
    
    # Ensure we have sufficient walkable cells for entities
    # Minimum 4: at least 1 agent, 1 box, 1 target, plus 1 extra for movement
    walkable = get_walkable_cells(grid)
    if len(walkable) < 4:
        return None
    
    return grid


def generate_random_positions(
    layout: Dict,
    num_agents: int,
    seed: int,
    round_num: int
) -> Optional[Dict[str, List[Dict]]]:
    """
    Generate random valid positions for agents, boxes, and targets.
    
    Uses seeded RNG for reproducibility. Ensures:
    - Agents spawn at the same position or adjacent to their boxes (easy pickup)
    - Targets are randomly distributed but reachable
    - No position overlaps
    - All positions are on walkable cells
    
    Args:
        layout: Base layout dictionary
        num_agents: Number of agents/boxes/targets to generate
        seed: Base random seed
        round_num: Round number (used to vary positions each round)
        
    Returns:
        Dict with 'agents', 'boxes', 'targets' lists, or None if generation fails
    """
    grid = layout['grid']
    walkable = get_walkable_cells(grid)
    
    # Need at least 2 positions per agent (agent+box can share, target separate)
    # Being conservative: need space for num_agents * 2 to account for some overlap
    required_positions = num_agents * 2
    
    if len(walkable) < required_positions:
        print(f"{Fore.RED}❌ Not enough walkable cells ({len(walkable)}) for {num_agents} agents{Style.RESET_ALL}")
        return None
    
    # Create seeded RNG for this round
    rng = random.Random(seed + round_num)
    
    # Shuffle walkable cells
    available = walkable.copy()
    rng.shuffle(available)
    
    agents = []
    boxes = []
    targets = []
    used_positions = set()
    
    max_attempts = 100  # Maximum attempts to find valid configuration
    
    for agent_id in range(num_agents):
        found_valid = False
        
        for attempt in range(max_attempts):
            # Re-shuffle if we've exhausted current order
            if len(available) < 2:
                available = [p for p in walkable if p not in used_positions]
                if not available:
                    break
                rng.shuffle(available)
            
            if len(available) < 2:
                break
            
            # Pick agent/box starting position
            agent_box_start = available.pop(0)
            
            # Find adjacent positions for box (including same position)
            box_candidates = get_adjacent_walkable_cells(grid, agent_box_start)
            box_candidates = [p for p in box_candidates if p not in used_positions]
            
            if not box_candidates:
                continue
            
            # Randomly pick box position from adjacents
            box_pos = rng.choice(box_candidates)
            
            # Agent spawns at the starting position
            agent_pos = agent_box_start
            
            # Now find a target position that's reachable from box
            target_pos = None
            temp_available = [p for p in available if p not in used_positions and p != box_pos and p != agent_pos]
            
            if not temp_available:
                # Try from all walkable positions
                temp_available = [p for p in walkable if p not in used_positions and p != box_pos and p != agent_pos]
            
            if not temp_available:
                continue
            
            # Shuffle and try to find reachable target
            rng.shuffle(temp_available)
            for candidate_target in temp_available:
                if bfs_reachable(grid, box_pos, candidate_target):
                    target_pos = candidate_target
                    break
            
            if target_pos is None:
                continue
            
            # Valid configuration found
            agents.append({'id': agent_id, 'x': agent_pos[0], 'y': agent_pos[1]})
            boxes.append({'id': agent_id, 'x': box_pos[0], 'y': box_pos[1]})
            targets.append({'id': agent_id, 'x': target_pos[0], 'y': target_pos[1]})
            
            used_positions.add(agent_pos)
            used_positions.add(box_pos)
            used_positions.add(target_pos)
            
            # Remove used positions from available
            available = [p for p in available if p not in used_positions]
            
            found_valid = True
            break
        
        if not found_valid:
            print(f"{Fore.RED}❌ Failed to find valid position for agent {agent_id}{Style.RESET_ALL}")
            return None
    
    return {
        'agents': agents,
        'boxes': boxes,
        'targets': targets
    }


def create_benchmark_layout(base_layout: Dict, positions: Dict, num_agents: int) -> Dict:
    """
    Create a new layout with randomized entity positions.
    
    Args:
        base_layout: Original layout to clone
        positions: Dict with 'agents', 'boxes', 'targets' from generate_random_positions
        num_agents: Number of agents
        
    Returns:
        New layout dict with updated positions
    """
    # Deep copy the base layout
    new_layout = copy.deepcopy(base_layout)
    
    # Update entities
    new_layout['agents'] = positions['agents']
    new_layout['boxes'] = positions['boxes']
    new_layout['targets'] = positions['targets']
    
    # Update agent goals (each agent assigned to same-id box/target)
    new_layout['agent_goals'] = {str(i): i for i in range(num_agents)}
    
    # Update grid to reflect new positions (for visualization)
    grid = [list(row) for row in new_layout['grid']]
    
    # Clear old entity markers
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell in 'ABT@':
                grid[y][x] = '.'
    
    # Place new entities
    for agent in positions['agents']:
        grid[agent['y']][agent['x']] = 'A'
    for box in positions['boxes']:
        grid[box['y']][box['x']] = 'B'
    for target in positions['targets']:
        grid[target['y']][target['x']] = 'T'
    
    new_layout['grid'] = [''.join(row) for row in grid]
    
    return new_layout


def run_single_round(
    layout: Dict,
    round_num: int,
    config: BenchmarkConfig,
    output_dir: str
) -> RoundResult:
    """
    Execute a single benchmark round.
    
    Args:
        layout: Layout configuration for this round
        round_num: Current round number (1-indexed)
        config: Benchmark configuration
        output_dir: Directory for saving logs
        
    Returns:
        RoundResult with metrics and status
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"📍 ROUND {round_num}/{config.num_rounds}")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    status = 'success'
    
    try:
        # Create game engine
        layout_dims = layout['dimensions']
        num_agents = len(layout['agents'])
        
        game_engine = GameEngine(
            width=layout_dims['width'],
            height=layout_dims['height'],
            num_agents=num_agents
        )
        
        # Configure for benchmark mode
        game_engine.timeout_seconds = config.time_limit_seconds
        game_engine.silent_mode = False  # Keep output visible for now
        
        # Configure spatial hints
        game_engine.central_negotiator.set_spatial_hints(config.spatial_hints_enabled)
        
        # Reset token counter for fresh metrics
        game_engine.reset_token_usage()
        
        # Load the layout
        game_engine.warehouse_map = WarehouseMap.from_layout(layout)
        
        # Initialize agents
        game_engine.agents = {}
        for agent in layout['agents']:
            agent_id = agent['id']
            position = (agent['x'], agent['y'])
            robot = RobotAgent(agent_id, position)
            game_engine.agents[agent_id] = robot
        
        # Display round info
        print(f"   Layout: {layout.get('name', 'Benchmark Layout')}")
        print(f"   Dimensions: {layout_dims['width']}x{layout_dims['height']}")
        print(f"   Agents: {num_agents}")
        
        # Initialize and run simulation
        game_engine.initialize_simulation()
        
        start_time = time.time()
        
        # Run simulation loop
        while game_engine.run_simulation_step():
            # Check for stop request
            if game_engine.stop_requested:
                status = 'timeout'
                break
        
        elapsed = time.time() - start_time
        
        # Determine status
        if game_engine.stop_requested or (config.time_limit_seconds > 0 and elapsed >= config.time_limit_seconds):
            status = 'timeout'
            print(f"\n{Fore.YELLOW}⏱️  Round {round_num} TIMEOUT after {elapsed:.1f}s{Style.RESET_ALL}")
        elif game_engine.simulation_complete:
            status = 'success'
            print(f"\n{Fore.GREEN}✅ Round {round_num} COMPLETED in {elapsed:.1f}s{Style.RESET_ALL}")
        else:
            status = 'failed'
            print(f"\n{Fore.RED}❌ Round {round_num} FAILED after {elapsed:.1f}s{Style.RESET_ALL}")
        
        # Calculate metrics
        metrics = game_engine.calculate_performance_metrics()
        
        # Override CSR to 0 if timeout/failed
        if status != 'success':
            metrics['cooperative_success_rate'] = 0.0
        
        # Save simulation log
        log_filename = f"sim_log_round_{round_num}.json"
        game_engine.save_simulation_log_to_path(output_dir, log_filename)
        
        return RoundResult(
            round_num=round_num,
            status=status,
            cooperative_success_rate=metrics['cooperative_success_rate'],
            makespan_seconds=metrics['makespan_seconds'],
            collision_rate=metrics['collision_rate'],
            path_efficiency=metrics['path_efficiency'],
            total_tokens_used=metrics['total_tokens_used'],
            avg_conflict_resolution_time_ms=metrics['avg_conflict_resolution_time_ms'],
            total_turns=metrics['total_turns'],
            total_negotiations=metrics['total_negotiations'],
            total_collisions=metrics['total_collisions']
        )
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ Round {round_num} ERROR: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        
        return RoundResult(
            round_num=round_num,
            status='failed',
            cooperative_success_rate=0.0,
            makespan_seconds=0.0,
            collision_rate=0.0,
            path_efficiency=0.0,
            total_tokens_used=0,
            avg_conflict_resolution_time_ms=0.0,
            total_turns=0,
            total_negotiations=0,
            total_collisions=0
        )


def save_benchmark_csv(results: List[RoundResult], output_dir: str):
    """
    Save benchmark results to CSV file.
    
    Args:
        results: List of RoundResult objects
        output_dir: Directory to save CSV
    """
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    
    fieldnames = [
        'Round', 'Status', 'CSR', 'Makespan_s', 'CollisionRate',
        'PathEfficiency', 'TokenCost', 'AvgResolutionTime_ms',
        'TotalTurns', 'TotalNegotiations', 'TotalCollisions'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'Round': result.round_num,
                'Status': result.status,
                'CSR': result.cooperative_success_rate,
                'Makespan_s': result.makespan_seconds,
                'CollisionRate': result.collision_rate,
                'PathEfficiency': result.path_efficiency,
                'TokenCost': result.total_tokens_used,
                'AvgResolutionTime_ms': result.avg_conflict_resolution_time_ms,
                'TotalTurns': result.total_turns,
                'TotalNegotiations': result.total_negotiations,
                'TotalCollisions': result.total_collisions
            })
    
    print(f"📄 Results CSV saved: {csv_path}")


def save_benchmark_summary(
    results: List[RoundResult],
    config: BenchmarkConfig,
    layout_name: str,
    output_dir: str
):
    """
    Save benchmark summary JSON with aggregate statistics.
    
    Args:
        results: List of RoundResult objects
        config: Benchmark configuration
        layout_name: Name of the layout used
        output_dir: Directory to save summary
    """
    # Calculate aggregate statistics
    success_count = sum(1 for r in results if r.status == 'success')
    timeout_count = sum(1 for r in results if r.status == 'timeout')
    failed_count = sum(1 for r in results if r.status == 'failed')
    
    # Calculate averages (excluding failed rounds for some metrics)
    valid_results = [r for r in results if r.status != 'failed']
    
    def safe_avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    summary = {
        'benchmark_info': {
            'timestamp': datetime.now().isoformat(),
            'layout_name': layout_name,
            'config': asdict(config)
        },
        'overall_results': {
            'total_rounds': len(results),
            'successful_rounds': success_count,
            'timeout_rounds': timeout_count,
            'failed_rounds': failed_count,
            'success_rate': (success_count / len(results) * 100) if results else 0
        },
        'average_metrics': {
            'avg_csr': safe_avg([r.cooperative_success_rate for r in results]),
            'avg_makespan_seconds': safe_avg([r.makespan_seconds for r in valid_results]),
            'avg_collision_rate': safe_avg([r.collision_rate for r in valid_results]),
            'avg_path_efficiency': safe_avg([r.path_efficiency for r in valid_results]),
            'total_tokens_used': sum(r.total_tokens_used for r in results),
            'avg_tokens_per_round': safe_avg([r.total_tokens_used for r in results]),
            'avg_resolution_time_ms': safe_avg([r.avg_conflict_resolution_time_ms for r in valid_results]),
            'avg_turns_per_round': safe_avg([r.total_turns for r in valid_results]),
            'avg_negotiations_per_round': safe_avg([r.total_negotiations for r in valid_results])
        },
        'per_round_results': [asdict(r) for r in results]
    }
    
    summary_path = os.path.join(output_dir, 'benchmark_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Summary JSON saved: {summary_path}")
    
    return summary


def create_layout_from_grid(grid: List[str], name: str = "random_map") -> Dict:
    """
    Create a layout dictionary from a generated grid.
    
    Args:
        grid: List of strings representing the warehouse grid
        name: Name for the layout
        
    Returns:
        Layout dictionary (without agents, boxes, targets - those are added later)
    """
    height = len(grid)
    width = len(grid[0]) if grid else 0
    
    return {
        'version': 1,
        'name': name,
        'description': 'Randomly generated map',
        'dimensions': {
            'width': width,
            'height': height
        },
        'grid': grid,
        'agents': [],
        'boxes': [],
        'targets': [],
        'agent_goals': {}
    }


def prompt_for_random_map_generation(seed: int) -> Optional[Dict]:
    """
    Prompt user for random map generation parameters and generate the map.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Generated layout or None if cancelled
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"🎲 RANDOM MAP GENERATION")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    try:
        # Get map width
        width_input = input(f"\n{Fore.CYAN}Map width (8-50, default 20): {Style.RESET_ALL}").strip()
        width = int(width_input) if width_input else 20
        
        if width < 8 or width > 50:
            print(f"{Fore.RED}❌ Width must be between 8 and 50{Style.RESET_ALL}")
            return None
        
        # Get map height
        height_input = input(f"{Fore.CYAN}Map height (8-50, default 15): {Style.RESET_ALL}").strip()
        height = int(height_input) if height_input else 15
        
        if height < 8 or height > 50:
            print(f"{Fore.RED}❌ Height must be between 8 and 50{Style.RESET_ALL}")
            return None
        
        # Calculate max internal walls
        max_internal_cells = (width - 2) * (height - 2)
        
        # Get wall specification (percentage or count)
        print(f"\n{Fore.YELLOW}Specify walls as percentage (%) or absolute count:{Style.RESET_ALL}")
        wall_input = input(f"{Fore.CYAN}Walls (e.g., '20%' or '50', default 15%): {Style.RESET_ALL}").strip()
        
        if not wall_input:
            wall_input = "15%"
        
        # Parse wall input
        if wall_input.endswith('%'):
            # Percentage
            try:
                percentage = float(wall_input[:-1])
                if percentage < 0 or percentage > 70:
                    print(f"{Fore.RED}❌ Percentage must be between 0 and 70{Style.RESET_ALL}")
                    return None
                wall_count = int(max_internal_cells * percentage / 100)
            except ValueError:
                print(f"{Fore.RED}❌ Invalid percentage format{Style.RESET_ALL}")
                return None
        else:
            # Absolute count
            try:
                wall_count = int(wall_input)
                if wall_count < 0 or wall_count > max_internal_cells:
                    print(f"{Fore.RED}❌ Wall count must be between 0 and {max_internal_cells}{Style.RESET_ALL}")
                    return None
            except ValueError:
                print(f"{Fore.RED}❌ Invalid wall count format{Style.RESET_ALL}")
                return None
        
        # Generate the grid
        print(f"\n{Fore.YELLOW}🔨 Generating random map ({width}x{height} with ~{wall_count} walls)...{Style.RESET_ALL}")
        grid = generate_random_grid(width, height, wall_count, seed)
        
        if grid is None:
            print(f"{Fore.RED}❌ Failed to generate valid map{Style.RESET_ALL}")
            return None
        
        # Create layout
        layout = create_layout_from_grid(grid, f"random_{width}x{height}")
        
        # Display generated map
        print(f"\n{Fore.GREEN}✅ Map generated successfully!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Preview:{Style.RESET_ALL}")
        for row in grid:
            print(f"  {row}")
        
        walkable = get_walkable_cells(grid)
        print(f"\n{Fore.CYAN}Walkable cells: {len(walkable)}{Style.RESET_ALL}")
        
        return layout
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Cancelled.{Style.RESET_ALL}")
        return None
    except ValueError as e:
        print(f"{Fore.RED}❌ Invalid input: {e}{Style.RESET_ALL}")
        return None


def display_final_summary(summary: Dict):
    """Display final benchmark summary to console."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"📊 BENCHMARK SUMMARY")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    info = summary['benchmark_info']
    overall = summary['overall_results']
    metrics = summary['average_metrics']
    
    print(f"\n{Fore.WHITE}Layout: {info['layout_name']}{Style.RESET_ALL}")
    print(f"Agents: {info['config']['num_agents']} | Rounds: {overall['total_rounds']} | Seed: {info['config']['seed']}")
    
    print(f"\n{Fore.GREEN}Round Outcomes:{Style.RESET_ALL}")
    print(f"   ✅ Successful: {overall['successful_rounds']}")
    print(f"   ⏱️  Timeout: {overall['timeout_rounds']}")
    print(f"   ❌ Failed: {overall['failed_rounds']}")
    print(f"   📈 Success Rate: {overall['success_rate']:.1f}%")
    
    print(f"\n{Fore.YELLOW}Average Performance Metrics:{Style.RESET_ALL}")
    print(f"   CSR (Cooperative Success Rate): {metrics['avg_csr']:.2f}%")
    print(f"   Makespan: {metrics['avg_makespan_seconds']:.2f}s")
    print(f"   Collision Rate: {metrics['avg_collision_rate']:.3f}")
    print(f"   Path Efficiency: {metrics['avg_path_efficiency']:.2f}%")
    print(f"   Total Tokens Used: {metrics['total_tokens_used']}")
    print(f"   Avg Tokens/Round: {metrics['avg_tokens_per_round']:.0f}")
    print(f"   Avg Resolution Time: {metrics['avg_resolution_time_ms']:.2f}ms")
    print(f"   Avg Turns/Round: {metrics['avg_turns_per_round']:.1f}")
    print(f"   Avg Negotiations/Round: {metrics['avg_negotiations_per_round']:.1f}")
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


def run_benchmark(base_layout: Dict, config: BenchmarkConfig) -> List[RoundResult]:
    """
    Run the complete benchmark with all rounds.
    
    Args:
        base_layout: Base layout to use for benchmark
        config: Benchmark configuration
        
    Returns:
        List of RoundResult objects
    """
    layout_name = base_layout.get('name', 'unknown').replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = os.path.join(
        'logs', 'Benchmarks',
        f"{layout_name}_{config.num_agents}agents_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{Fore.GREEN}📁 Output directory: {output_dir}{Style.RESET_ALL}")
    
    results: List[RoundResult] = []
    
    for round_num in range(1, config.num_rounds + 1):
        # Generate random positions for this round
        print(f"\n🎲 Generating random positions for round {round_num}...")
        positions = generate_random_positions(
            base_layout,
            config.num_agents,
            config.seed,
            round_num
        )
        
        if positions is None:
            print(f"{Fore.RED}❌ Position generation failed. Stopping benchmark.{Style.RESET_ALL}")
            break
        
        # Create layout for this round
        round_layout = create_benchmark_layout(base_layout, positions, config.num_agents)
        round_layout['name'] = f"{layout_name}_round_{round_num}"
        
        # Run the round
        result = run_single_round(round_layout, round_num, config, output_dir)
        results.append(result)
        
        # Brief pause between rounds
        if round_num < config.num_rounds:
            print(f"\n⏳ Starting next round in 2 seconds...")
            time.sleep(2)
    
    # Save results
    if results:
        save_benchmark_csv(results, output_dir)
        summary = save_benchmark_summary(results, config, layout_name, output_dir)
        display_final_summary(summary)
    
    return results


def main():
    """Main entry point for benchmark tool."""
    print(f"{Fore.CYAN}{'='*60}")
    print(f"🔬 LLM Multi-Robot Navigation Benchmark Tool")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print(f"{Fore.RED}⚠️  WARNING: OpenRouter API key not configured!{Style.RESET_ALL}")
        print("Please set OPENROUTER_API_KEY in your .env file")
        return
    
    # Load configuration
    config = load_benchmark_config()
    
    # Validate configuration
    if config.num_agents < 1 or config.num_agents > 10:
        print(f"{Fore.RED}❌ Invalid number of agents ({config.num_agents}). Must be 1-10.{Style.RESET_ALL}")
        return
    
    if config.num_rounds < 1:
        print(f"{Fore.RED}❌ Invalid number of rounds ({config.num_rounds}). Must be >= 1.{Style.RESET_ALL}")
        return
    
    # Select layout or generate random map
    print(f"\n{Fore.CYAN}Choose layout mode:{Style.RESET_ALL}")
    print(f"  1. Select existing layout")
    print(f"  2. Generate random map")
    
    mode_choice = input(f"\n{Fore.CYAN}Select mode (1/2, default 1): {Style.RESET_ALL}").strip()
    
    if mode_choice == '2':
        # Generate random map
        base_layout = prompt_for_random_map_generation(config.seed)
        if base_layout is None:
            print(f"{Fore.RED}Random map generation cancelled. Exiting.{Style.RESET_ALL}")
            return
    else:
        # Select existing layout
        print(f"\n{Fore.CYAN}Select a layout for benchmarking:{Style.RESET_ALL}")
        base_layout = select_layout_interactive()
        
        if base_layout is None:
            print(f"{Fore.RED}No layout selected. Exiting.{Style.RESET_ALL}")
            return
    
    # Confirm benchmark start
    print(f"\n{Fore.YELLOW}Ready to start benchmark:{Style.RESET_ALL}")
    print(f"   Layout: {base_layout.get('name', 'Unknown')}")
    print(f"   Agents: {config.num_agents}")
    print(f"   Rounds: {config.num_rounds}")
    print(f"   Time Limit: {config.time_limit_seconds}s/round")
    print(f"   Seed: {config.seed}")
    
    confirm = input(f"\n{Fore.CYAN}Start benchmark? (Y/n): {Style.RESET_ALL}").strip().lower()
    if confirm == 'n':
        print("Benchmark cancelled.")
        return
    
    try:
        # Run benchmark
        print(f"\n{Fore.GREEN}🚀 Starting Benchmark...{Style.RESET_ALL}")
        results = run_benchmark(base_layout, config)
        
        if results:
            print(f"\n{Fore.GREEN}✅ Benchmark completed with {len(results)} rounds!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}⚠️  Benchmark ended with no completed rounds.{Style.RESET_ALL}")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Benchmark interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Benchmark error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{Fore.CYAN}Thank you for using the Benchmark Tool!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
