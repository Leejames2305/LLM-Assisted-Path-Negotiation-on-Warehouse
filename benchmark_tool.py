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

# Configuration for benchmark runs
@dataclass
class BenchmarkConfig:
    num_agents: int
    num_rounds: int
    time_limit_seconds: int
    seed: int
    spatial_hints_enabled: bool
    
    @classmethod
    def from_env(cls) -> 'BenchmarkConfig':
        # Load configuration from environment variables
        return cls(
            num_agents=int(os.getenv('BENCHMARK_NUM_AGENTS', '2')),
            num_rounds=int(os.getenv('BENCHMARK_NUM_ROUNDS', '5')),
            time_limit_seconds=int(os.getenv('BENCHMARK_TIME_LIMIT_SECONDS', '300')),
            seed=int(os.getenv('BENCHMARK_SEED', '42')),
            spatial_hints_enabled=os.getenv('BENCHMARK_SPATIAL_HINTS_ENABLED', 'true').lower() == 'true'
        )

# Results from a single benchmark round
@dataclass
class RoundResult:
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

# Configuration for lifelong benchmark runs
@dataclass
class LifelongBenchmarkConfig:
    num_agents: int
    num_rounds: int
    duration_seconds: int  # Wall-clock time limit per round
    seed: int
    spatial_hints_enabled: bool

    @classmethod
    def from_env(cls) -> 'LifelongBenchmarkConfig':
        return cls(
            num_agents=int(os.getenv('LIFELONG_NUM_AGENTS', '2')),
            num_rounds=int(os.getenv('LIFELONG_NUM_ROUNDS', '3')),
            duration_seconds=int(os.getenv('LIFELONG_DURATION_SECONDS', '120')),
            seed=int(os.getenv('LIFELONG_SEED', '42')),
            spatial_hints_enabled=os.getenv('LIFELONG_SPATIAL_HINTS_ENABLED', 'true').lower() == 'true'
        )

# Results from a single lifelong benchmark round
@dataclass
class LifelongRoundResult:
    round_num: int
    status: str  # Always 'completed' in lifelong mode
    total_tasks_completed: int
    throughput_tasks_per_second: float
    throughput_tasks_per_turn: float
    total_turns: int
    total_negotiations: int
    collision_rate: float
    total_tokens_used: int
    tokens_per_task: float
    avg_conflict_resolution_time_ms: float
    duration_seconds: float


# Configuration for async benchmark runs
@dataclass
class AsyncBenchmarkConfig:
    num_agents: int
    num_rounds: int
    time_limit_seconds: int
    seed: int
    spatial_hints_enabled: bool

    @classmethod
    def from_env(cls) -> 'AsyncBenchmarkConfig':
        return cls(
            num_agents=int(os.getenv('BENCHMARK_NUM_AGENTS', '2')),
            num_rounds=int(os.getenv('BENCHMARK_NUM_ROUNDS', '5')),
            time_limit_seconds=int(os.getenv('BENCHMARK_TIME_LIMIT_SECONDS', '300')),
            seed=int(os.getenv('BENCHMARK_SEED', '42')),
            spatial_hints_enabled=os.getenv('BENCHMARK_SPATIAL_HINTS_ENABLED', 'true').lower() == 'true'
        )

# Results from a single async benchmark round
@dataclass
class AsyncRoundResult:
    round_num: int
    status: str  # 'success', 'timeout', 'failed'
    total_ticks: int
    total_negotiation_events: int
    total_tasks_completed: int
    total_tokens_used: int
    avg_conflict_resolution_time_ms: float
    collision_rate: float
    duration_seconds: float


def load_benchmark_config() -> BenchmarkConfig:
    # Load and display benchmark configuration from .env
    config = BenchmarkConfig.from_env()
    
    print(f"\n{Fore.CYAN}📊 Benchmark Configuration:{Style.RESET_ALL}")
    print(f"   Number of Agents: {config.num_agents}")
    print(f"   Number of Rounds: {config.num_rounds}")
    print(f"   Time Limit/Round: {config.time_limit_seconds}s")
    print(f"   Random Seed: {config.seed}")
    print(f"   Spatial Hints: {'Enabled' if config.spatial_hints_enabled else 'Disabled'}")
    
    return config

# List all walkable cells in the grid that are not walls
def get_walkable_cells(grid: List[str]) -> List[Tuple[int, int]]:
    walkable = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell in '.@ABT':  # Walkable cells (floor, agent, box, target markers)
                walkable.append((x, y))
    return walkable

# Use BFS to check if goal is reachable from start
def bfs_reachable(grid: List[str], start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
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

# Get all walkable cells adjacent to the given position (including the position itself)
def get_adjacent_walkable_cells(grid: List[str], pos: Tuple[int, int]) -> List[Tuple[int, int]]:
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

# Get all cells reachable from start using BFS
def bfs_all_reachable(grid: List[str], start: Tuple[int, int]) -> set:
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

# Generate a random grid with walls while ensuring connectivity
def generate_random_grid(width: int, height: int, wall_count: int, seed: int, min_walkable: int = 4) -> Optional[Tuple[List[str], int]]:
    rng = random.Random(seed)
    
    # Initialize grid with borders as walls and interior as floor
    # Work on mutable grid throughout to avoid repeated conversions
    grid_list = []
    for y in range(height):
        if y == 0 or y == height - 1:
            grid_list.append(list('#' * width))
        else:
            grid_list.append(list('#' + '.' * (width - 2) + '#'))
    
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
    failed_attempts = 0
    # Cap consecutive failures to keep generation time predictable
    max_consecutive_failures = max(10, wall_count)
    
    # Shuffle internal cells for random selection
    rng.shuffle(internal_cells)
    cell_index = 0
    
    for attempt in range(max_attempts):
        if walls_placed >= wall_count:
            break
        
        # Pick random internal cell
        if cell_index >= len(internal_cells):
            break
        
        wall_pos = internal_cells[cell_index]
        x, y = wall_pos
        cell_index += 1
        
        # Skip if already a wall
        if grid_list[y][x] == '#':
            continue
        
        # Tentatively place wall
        original_cell = grid_list[y][x]
        grid_list[y][x] = '#'
        test_grid = [''.join(row) for row in grid_list]
        
        # Check if all walkable cells are still connected
        walkable = get_walkable_cells(test_grid)
        
        if len(walkable) < min_walkable:
            # Not enough walkable cells, revert
            grid_list[y][x] = original_cell
            failed_attempts += 1
            if failed_attempts >= max_consecutive_failures:
                break
            continue
        
        # Pick any walkable cell as start and check if all others are reachable
        start = walkable[0]
        reachable = bfs_all_reachable(test_grid, start)
        
        if len(reachable) == len(walkable):
            # All walkable cells are connected, accept this wall
            walls_placed += 1
            failed_attempts = 0
        else:
            # Revert wall placement
            grid_list[y][x] = original_cell
            failed_attempts += 1
            if failed_attempts >= max_consecutive_failures:
                break
    
    # Convert to immutable grid
    grid = [''.join(row) for row in grid_list]
    
    # Ensure we have sufficient walkable cells for entities
    walkable = get_walkable_cells(grid)
    if len(walkable) < min_walkable:
        return None
    
    # Return None if we couldn't place at least 80% of requested walls
    # This indicates the parameters are incompatible
    if walls_placed < wall_count * 0.8:
        return None
    
    return (grid, walls_placed)


# Generate random valid positions for agents, boxes, and targets, agents spawn near boxes, targets reachable, no overlaps, all on walkable cells
def generate_random_positions(
    layout: Dict,
    num_agents: int,
    seed: int,
    round_num: int
) -> Optional[Dict[str, List[Dict]]]:
    
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

# Create a new layout with randomized entity positions
def create_benchmark_layout(base_layout: Dict, positions: Dict, num_agents: int) -> Dict:

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

# Run a single benchmark round
def run_single_round(
    layout: Dict,
    round_num: int,
    config: BenchmarkConfig,
    output_dir: str
) -> RoundResult:
    
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

# Save benchmark results to CSV file
def save_benchmark_csv(results: List[RoundResult], output_dir: str):

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

# Save benchmark summary JSON with aggregate statistics
def save_benchmark_summary(
    results: List[RoundResult],
    config: BenchmarkConfig,
    layout_name: str,
    output_dir: str
):

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

# Create a layout dictionary from a generated grid
def create_layout_from_grid(grid: List[str], name: str = "random_map") -> Dict:

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

# Prompt user for random map generation parameters and generate the map
def prompt_for_random_map_generation(seed: int, num_agents: int) -> Optional[Dict]:
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
        
        # Calculate minimum walkable cells needed for the configured agents
        # Need at least 2 cells per agent (conservative estimate)
        min_walkable = max(4, num_agents * 2)
        
        # Warn if grid is too small for agent count
        if max_internal_cells < min_walkable:
            print(f"{Fore.RED}❌ Grid too small for {num_agents} agents. Need at least {min_walkable} walkable cells.{Style.RESET_ALL}")
            return None
        
        # Get wall specification (percentage or count)
        print(f"\n{Fore.YELLOW}Specify walls as percentage (%) or absolute count:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}(Grid will have {max_internal_cells} internal cells, needs {min_walkable} walkable for {num_agents} agents){Style.RESET_ALL}")
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
        
        # Check if enough space will remain for agents
        if max_internal_cells - wall_count < min_walkable:
            print(f"{Fore.RED}❌ Too many walls! With {wall_count} walls, not enough space for {num_agents} agents.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Try reducing walls or increasing map size.{Style.RESET_ALL}")
            return None
        
        # Generate the grid
        print(f"\n{Fore.YELLOW}🔨 Generating random map ({width}x{height} with ~{wall_count} walls)...{Style.RESET_ALL}")
        result = generate_random_grid(width, height, wall_count, seed, min_walkable)
        
        if result is None:
            print(f"{Fore.RED}❌ Failed to generate valid map with requested parameters{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Try reducing wall density or adjusting map size.{Style.RESET_ALL}")
            return None
        
        grid, actual_walls = result
        
        # Create layout
        layout = create_layout_from_grid(grid, f"random_{width}x{height}")
        
        # Display generated map
        print(f"\n{Fore.GREEN}✅ Map generated successfully!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Preview:{Style.RESET_ALL}")
        for row in grid:
            print(f"  {row}")
        
        walkable = get_walkable_cells(grid)
        total_cells = width * height
        wall_cells = sum(row.count('#') for row in grid)
        
        print(f"\n{Fore.CYAN}Map Statistics:{Style.RESET_ALL}")
        print(f"  Total cells: {total_cells}")
        print(f"  Wall cells: {wall_cells} ({wall_cells/total_cells*100:.1f}%)")
        print(f"  Internal walls placed: {actual_walls} (requested: {wall_count})")
        print(f"  Walkable cells: {len(walkable)} (minimum needed: {min_walkable})")
        
        if actual_walls < wall_count:
            print(f"\n{Fore.YELLOW}ℹ️  Note: Placed {actual_walls}/{wall_count} walls to maintain connectivity{Style.RESET_ALL}")
        
        return layout
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Cancelled.{Style.RESET_ALL}")
        return None
    except ValueError as e:
        print(f"{Fore.RED}❌ Invalid input: {e}{Style.RESET_ALL}")
        return None

# Display final benchmark summary to console
def display_final_summary(summary: Dict):

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

# Run the complete benchmark with ALL rounds
def run_benchmark(base_layout: Dict, config: BenchmarkConfig) -> List[RoundResult]:

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

# Run a single lifelong benchmark round
def run_lifelong_round(
    layout: Dict,
    round_num: int,
    config: 'LifelongBenchmarkConfig',
    output_dir: str
) -> 'LifelongRoundResult':

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"🔁 LIFELONG ROUND {round_num}/{config.num_rounds}")
    print(f"{'='*60}{Style.RESET_ALL}")

    try:
        layout_dims = layout['dimensions']
        num_agents = len(layout['agents'])

        game_engine = GameEngine(
            width=layout_dims['width'],
            height=layout_dims['height'],
            num_agents=num_agents
        )

        # Configure for lifelong mode
        game_engine.simulation_mode = 'lifelong'
        game_engine.timeout_seconds = config.duration_seconds
        game_engine.silent_mode = False
        game_engine.central_negotiator.set_spatial_hints(config.spatial_hints_enabled)
        game_engine.reset_token_usage()

        game_engine.warehouse_map = WarehouseMap.from_layout(layout)
        game_engine.agents = {}
        for agent in layout['agents']:
            agent_id = agent['id']
            position = (agent['x'], agent['y'])
            robot = RobotAgent(agent_id, position)
            game_engine.agents[agent_id] = robot

        print(f"   Layout: {layout.get('name', 'Benchmark Layout')}")
        print(f"   Dimensions: {layout_dims['width']}x{layout_dims['height']}")
        print(f"   Agents: {num_agents}")
        print(f"   Duration: {config.duration_seconds}s")

        game_engine.initialize_simulation()
        start_time = time.time()

        while game_engine.run_simulation_step():
            if game_engine.stop_requested:
                break

        elapsed = time.time() - start_time
        print(f"\n{Fore.GREEN}✅ Lifelong round {round_num} completed after {elapsed:.1f}s{Style.RESET_ALL}")

        metrics = game_engine.calculate_performance_metrics()
        tokens = metrics.get('total_tokens_used', 0)
        tasks = game_engine.successful_deliveries

        log_filename = f"lifelong_log_round_{round_num}.json"
        game_engine.save_simulation_log_to_path(output_dir, log_filename)

        return LifelongRoundResult(
            round_num=round_num,
            status='completed',
            total_tasks_completed=tasks,
            throughput_tasks_per_second=metrics.get('throughput_tasks_per_second', 0),
            throughput_tasks_per_turn=metrics.get('throughput_tasks_per_turn', 0),
            total_turns=metrics.get('total_turns', 0),
            total_negotiations=metrics.get('total_negotiations', 0),
            collision_rate=metrics.get('collision_rate', 0),
            total_tokens_used=tokens,
            tokens_per_task=round(tokens / max(tasks, 1), 2),
            avg_conflict_resolution_time_ms=metrics.get('avg_conflict_resolution_time_ms', 0),
            duration_seconds=round(elapsed, 2)
        )

    except Exception as e:
        print(f"\n{Fore.RED}❌ Lifelong round {round_num} ERROR: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return LifelongRoundResult(
            round_num=round_num,
            status='failed',
            total_tasks_completed=0,
            throughput_tasks_per_second=0.0,
            throughput_tasks_per_turn=0.0,
            total_turns=0,
            total_negotiations=0,
            collision_rate=0.0,
            total_tokens_used=0,
            tokens_per_task=0.0,
            avg_conflict_resolution_time_ms=0.0,
            duration_seconds=0.0
        )

# Run the complete lifelong benchmark
def run_lifelong_benchmark(base_layout: Dict, config: 'LifelongBenchmarkConfig') -> List['LifelongRoundResult']:

    layout_name = base_layout.get('name', 'unknown').replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(
        'logs', 'Benchmarks',
        f"lifelong_{layout_name}_{config.num_agents}agents_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{Fore.GREEN}📁 Lifelong output directory: {output_dir}{Style.RESET_ALL}")

    results: List[LifelongRoundResult] = []

    for round_num in range(1, config.num_rounds + 1):
        print(f"\n🎲 Generating random positions for lifelong round {round_num}...")
        positions = generate_random_positions(base_layout, config.num_agents, config.seed, round_num)

        if positions is None:
            print(f"{Fore.RED}❌ Position generation failed. Stopping benchmark.{Style.RESET_ALL}")
            break

        round_layout = create_benchmark_layout(base_layout, positions, config.num_agents)
        round_layout['name'] = f"{layout_name}_lifelong_round_{round_num}"

        result = run_lifelong_round(round_layout, round_num, config, output_dir)
        results.append(result)

        if round_num < config.num_rounds:
            print(f"\n⏳ Starting next lifelong round in 2 seconds...")
            time.sleep(2)

    if results:
        # Save lifelong CSV
        csv_path = os.path.join(output_dir, 'lifelong_results.csv')
        fieldnames = [
            'Round', 'Status', 'TotalTasks', 'Throughput_tasks_per_s',
            'Throughput_tasks_per_turn', 'TotalTurns', 'TotalNegotiations',
            'CollisionRate', 'TotalTokens', 'TokensPerTask',
            'AvgResolutionTime_ms', 'Duration_s'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'Round': r.round_num,
                    'Status': r.status,
                    'TotalTasks': r.total_tasks_completed,
                    'Throughput_tasks_per_s': r.throughput_tasks_per_second,
                    'Throughput_tasks_per_turn': r.throughput_tasks_per_turn,
                    'TotalTurns': r.total_turns,
                    'TotalNegotiations': r.total_negotiations,
                    'CollisionRate': r.collision_rate,
                    'TotalTokens': r.total_tokens_used,
                    'TokensPerTask': r.tokens_per_task,
                    'AvgResolutionTime_ms': r.avg_conflict_resolution_time_ms,
                    'Duration_s': r.duration_seconds,
                })
        print(f"📄 Lifelong CSV saved: {csv_path}")

        # Save lifelong summary JSON
        def safe_avg(vals):
            return sum(vals) / len(vals) if vals else 0.0

        completed = [r for r in results if r.status != 'failed']
        summary = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'layout_name': layout_name,
                'config': asdict(config),
                'mode': 'lifelong'
            },
            'overall_results': {
                'total_rounds': len(results),
                'completed_rounds': len(completed),
                'failed_rounds': len(results) - len(completed),
            },
            'average_metrics': {
                'avg_tasks_per_round': safe_avg([r.total_tasks_completed for r in completed]),
                'avg_throughput_tasks_per_second': safe_avg([r.throughput_tasks_per_second for r in completed]),
                'avg_throughput_tasks_per_turn': safe_avg([r.throughput_tasks_per_turn for r in completed]),
                'avg_turns_per_round': safe_avg([r.total_turns for r in completed]),
                'avg_negotiations_per_round': safe_avg([r.total_negotiations for r in completed]),
                'avg_collision_rate': safe_avg([r.collision_rate for r in completed]),
                'total_tokens_used': sum(r.total_tokens_used for r in results),
                'avg_tokens_per_round': safe_avg([r.total_tokens_used for r in results]),
                'avg_tokens_per_task': safe_avg([r.tokens_per_task for r in completed]),
                'avg_resolution_time_ms': safe_avg([r.avg_conflict_resolution_time_ms for r in completed]),
            },
            'per_round_results': [asdict(r) for r in results]
        }
        summary_path = os.path.join(output_dir, 'lifelong_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"📊 Lifelong summary saved: {summary_path}")

        # Display lifelong summary
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"🔁 LIFELONG BENCHMARK SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}")
        info = summary['benchmark_info']
        avg = summary['average_metrics']
        print(f"\n{Fore.WHITE}Layout: {info['layout_name']} | Agents: {info['config']['num_agents']} | Duration: {info['config']['duration_seconds']}s/round{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Average Performance:{Style.RESET_ALL}")
        print(f"   Tasks/Round:        {avg['avg_tasks_per_round']:.1f}")
        print(f"   Throughput (t/s):   {avg['avg_throughput_tasks_per_second']:.4f}")
        print(f"   Throughput (t/turn):{avg['avg_throughput_tasks_per_turn']:.4f}")
        print(f"   Avg Turns/Round:    {avg['avg_turns_per_round']:.1f}")
        print(f"   Avg Negotiations:   {avg['avg_negotiations_per_round']:.1f}")
        print(f"   Avg Collision Rate: {avg['avg_collision_rate']:.3f}")
        print(f"   Total Tokens:       {avg['total_tokens_used']}")
        print(f"   Avg Tokens/Task:    {avg['avg_tokens_per_task']:.1f}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    return results

# Run a single async benchmark round
def run_async_round(
    layout: Dict,
    round_num: int,
    config: 'AsyncBenchmarkConfig',
    output_dir: str
) -> 'AsyncRoundResult':

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"⚡ ASYNC ROUND {round_num}/{config.num_rounds}")
    print(f"{'='*60}{Style.RESET_ALL}")

    try:
        layout_dims = layout['dimensions']
        num_agents = len(layout['agents'])

        game_engine = GameEngine(
            width=layout_dims['width'],
            height=layout_dims['height'],
            num_agents=num_agents
        )

        # Configure for async mode with live display
        game_engine.simulation_mode = 'async'
        game_engine.timeout_seconds = config.time_limit_seconds
        game_engine.silent_mode = False  # Show live matplotlib window
        game_engine.central_negotiator.set_spatial_hints(config.spatial_hints_enabled)
        game_engine.reset_token_usage()

        game_engine.warehouse_map = WarehouseMap.from_layout(layout)
        game_engine.agents = {}
        for agent in layout['agents']:
            agent_id = agent['id']
            position = (agent['x'], agent['y'])
            robot = RobotAgent(agent_id, position)
            game_engine.agents[agent_id] = robot

        print(f"   Layout: {layout.get('name', 'Benchmark Layout')}")
        print(f"   Dimensions: {layout_dims['width']}x{layout_dims['height']}")
        print(f"   Agents: {num_agents}")

        game_engine.initialize_simulation()
        start_time = time.time()

        while game_engine.run_simulation_step():
            if game_engine.stop_requested:
                break

        elapsed = time.time() - start_time

        # Determine status
        if game_engine.stop_requested or (config.time_limit_seconds > 0 and elapsed >= config.time_limit_seconds):
            status = 'timeout'
            print(f"\n{Fore.YELLOW}⏱️  Async round {round_num} TIMEOUT after {elapsed:.1f}s{Style.RESET_ALL}")
        elif game_engine.simulation_complete:
            status = 'success'
            print(f"\n{Fore.GREEN}✅ Async round {round_num} COMPLETED in {elapsed:.1f}s{Style.RESET_ALL}")
        else:
            status = 'failed'
            print(f"\n{Fore.RED}❌ Async round {round_num} FAILED after {elapsed:.1f}s{Style.RESET_ALL}")

        metrics = game_engine.calculate_performance_metrics()

        # Count negotiation events from the in-memory logger (path-based log)
        neg_events = 0
        if game_engine.log_enabled and game_engine.logger:
            neg_events = len(game_engine.logger.log_data.get('negotiation_events', []))

        # Save async log
        log_filename = f"async_log_round_{round_num}.json"
        game_engine.save_simulation_log_to_path(output_dir, log_filename)

        return AsyncRoundResult(
            round_num=round_num,
            status=status,
            total_ticks=game_engine._async_tick,
            total_negotiation_events=neg_events,
            total_tasks_completed=game_engine.successful_deliveries,
            total_tokens_used=metrics.get('total_tokens_used', 0),
            avg_conflict_resolution_time_ms=metrics.get('avg_conflict_resolution_time_ms', 0),
            collision_rate=metrics.get('collision_rate', 0),
            duration_seconds=round(elapsed, 2)
        )

    except Exception as e:
        print(f"\n{Fore.RED}❌ Async round {round_num} ERROR: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return AsyncRoundResult(
            round_num=round_num,
            status='failed',
            total_ticks=0,
            total_negotiation_events=0,
            total_tasks_completed=0,
            total_tokens_used=0,
            avg_conflict_resolution_time_ms=0.0,
            collision_rate=0.0,
            duration_seconds=0.0
        )

# Run the complete async benchmark
def run_async_benchmark(base_layout: Dict, config: 'AsyncBenchmarkConfig') -> List['AsyncRoundResult']:

    layout_name = base_layout.get('name', 'unknown').replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(
        'logs', 'Benchmarks',
        f"async_{layout_name}_{config.num_agents}agents_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{Fore.GREEN}📁 Async benchmark output directory: {output_dir}{Style.RESET_ALL}")

    results: List[AsyncRoundResult] = []

    for round_num in range(1, config.num_rounds + 1):
        print(f"\n🎲 Generating random positions for async round {round_num}...")
        positions = generate_random_positions(base_layout, config.num_agents, config.seed, round_num)

        if positions is None:
            print(f"{Fore.RED}❌ Position generation failed. Stopping benchmark.{Style.RESET_ALL}")
            break

        round_layout = create_benchmark_layout(base_layout, positions, config.num_agents)
        round_layout['name'] = f"{layout_name}_async_round_{round_num}"

        result = run_async_round(round_layout, round_num, config, output_dir)
        results.append(result)

        if round_num < config.num_rounds:
            print(f"\n⏳ Starting next async round in 2 seconds...")
            time.sleep(2)

    if results:
        def safe_avg(vals):
            return sum(vals) / len(vals) if vals else 0.0

        completed = [r for r in results if r.status != 'failed']

        # Save CSV
        csv_path = os.path.join(output_dir, 'async_results.csv')
        fieldnames = [
            'Round', 'Status', 'TotalTicks', 'NegotiationEvents', 'TasksCompleted',
            'TotalTokens', 'AvgResolutionTime_ms', 'CollisionRate', 'Duration_s'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'Round': r.round_num,
                    'Status': r.status,
                    'TotalTicks': r.total_ticks,
                    'NegotiationEvents': r.total_negotiation_events,
                    'TasksCompleted': r.total_tasks_completed,
                    'TotalTokens': r.total_tokens_used,
                    'AvgResolutionTime_ms': r.avg_conflict_resolution_time_ms,
                    'CollisionRate': r.collision_rate,
                    'Duration_s': r.duration_seconds,
                })
        print(f"📄 Async CSV saved: {csv_path}")

        # Save summary JSON
        summary = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'layout_name': layout_name,
                'config': asdict(config),
                'mode': 'async'
            },
            'overall_results': {
                'total_rounds': len(results),
                'completed_rounds': len(completed),
                'failed_rounds': len(results) - len(completed),
            },
            'average_metrics': {
                'avg_ticks_per_round': safe_avg([r.total_ticks for r in completed]),
                'avg_negotiation_events_per_round': safe_avg([r.total_negotiation_events for r in completed]),
                'avg_tasks_per_round': safe_avg([r.total_tasks_completed for r in completed]),
                'total_tokens_used': sum(r.total_tokens_used for r in results),
                'avg_tokens_per_round': safe_avg([r.total_tokens_used for r in results]),
                'avg_collision_rate': safe_avg([r.collision_rate for r in completed]),
                'avg_resolution_time_ms': safe_avg([r.avg_conflict_resolution_time_ms for r in completed]),
            },
            'per_round_results': [asdict(r) for r in results]
        }
        summary_path = os.path.join(output_dir, 'async_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"📊 Async summary saved: {summary_path}")

        # Display summary
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"⚡ ASYNC BENCHMARK SUMMARY")
        print(f"{'='*60}{Style.RESET_ALL}")
        info = summary['benchmark_info']
        avg = summary['average_metrics']
        print(f"\n{Fore.WHITE}Layout: {info['layout_name']} | Agents: {info['config']['num_agents']} | Time limit: {info['config']['time_limit_seconds']}s/round{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Average Performance:{Style.RESET_ALL}")
        print(f"   Ticks/Round:          {avg['avg_ticks_per_round']:.1f}")
        print(f"   Negotiation Events:   {avg['avg_negotiation_events_per_round']:.1f}")
        print(f"   Tasks/Round:          {avg['avg_tasks_per_round']:.1f}")
        print(f"   Total Tokens:         {avg['total_tokens_used']}")
        print(f"   Avg Tokens/Round:     {avg['avg_tokens_per_round']:.0f}")
        print(f"   Avg Collision Rate:   {avg['avg_collision_rate']:.3f}")
        print(f"   Avg Resolution Time:  {avg['avg_resolution_time_ms']:.2f}ms")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    return results

# Main entry point
def main():

    print(f"{Fore.CYAN}{'='*60}")
    print(f"🔬 LLM Multi-Robot Navigation Benchmark Tool")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print(f"{Fore.RED}⚠️  WARNING: OpenRouter API key not configured!{Style.RESET_ALL}")
        print("Please set OPENROUTER_API_KEY in your .env file")
        return

    # --- Benchmark mode selection ---
    print(f"\n{Fore.CYAN}Select benchmark mode:{Style.RESET_ALL}")
    print(f"  1. Turn-based benchmark  (CSR / makespan focus)")
    print(f"  2. Lifelong benchmark    (throughput / tasks-per-second focus)")
    print(f"  3. Async benchmark       (parallel ticks / negotiation events focus)")
    bench_mode = input(f"\n{Fore.CYAN}Select mode (1/2/3, default 1): {Style.RESET_ALL}").strip()

    is_lifelong = bench_mode == '2'
    is_async = bench_mode == '3'

    if is_lifelong:
        config = LifelongBenchmarkConfig.from_env()
        print(f"\n{Fore.CYAN}📊 Lifelong Benchmark Configuration:{Style.RESET_ALL}")
        print(f"   Number of Agents: {config.num_agents}")
        print(f"   Number of Rounds: {config.num_rounds}")
        print(f"   Duration/Round:   {config.duration_seconds}s")
        print(f"   Random Seed:      {config.seed}")
        print(f"   Spatial Hints:    {'Enabled' if config.spatial_hints_enabled else 'Disabled'}")
        num_agents = config.num_agents
        num_rounds = config.num_rounds
    elif is_async:
        config = AsyncBenchmarkConfig.from_env()
        print(f"\n{Fore.CYAN}📊 Async Benchmark Configuration:{Style.RESET_ALL}")
        print(f"   Number of Agents: {config.num_agents}")
        print(f"   Number of Rounds: {config.num_rounds}")
        print(f"   Time Limit/Round: {config.time_limit_seconds}s")
        print(f"   Random Seed:      {config.seed}")
        print(f"   Spatial Hints:    {'Enabled' if config.spatial_hints_enabled else 'Disabled'}")
        num_agents = config.num_agents
        num_rounds = config.num_rounds
    else:
        config = load_benchmark_config()
        num_agents = config.num_agents
        num_rounds = config.num_rounds

        # Validate turn-based configuration
        if config.num_agents < 1 or config.num_agents > 10:
            print(f"{Fore.RED}❌ Invalid number of agents ({config.num_agents}). Must be 1-10.{Style.RESET_ALL}")
            return
        if config.num_rounds < 1:
            print(f"{Fore.RED}❌ Invalid number of rounds ({config.num_rounds}). Must be >= 1.{Style.RESET_ALL}")
            return

    # Select layout or generate random map
    print(f"\n{Fore.CYAN}Choose layout source:{Style.RESET_ALL}")
    print(f"  1. Select existing layout")
    print(f"  2. Generate random map")
    
    mode_choice = input(f"\n{Fore.CYAN}Select mode (1/2, default 1): {Style.RESET_ALL}").strip()
    
    if mode_choice == '2':
        base_layout = prompt_for_random_map_generation(config.seed, num_agents)
        if base_layout is None:
            print(f"{Fore.RED}Random map generation cancelled. Exiting.{Style.RESET_ALL}")
            return
    else:
        print(f"\n{Fore.CYAN}Select a layout for benchmarking:{Style.RESET_ALL}")
        base_layout = select_layout_interactive()
        if base_layout is None:
            print(f"{Fore.RED}No layout selected. Exiting.{Style.RESET_ALL}")
            return

    # Confirm benchmark start
    mode_label = 'lifelong ' if is_lifelong else ('async ' if is_async else '')
    print(f"\n{Fore.YELLOW}Ready to start {mode_label}benchmark:{Style.RESET_ALL}")
    print(f"   Layout: {base_layout.get('name', 'Unknown')}")
    print(f"   Agents: {num_agents}")
    print(f"   Rounds: {num_rounds}")
    if is_lifelong:
        print(f"   Duration/Round: {config.duration_seconds}s")
    else:
        print(f"   Time Limit: {config.time_limit_seconds}s/round")
    print(f"   Seed: {config.seed}")

    confirm = input(f"\n{Fore.CYAN}Start benchmark? (Y/n): {Style.RESET_ALL}").strip().lower()
    if confirm == 'n':
        print("Benchmark cancelled.")
        return

    try:
        if is_lifelong:
            print(f"\n{Fore.GREEN}🚀 Starting Lifelong Benchmark...{Style.RESET_ALL}")
            results = run_lifelong_benchmark(base_layout, config)
        elif is_async:
            print(f"\n{Fore.GREEN}🚀 Starting Async Benchmark...{Style.RESET_ALL}")
            results = run_async_benchmark(base_layout, config)
        else:
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
