"""
Main Game Engine for Multi-Robot Warehouse Simulation
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from colorama import init, Fore, Back, Style

from ..map_generator import WarehouseMap, CellType
from ..agents import RobotAgent
from ..llm.central_negotiator import CentralNegotiator
from ..navigation import ConflictDetector, SimplePathfinder

# Initialize colorama for colored terminal output
init(autoreset=True)

class GameEngine:
    def __init__(self, width: int = 8, height: int = 6, num_agents: int = 2):
        self.width = width
        self.height = height
        self.num_agents = max(2, min(num_agents, 4))  # Ensure 2-4 agents
        
        # Core components
        self.warehouse_map = WarehouseMap(width, height)
        self.agents = {}
        self.central_negotiator = CentralNegotiator()
        self.conflict_detector = ConflictDetector(width, height)
        self.pathfinder = SimplePathfinder(width, height)
        
        # Simulation state
        self.current_turn = 0
        self.max_turns = 100
        self.is_running = False
        self.simulation_complete = False
        
        # Logging
        self.log_enabled = os.getenv('LOG_SIMULATION', 'true').lower() == 'true'
        self.simulation_log = []
        
    def initialize_simulation(self):
        """Initialize a new simulation"""
        print(f"{Fore.CYAN}Initializing Multi-Robot Warehouse Simulation...{Style.RESET_ALL}")
        
        # Generate map
        self.warehouse_map.generate_map(num_agents=self.num_agents, wall_density=0.1)
        print(f"Generated {self.width}x{self.height} warehouse map with {self.num_agents} agents")
        
        # Create agents
        self.agents = {}
        for agent_id in range(self.num_agents):
            position = self.warehouse_map.agents[agent_id]
            agent = RobotAgent(agent_id, position)
            
            # Assign target (simplified: each agent goes to a target)
            if agent_id in self.warehouse_map.agent_goals:
                target_id = self.warehouse_map.agent_goals[agent_id]
                target_pos = self.warehouse_map.targets[target_id]
                agent.set_target(target_pos)
            
            self.agents[agent_id] = agent
        
        # Initial pathfinding
        self._plan_initial_paths()
        
        # Log initial state
        self._log_turn_state("SIMULATION_START")
        
        print(f"{Fore.GREEN}Simulation initialized successfully!{Style.RESET_ALL}")
        self.display_map()
    
    def _plan_initial_paths(self):
        """Plan initial paths for all agents"""
        print("Planning initial paths for all agents...")
        
        for agent_id, agent in self.agents.items():
            map_state = self.warehouse_map.get_state_dict()
            path = agent.plan_path(map_state)
            print(f"Agent {agent_id}: Path planned ({len(path)} steps)")
    
    def run_simulation_step(self) -> bool:
        """
        Run one step of the simulation
        
        Returns:
            bool: True if simulation should continue, False if complete
        """
        if self.simulation_complete or self.current_turn >= self.max_turns:
            return False
        
        print(f"\n{Fore.YELLOW}=== TURN {self.current_turn + 1} ==={Style.RESET_ALL}")
        
        # Update all agents for new turn
        for agent in self.agents.values():
            agent.update_turn()
        
        # Get planned moves for all active agents
        planned_moves = self._get_planned_moves()
        
        if not planned_moves:
            print("No agents have planned moves. Simulation complete!")
            self.simulation_complete = True
            return False
        
        # Check for conflicts
        conflict_info = self.conflict_detector.detect_path_conflicts(planned_moves, self.current_turn)
        
        if conflict_info['has_conflicts']:
            print(f"{Fore.RED}CONFLICT DETECTED!{Style.RESET_ALL}")
            print(f"Conflicting agents: {conflict_info['conflicting_agents']}")
            print(f"Conflict points: {conflict_info['conflict_points']}")
            
            # Use Central Negotiator to resolve conflicts
            resolution = self._negotiate_conflicts(conflict_info, planned_moves)
            self._execute_negotiated_actions(resolution)
        else:
            print(f"{Fore.GREEN}No conflicts detected. Executing planned moves...{Style.RESET_ALL}")
            self._execute_planned_moves(planned_moves)
        
        # Update map with new agent positions
        self._update_map_state()
        
        # Check if simulation is complete
        if self._check_completion():
            print(f"{Fore.GREEN}🎉 All agents reached their targets! Simulation complete!{Style.RESET_ALL}")
            self.simulation_complete = True
        
        # Log turn state
        self._log_turn_state("TURN_COMPLETE")
        
        # Display current state
        self.display_map()
        self._display_agent_status()
        
        self.current_turn += 1
        return not self.simulation_complete
    
    def _get_planned_moves(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get planned moves for all active agents"""
        planned_moves = {}
        
        for agent_id, agent in self.agents.items():
            if not agent.is_waiting and agent.target_position:
                # Get or update path
                if not agent.planned_path:
                    map_state = self.warehouse_map.get_state_dict()
                    agent.plan_path(map_state)
                
                if agent.planned_path:
                    planned_moves[agent_id] = agent.planned_path.copy()
        
        return planned_moves
    
    def _negotiate_conflicts(self, conflict_info: Dict, planned_moves: Dict) -> Dict:
        """Use Central Negotiator to resolve conflicts"""
        print("🤖 Initiating LLM-based conflict negotiation...")
        
        # Prepare conflict data for negotiator
        conflict_data = {
            'agents': [],
            'conflict_points': conflict_info['conflict_points'],
            'map_state': self.warehouse_map.get_state_dict(),
            'turn': self.current_turn
        }
        
        # Add agent data for conflicting agents
        for agent_id in conflict_info['conflicting_agents']:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                conflict_data['agents'].append({
                    'id': agent_id,
                    'current_pos': agent.position,
                    'target_pos': agent.target_position,
                    'planned_path': planned_moves.get(agent_id, [])
                })
        
        # Get negotiation result from Central LLM
        resolution = self.central_negotiator.negotiate_path_conflict(conflict_data)
        
        print(f"🤖 Negotiation complete: {resolution.get('resolution', 'unknown')}")
        print(f"📝 Reasoning: {resolution.get('reasoning', 'No reasoning provided')}")
        
        return resolution
    
    def _execute_negotiated_actions(self, resolution: Dict):
        """Execute actions determined by negotiation"""
        agent_actions = resolution.get('agent_actions', {})
        
        for agent_id, action_data in agent_actions.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                map_state = self.warehouse_map.get_state_dict()
                
                success = agent.execute_negotiated_action(action_data, map_state)
                action_type = action_data.get('action', 'unknown')
                
                if success:
                    print(f"✅ Agent {agent_id}: {action_type} executed successfully")
                else:
                    print(f"❌ Agent {agent_id}: {action_type} execution failed")
    
    def _execute_planned_moves(self, planned_moves: Dict):
        """Execute planned moves without conflicts"""
        for agent_id, path in planned_moves.items():
            if agent_id in self.agents and len(path) > 1:
                agent = self.agents[agent_id]
                next_pos = path[1]  # path[0] is current position
                
                map_state = self.warehouse_map.get_state_dict()
                success = agent.move_to(next_pos, map_state)
                
                if success:
                    print(f"✅ Agent {agent_id}: Moved to {next_pos}")
                else:
                    print(f"❌ Agent {agent_id}: Move to {next_pos} failed")
    
    def _update_map_state(self):
        """Update warehouse map with current agent positions"""
        # Clear agent positions on map
        for y in range(self.height):
            for x in range(self.width):
                if self.warehouse_map.grid[y, x] in [CellType.AGENT.value, CellType.AGENT_WITH_BOX.value]:
                    self.warehouse_map.grid[y, x] = CellType.EMPTY.value
        
        # Place agents at new positions
        for agent_id, agent in self.agents.items():
            x, y = agent.position
            if agent.carrying_box:
                self.warehouse_map.grid[y, x] = CellType.AGENT_WITH_BOX.value
            else:
                self.warehouse_map.grid[y, x] = CellType.AGENT.value
            
            # Update warehouse map's agent tracking
            self.warehouse_map.agents[agent_id] = agent.position
    
    def _check_completion(self) -> bool:
        """Check if all agents have reached their targets"""
        for agent in self.agents.values():
            if agent.target_position and agent.position != agent.target_position:
                return False
        return True
    
    def display_map(self):
        """Display the current warehouse map with colors"""
        print(f"\n{Fore.CYAN}Current Warehouse State:{Style.RESET_ALL}")
        
        # Add column numbers
        header = "   " + " ".join([str(i) for i in range(self.width)])
        print(header)
        
        for y in range(self.height):
            row = f"{y}: "
            for x in range(self.width):
                cell = self.warehouse_map.grid[y, x]
                
                # Color coding
                if cell == CellType.AGENT.value:
                    row += f"{Fore.BLUE}{cell}{Style.RESET_ALL} "
                elif cell == CellType.AGENT_WITH_BOX.value:
                    row += f"{Fore.MAGENTA}{cell}{Style.RESET_ALL}"
                elif cell == CellType.BOX.value:
                    row += f"{Fore.YELLOW}{cell}{Style.RESET_ALL} "
                elif cell == CellType.TARGET.value:
                    row += f"{Fore.GREEN}{cell}{Style.RESET_ALL} "
                elif cell == CellType.WALL.value:
                    row += f"{Back.BLACK}{cell}{Style.RESET_ALL} "
                else:
                    row += f"{cell} "
            
            print(row)
    
    def _display_agent_status(self):
        """Display detailed status of all agents"""
        print(f"\n{Fore.CYAN}Agent Status:{Style.RESET_ALL}")
        for agent_id, agent in self.agents.items():
            status = agent.get_status()
            target_dist = agent.distance_to_target()
            
            status_color = Fore.GREEN if status['position'] == status['target'] else Fore.WHITE
            print(f"{status_color}Agent {agent_id}: {status['position']} → {status['target']} (dist: {target_dist:.0f}){Style.RESET_ALL}")
            
            if status['is_waiting']:
                print(f"  ⏳ Waiting {status['wait_turns_remaining']} more turns")
            
            if status['planned_path']:
                print(f"  🗺️  Path: {status['planned_path'][:5]}{'...' if len(status['planned_path']) > 5 else ''}")
    
    def _log_turn_state(self, event_type: str):
        """Log current simulation state"""
        if not self.log_enabled:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'turn': self.current_turn,
            'event_type': event_type,
            'map_state': self.warehouse_map.get_state_dict(),
            'agent_status': {aid: agent.get_status() for aid, agent in self.agents.items()}
        }
        
        self.simulation_log.append(log_entry)
    
    def save_simulation_log(self, filename: Optional[str] = None):
        """Save simulation log to file"""
        if not self.simulation_log:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_log_{timestamp}.json"
        
        log_path = os.path.join("logs", filename)
        os.makedirs("logs", exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(self.simulation_log, f, indent=2)
        
        print(f"Simulation log saved to: {log_path}")
    
    def run_interactive_simulation(self):
        """Run simulation with step-by-step user input"""
        self.initialize_simulation()
        
        print(f"\n{Fore.CYAN}🚀 Starting Interactive Simulation{Style.RESET_ALL}")
        print("Commands: [Enter] = Next step, 'q' = Quit, 'auto' = Auto-run")
        
        auto_mode = False
        
        while self.run_simulation_step():
            if not auto_mode:
                user_input = input("\nPress Enter for next step (or command): ").strip().lower()
                
                if user_input == 'q':
                    print("Simulation terminated by user.")
                    break
                elif user_input == 'auto':
                    auto_mode = True
                    print("Switching to auto mode...")
            else:
                time.sleep(1)  # Auto delay
        
        # Save log when simulation ends
        self.save_simulation_log()
        print(f"\n{Fore.GREEN}Simulation completed in {self.current_turn} turns!{Style.RESET_ALL}")
