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
                    print(f"❌ Agent {agent_id}: {action_type} execution failed")\n    \n    def _execute_planned_moves(self, planned_moves: Dict):\n        \"\"\"Execute planned moves without conflicts\"\"\"\n        for agent_id, path in planned_moves.items():\n            if agent_id in self.agents and len(path) > 1:\n                agent = self.agents[agent_id]\n                next_pos = path[1]  # path[0] is current position\n                \n                map_state = self.warehouse_map.get_state_dict()\n                success = agent.move_to(next_pos, map_state)\n                \n                if success:\n                    print(f\"✅ Agent {agent_id}: Moved to {next_pos}\")\n                else:\n                    print(f\"❌ Agent {agent_id}: Move to {next_pos} failed\")\n    \n    def _update_map_state(self):\n        \"\"\"Update warehouse map with current agent positions\"\"\"\n        # Clear agent positions on map\n        for y in range(self.height):\n            for x in range(self.width):\n                if self.warehouse_map.grid[y, x] in [CellType.AGENT.value, CellType.AGENT_WITH_BOX.value]:\n                    self.warehouse_map.grid[y, x] = CellType.EMPTY.value\n        \n        # Place agents at new positions\n        for agent_id, agent in self.agents.items():\n            x, y = agent.position\n            if agent.carrying_box:\n                self.warehouse_map.grid[y, x] = CellType.AGENT_WITH_BOX.value\n            else:\n                self.warehouse_map.grid[y, x] = CellType.AGENT.value\n            \n            # Update warehouse map's agent tracking\n            self.warehouse_map.agents[agent_id] = agent.position\n    \n    def _check_completion(self) -> bool:\n        \"\"\"Check if all agents have reached their targets\"\"\"\n        for agent in self.agents.values():\n            if agent.target_position and agent.position != agent.target_position:\n                return False\n        return True\n    \n    def display_map(self):\n        \"\"\"Display the current warehouse map with colors\"\"\"\n        print(f\"\\n{Fore.CYAN}Current Warehouse State:{Style.RESET_ALL}\")\n        \n        # Add column numbers\n        header = \"   \" + \" \".join([str(i) for i in range(self.width)])\n        print(header)\n        \n        for y in range(self.height):\n            row = f\"{y}: \"\n            for x in range(self.width):\n                cell = self.warehouse_map.grid[y, x]\n                \n                # Color coding\n                if cell == CellType.AGENT.value:\n                    row += f\"{Fore.BLUE}{cell}{Style.RESET_ALL} \"\n                elif cell == CellType.AGENT_WITH_BOX.value:\n                    row += f\"{Fore.MAGENTA}{cell}{Style.RESET_ALL}\"\n                elif cell == CellType.BOX.value:\n                    row += f\"{Fore.YELLOW}{cell}{Style.RESET_ALL} \"\n                elif cell == CellType.TARGET.value:\n                    row += f\"{Fore.GREEN}{cell}{Style.RESET_ALL} \"\n                elif cell == CellType.WALL.value:\n                    row += f\"{Back.BLACK}{cell}{Style.RESET_ALL} \"\n                else:\n                    row += f\"{cell} \"\n            \n            print(row)\n    \n    def _display_agent_status(self):\n        \"\"\"Display detailed status of all agents\"\"\"\n        print(f\"\\n{Fore.CYAN}Agent Status:{Style.RESET_ALL}\")\n        for agent_id, agent in self.agents.items():\n            status = agent.get_status()\n            target_dist = agent.distance_to_target()\n            \n            status_color = Fore.GREEN if status['position'] == status['target'] else Fore.WHITE\n            print(f\"{status_color}Agent {agent_id}: {status['position']} → {status['target']} (dist: {target_dist:.0f}){Style.RESET_ALL}\")\n            \n            if status['is_waiting']:\n                print(f\"  ⏳ Waiting {status['wait_turns_remaining']} more turns\")\n            \n            if status['planned_path']:\n                print(f\"  🗺️  Path: {status['planned_path'][:5]}{'...' if len(status['planned_path']) > 5 else ''}\")\n    \n    def _log_turn_state(self, event_type: str):\n        \"\"\"Log current simulation state\"\"\"\n        if not self.log_enabled:\n            return\n        \n        log_entry = {\n            'timestamp': datetime.now().isoformat(),\n            'turn': self.current_turn,\n            'event_type': event_type,\n            'map_state': self.warehouse_map.get_state_dict(),\n            'agent_status': {aid: agent.get_status() for aid, agent in self.agents.items()}\n        }\n        \n        self.simulation_log.append(log_entry)\n    \n    def save_simulation_log(self, filename: Optional[str] = None):\n        \"\"\"Save simulation log to file\"\"\"\n        if not self.simulation_log:\n            return\n        \n        if filename is None:\n            timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n            filename = f\"simulation_log_{timestamp}.json\"\n        \n        log_path = os.path.join(\"logs\", filename)\n        os.makedirs(\"logs\", exist_ok=True)\n        \n        with open(log_path, 'w') as f:\n            json.dump(self.simulation_log, f, indent=2)\n        \n        print(f\"Simulation log saved to: {log_path}\")\n    \n    def run_interactive_simulation(self):\n        \"\"\"Run simulation with step-by-step user input\"\"\"\n        self.initialize_simulation()\n        \n        print(f\"\\n{Fore.CYAN}🚀 Starting Interactive Simulation{Style.RESET_ALL}\")\n        print(\"Commands: [Enter] = Next step, 'q' = Quit, 'auto' = Auto-run\")\n        \n        auto_mode = False\n        \n        while self.run_simulation_step():\n            if not auto_mode:\n                user_input = input(\"\\nPress Enter for next step (or command): \").strip().lower()\n                \n                if user_input == 'q':\n                    print(\"Simulation terminated by user.\")\n                    break\n                elif user_input == 'auto':\n                    auto_mode = True\n                    print(\"Switching to auto mode...\")\n            else:\n                time.sleep(1)  # Auto delay\n        \n        # Save log when simulation ends\n        self.save_simulation_log()\n        print(f\"\\n{Fore.GREEN}Simulation completed in {self.current_turn} turns!{Style.RESET_ALL}\")
