#!/usr/bin/env python3
"""
Quick test with updated positions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from test_negotiation import NegotiationTester
from src.navigation import SimplePathfinder

def test_updated_positions():
    print("🔍 TESTING UPDATED AGENT POSITIONS")
    
    tester = NegotiationTester()
    warehouse = tester.create_forced_conflict_map("s_shaped_corridor")
    
    print("Updated map:")
    print(warehouse.display())
    
    print(f"\nNew positions:")
    print(f"Agents: {warehouse.agents}")
    print(f"Targets: {warehouse.targets}")
    
    # Create pathfinder
    pathfinder = SimplePathfinder(warehouse.width, warehouse.height)
    
    # Get walls
    walls = set()
    for y in range(warehouse.height):
        for x in range(warehouse.width):
            if warehouse.grid[y, x] == '#':
                walls.add((x, y))
    
    # Test pathfinding for each agent
    for agent_id, start_pos in warehouse.agents.items():
        goal_id = warehouse.agent_goals.get(agent_id)
        if goal_id in warehouse.targets:
            target_pos = warehouse.targets[goal_id]
            
            print(f"\n--- Agent {agent_id}: {start_pos} → {target_pos} ---")
            
            # Test without avoiding other agents
            path_no_avoid = pathfinder.find_path_with_obstacles(
                start=start_pos,
                goal=target_pos,
                walls=walls,
                agent_positions={},
                exclude_agent=agent_id
            )
            print(f"Path (no avoid): {len(path_no_avoid) if path_no_avoid else 0} steps")
            
            # Test avoiding other agents
            other_agents = {aid: pos for aid, pos in warehouse.agents.items() if aid != agent_id}
            path_with_avoid = pathfinder.find_path_with_obstacles(
                start=start_pos,
                goal=target_pos,
                walls=walls,
                agent_positions=other_agents,
                exclude_agent=agent_id
            )
            print(f"Path (avoid others): {len(path_with_avoid) if path_with_avoid else 0} steps")
            
            if path_no_avoid:
                print(f"Path preview: {path_no_avoid[:5]}...")
            
            # Check if positions are valid
            print(f"Start in walls? {start_pos in walls}")
            print(f"Target in walls? {target_pos in walls}")

if __name__ == "__main__":
    test_updated_positions()
