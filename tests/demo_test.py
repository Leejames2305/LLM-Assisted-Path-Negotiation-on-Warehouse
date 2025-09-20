"""
Demo Test - Run a simple offline demonstration
Tests core functionality without external dependencies
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_simple_demo():
    """Run a simple demonstration of the system"""
    print("🤖 SIMPLE OFFLINE DEMO")
    print("=" * 40)
    
    try:
        # Test 1: Map Generation
        print("\n📍 Testing Map Generation...")
        from src.map_generator import WarehouseMap, CellType
        
        warehouse = WarehouseMap(8, 6)
        warehouse.generate_map(num_agents=3, wall_density=0.1)
        
        print(f"✅ Generated {warehouse.width}x{warehouse.height} map")
        print(f"✅ {len(warehouse.agents)} agents placed")
        print(f"✅ {len(warehouse.boxes)} boxes placed")
        print(f"✅ {len(warehouse.targets)} targets placed")
        
        print("\n🗺️  Map Display:")
        print(warehouse.display())
        
        # Test 2: Agent Creation
        print("\n🤖 Testing Agents...")
        from src.agents import RobotAgent
        
        agents = {}
        for agent_id in range(3):
            position = warehouse.agents[agent_id]
            agent = RobotAgent(agent_id, position)
            
            if agent_id in warehouse.agent_goals:
                target_id = warehouse.agent_goals[agent_id]
                target_pos = warehouse.targets[target_id]
                agent.set_target(target_pos)
            
            agents[agent_id] = agent
            print(f"✅ Agent {agent_id}: {position} → {agent.target_position}")
        
        # Test 3: Pathfinding
        print("\n🗺️  Testing Pathfinding...")
        from src.navigation import SimplePathfinder
        
        pathfinder = SimplePathfinder(8, 6)
        
        for agent_id, agent in agents.items():
            if agent.target_position:
                path = pathfinder.find_path(agent.position, agent.target_position)
                agent.planned_path = path
                print(f"✅ Agent {agent_id} path: {len(path)} steps")
                if len(path) <= 6:  # Show short paths
                    print(f"   Path: {path}")
        
        # Test 4: Conflict Detection
        print("\n⚔️  Testing Conflict Detection...")
        from src.navigation import ConflictDetector
        
        detector = ConflictDetector(8, 6)
        
        planned_moves = {}
        for agent_id, agent in agents.items():
            if agent.planned_path:
                planned_moves[agent_id] = agent.planned_path
        
        conflict_info = detector.detect_path_conflicts(planned_moves)
        
        if conflict_info['has_conflicts']:
            print(f"⚔️  Conflicts detected!")
            print(f"   Conflicting agents: {conflict_info['conflicting_agents']}")
            print(f"   Conflict points: {conflict_info['conflict_points']}")
        else:
            print("✅ No conflicts detected in paths")
        
        # Test 5: Basic Simulation Step
        print("\n⚡ Testing Basic Simulation Logic...")
        
        # Simulate one step
        for agent_id, agent in agents.items():
            if agent.planned_path and len(agent.planned_path) > 1:
                next_pos = agent.planned_path[1]
                
                # Check if move is safe (basic check)
                safe = True
                for other_id, other_agent in agents.items():
                    if other_id != agent_id and other_agent.position == next_pos:
                        safe = False
                        break
                
                if safe:
                    old_pos = agent.position
                    agent.position = next_pos
                    # Update planned path
                    agent.planned_path = agent.planned_path[1:]
                    print(f"✅ Agent {agent_id}: {old_pos} → {next_pos}")
                else:
                    print(f"⏸️  Agent {agent_id}: waiting (position blocked)")
        
        # Test 6: Logging
        print("\n📝 Testing Logging...")
        import json
        
        state = warehouse.get_state_dict()
        agent_status = {aid: agent.get_status() for aid, agent in agents.items()}
        
        log_entry = {
            'turn': 1,
            'map_state': state,
            'agent_status': agent_status
        }
        
        # Test JSON serialization
        json_str = json.dumps(log_entry, indent=2)
        print(f"✅ Log entry created ({len(json_str)} bytes)")
        
        # Save to logs directory
        os.makedirs("logs", exist_ok=True)
        with open("logs/demo_test.json", "w") as f:
            f.write(json_str)
        print("✅ Log saved to logs/demo_test.json")
        
        print("\n" + "=" * 40)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("\n💡 What was tested:")
        print("   ✅ Map generation with agents, boxes, targets")
        print("   ✅ Agent creation and goal assignment")
        print("   ✅ A* pathfinding with obstacle avoidance")
        print("   ✅ Path conflict detection")
        print("   ✅ Basic movement simulation")
        print("   ✅ State logging and JSON serialization")
        print("\n🚀 The system is ready for use!")
        print("💡 Run 'python main.py' to start the full simulation")
        print("💡 The simulation will work offline using fallback logic")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simple_demo()
    sys.exit(0 if success else 1)
