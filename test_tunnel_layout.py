#!/usr/bin/env python3
"""
Test script for extreme layout generation - Perfect for triggering LLM negotiations!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.map_generator import WarehouseMap

def test_extreme_layouts():
    print("🔥 EXTREME LAYOUT TESTS - MAXIMUM CONFLICT GENERATION! 🔥\n")
    
    # Test 1: Original tunnel layout
    print("1. 🏗️  TUNNEL LAYOUT (Standard bottlenecks):")
    map_gen = WarehouseMap(8, 6)
    map_gen.generate_map(num_agents=3, layout_type='tunnel')
    print(map_gen.display())
    print(f"Agents: {map_gen.agents}")
    print(f"Targets: {map_gen.targets}")
    print()
    
    # Test 2: EXTREME single corridor - GUARANTEED conflicts!
    print("2. 🚨 EXTREME LAYOUT (Single serpentine corridor - MAXIMUM conflicts!):")
    map_gen2 = WarehouseMap(8, 6)
    map_gen2.generate_map(num_agents=4, layout_type='extreme')
    print(map_gen2.display())
    print(f"Agents: {map_gen2.agents}")
    print(f"Targets: {map_gen2.targets}")
    print("💥 This layout FORCES all agents through the same path!")
    print()
    
    # Test 3: Bridge layout - Single cell bottlenecks
    print("3. 🌉 BRIDGE LAYOUT (Chambers with single-cell bridges):")
    map_gen3 = WarehouseMap(8, 6)
    map_gen3.generate_map(num_agents=3, layout_type='bridge')
    print(map_gen3.display())
    print(f"Agents: {map_gen3.agents}")
    print(f"Targets: {map_gen3.targets}")
    print("🎯 Perfect for head-to-head negotiations!")
    print()
    
    # Test 4: Larger extreme layout for even more chaos
    print("4. 🔥 LARGE EXTREME LAYOUT (10x6 - Total mayhem!):")
    map_gen4 = WarehouseMap(10, 6)
    map_gen4.generate_map(num_agents=4, layout_type='extreme')
    print(map_gen4.display())
    print("🎪 This will create CONSTANT LLM negotiations!")
    print()
    
    # Test 5: Comparison with random for contrast
    print("5. 🎲 RANDOM LAYOUT (For comparison):")
    map_gen5 = WarehouseMap(8, 6)
    map_gen5.generate_map(num_agents=3, layout_type='random', wall_density=0.15)
    print(map_gen5.display())
    print()
    
    print("🎯 RECOMMENDATIONS FOR MAXIMUM LLM SHOWCASE:")
    print("   • Use 'extreme' layout with 3-4 agents")
    print("   • 'bridge' layout for strategic negotiations")
    print("   • 'tunnel' layout for balanced conflicts")
    print("   • Run with 4 agents for maximum chaos!")

if __name__ == "__main__":
    test_extreme_layouts()
