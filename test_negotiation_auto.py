#!/usr/bin/env python3
"""
Demo Negotiation Test - Run a quick automatic negotiation test
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from test_negotiation import NegotiationTester

def demo_negotiation():
    """Run a simple automatic negotiation test"""
    print("🤖 DEMO NEGOTIATION TEST")
    print("=" * 50)
    print("Running single corridor conflict scenario automatically...")
    
    tester = NegotiationTester()
    
    # Override the input() calls to make it automatic
    import builtins
    original_input = builtins.input
    input_responses = [""] * 20  # Just press enter for each step
    input_iter = iter(input_responses)
    
    def mock_input(prompt=""):
        try:
            response = next(input_iter)
            print(f"{prompt}{response}")
            return response
        except StopIteration:
            return ""
    
    builtins.input = mock_input
    
    try:
        # Run the test with max 5 turns
        scenario_data = tester.run_forced_conflict_test("single_corridor", max_turns=5)
        
        # Save the results
        log_file = tester.save_negotiation_log("demo_negotiation.json")
        
        print(f"\n🎉 DEMO COMPLETE!")
        print(f"📊 Results:")
        print(f"   Conflicts detected: {scenario_data.get('total_conflicts', 0)}")
        print(f"   Negotiations: {len(scenario_data.get('negotiations', []))}")
        print(f"   Log saved to: {log_file}")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original input
        builtins.input = original_input

if __name__ == "__main__":
    demo_negotiation()
