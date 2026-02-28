"""
Main entry point for LLM-Assisted Multi-Robot Navigation Negotiation
"""

import os
import sys
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simulation.game_engine import GameEngine
from src.map_generator import WarehouseMap
from src.map_generator.layout_selector import get_layout_for_game

# Initialize colorama and load environment
init(autoreset=True)
load_dotenv()

def main():
    """Main function to run the simulation"""
    print(f"{Fore.CYAN}=" * 60)
    print(f"{Fore.CYAN}🤖 LLM-Assisted Multi-Robot Navigation Negotiation 🤖")
    print(f"{Fore.CYAN}=" * 60)
    print(f"{Style.RESET_ALL}")
    
    # Check if API key is configured
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print(f"{Fore.RED}⚠️  WARNING: OpenRouter API key not configured!{Style.RESET_ALL}")
        print(f"Please set OPENROUTER_API_KEY in your .env file")
        print(f"Copy .env.example to .env and add your API key")
        
        # Ask if user wants to continue in demo mode
        choice = input(f"\nContinue without LLM features? (y/N): ").strip().lower()
        if choice != 'y':
            print("Exiting. Please configure your API key first.")
            return
        
        print(f"{Fore.YELLOW}Running in demo mode (LLM features will use fallbacks){Style.RESET_ALL}")
    
    # Get layout selection
    print(f"\n{Fore.CYAN}Loading Warehouse Layout...{Style.RESET_ALL}")
    layout = get_layout_for_game(allow_selection=True)
    
    if layout is None:
        print(f"{Fore.RED}No layout selected. Exiting.{Style.RESET_ALL}")
        return
    
    # Create simulation with loaded layout
    try:
        layout_dims = layout['dimensions']
        num_agents = len(layout['agents'])
        
        print(f"\n{Fore.CYAN}Creating Game Engine...{Style.RESET_ALL}")
        game_engine = GameEngine(
            width=layout_dims['width'],
            height=layout_dims['height'],
            num_agents=num_agents
        )
        
        # Load the layout into the warehouse map
        game_engine.warehouse_map = WarehouseMap.from_layout(layout)
        
        # Initialize agents from layout
        from src.agents import RobotAgent  # type: ignore
        game_engine.agents = {}
        for agent in layout['agents']:
            agent_id = agent['id']
            position = (agent['x'], agent['y'])
            robot = RobotAgent(agent_id, position)
            game_engine.agents[agent_id] = robot
        
        print(f"✅ Layout loaded: {layout.get('name', 'Untitled')}")
        print(f"   Dimensions: {layout_dims['width']}x{layout_dims['height']}")
        print(f"   Agents: {len(layout['agents'])}, Boxes: {len(layout['boxes'])}, Targets: {len(layout['targets'])}")
        
        # Select simulation mode
        print(f"\n{Fore.CYAN}Select simulation mode:{Style.RESET_ALL}")
        print(f"  1. Turn-based (default) — step-by-step, terminal output")
        print(f"  2. Async                — parallel moves, live matplotlib window")
        print(f"  3. Lifelong             — continuous tasks, turn-based")
        mode_input = input(f"\n{Fore.CYAN}Mode (1/2/3, default 1): {Style.RESET_ALL}").strip()
        if mode_input == '2':
            game_engine.simulation_mode = 'async'
            print(f"{Fore.GREEN}✅ Mode: Async (parallel execution){Style.RESET_ALL}")
        elif mode_input == '3':
            game_engine.simulation_mode = 'lifelong'
            print(f"{Fore.GREEN}✅ Mode: Lifelong (continuous tasks){Style.RESET_ALL}")
        else:
            game_engine.simulation_mode = 'turn_based'
            print(f"{Fore.GREEN}✅ Mode: Turn-based{Style.RESET_ALL}")

        game_engine.run_interactive_simulation()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Simulation interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error running simulation: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{Fore.CYAN}Thank you for using the Multi-Robot Navigation System!{Style.RESET_ALL}")

def print_system_info():
    """Print system information and requirements"""
    print(f"{Fore.CYAN}System Information:{Style.RESET_ALL}")
    print(f"Python: {sys.version}")
    
    # Check for required packages
    required_packages = ['requests', 'python-dotenv', 'numpy', 'colorama']
    
    for package in required_packages:
        try:
            if package == 'python-dotenv':
                __import__('dotenv')
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Please install: pip install {package}")
    
    print()

if __name__ == "__main__":
    print_system_info()
    main()
