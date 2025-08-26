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
    
    # Get simulation parameters
    print(f"\n{Fore.CYAN}Simulation Configuration:{Style.RESET_ALL}")
    
    try:
        num_agents = int(input("Number of agents (2-4, default 2): ") or "2")
        num_agents = max(2, min(num_agents, 4))
    except ValueError:
        num_agents = 2
    
    print(f"Using {num_agents} agents")
    
    # Create and run simulation
    try:
        game_engine = GameEngine(
            width=int(os.getenv('MAP_WIDTH', 8)),
            height=int(os.getenv('MAP_HEIGHT', 6)),
            num_agents=num_agents
        )
        
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
