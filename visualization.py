#!/usr/bin/env python3
"""
Simulation Visualization Tool
============================

Visualizes HMAS-2 negotiation simulation results with interactive turn-by-turn playback.
Creates animated maps showing agent movements, conflicts, and negotiation outcomes.

Features:
- Interactive turn navigation with play/pause controls
- Agent path visualization with different styles for planned vs negotiated paths
- Conflict and negotiation highlighting
- Box pickup/delivery tracking
- Comprehensive statistics display

Usage:
    python visualization.py [simulation_log.json]
    
If no file is provided, will prompt to select from available logs.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
from datetime import datetime
import argparse

class SimulationVisualizer:
    def __init__(self, log_file_path):
        """Initialize the visualizer with simulation log data"""
        self.log_file_path = log_file_path
        self.load_simulation_data()
        self.setup_visualization()
        self.current_turn = 0
        self.is_playing = False
        self._updating_slider = False  # Flag to prevent recursive updates
        
    def load_simulation_data(self):
        """Load and parse the simulation log JSON file"""
        try:
            with open(self.log_file_path, 'r') as f:
                self.sim_data = json.load(f)
            
            print(f"✅ Loaded simulation data: {self.log_file_path}")
            print(f"📊 Scenario: {self.sim_data['scenario']['type']}")
            print(f"🔢 Total turns: {len(self.sim_data['turns'])}")
            print(f"⚔️  Negotiation turns: {self.sim_data['summary']['negotiation_turns']}")
            
        except Exception as e:
            print(f"❌ Error loading simulation data: {e}")
            sys.exit(1)
    
    def detect_walls(self):
        """Detect walls by analyzing grid data or agent movements"""
        # Initialize all cells as potentially walls
        self.walls = set()
        self.passable = set()
        
        # First, try to use grid data if available
        if 'scenario' in self.sim_data and 'grid' in self.sim_data['scenario']:
            print("🗺️  Using grid data from simulation log for accurate wall detection")
            grid = self.sim_data['scenario']['grid']
            
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    if cell == '#':  # Wall cell
                        self.walls.add((x, y))
                    elif cell == '.':  # Passable cell
                        self.passable.add((x, y))
        else:
            print("⚠️  No grid data found, falling back to movement-based wall detection")
            # Fallback to the original detection method
            self._detect_walls_from_movement()
        
        print(f"🧱 Detected {len(self.walls)} wall cells and {len(self.passable)} passable cells")
    
    def _detect_walls_from_movement(self):
        """Fallback method: detect walls by analyzing agent movements"""
        # Collect all positions where agents have been
        for turn in self.sim_data['turns']:
            agent_states = turn.get('agent_states', {})
            for agent_id, state in agent_states.items():
                if 'position' in state and state['position']:
                    pos = tuple(state['position'])
                    self.passable.add(pos)
                
                # Also check paths for more passable positions
                if 'planned_path' in state and state['planned_path']:
                    for path_pos in state['planned_path']:
                        if len(path_pos) >= 2:
                            self.passable.add(tuple(path_pos))
            
            # Check results section for additional positions
            if 'results' in turn and 'agent_states_after' in turn['results']:
                for agent_id, state in turn['results']['agent_states_after'].items():
                    if 'position' in state and state['position']:
                        pos = tuple(state['position'])
                        self.passable.add(pos)
        
        # Add initial agent and target positions
        initial_agents = self.sim_data['scenario'].get('initial_agents', {})
        for agent_id, pos in initial_agents.items():
            if pos and len(pos) >= 2:
                self.passable.add(tuple(pos))
        
        initial_targets = self.sim_data['scenario'].get('initial_targets', {})
        for target_id, pos in initial_targets.items():
            if pos and len(pos) >= 2:
                self.passable.add(tuple(pos))
        
        # For s_shaped_corridor, deduce the corridor pattern
        if self.sim_data['scenario']['type'] == 's_shaped_corridor':
            self._detect_s_corridor_walls()
        else:
            # For other scenarios, mark all unvisited positions as walls
            for x in range(self.width):
                for y in range(self.height):
                    if (x, y) not in self.passable:
                        self.walls.add((x, y))
    
    def _detect_s_corridor_walls(self):
        """Detect walls for S-shaped corridor scenario"""
        # S-shaped corridor typically has a specific pattern
        # We'll infer walls based on the passable areas and create a corridor
        
        min_x = min(pos[0] for pos in self.passable) if self.passable else 0
        max_x = max(pos[0] for pos in self.passable) if self.passable else self.width - 1
        min_y = min(pos[1] for pos in self.passable) if self.passable else 0
        max_y = max(pos[1] for pos in self.passable) if self.passable else self.height - 1
        
        # Create walls around the corridor
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.passable:
                    # Check if this position is adjacent to a passable area
                    # If it's completely isolated, it's likely a wall
                    adjacent_passable = False
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        adj_x, adj_y = x + dx, y + dy
                        if 0 <= adj_x < self.width and 0 <= adj_y < self.height:
                            if (adj_x, adj_y) in self.passable:
                                adjacent_passable = True
                                break
                    
                    # If not adjacent to passable area, or at the edges, likely a wall
                    if not adjacent_passable or x == 0 or x == self.width-1 or y == 0 or y == self.height-1:
                        self.walls.add((x, y))
    
    def setup_visualization(self):
        """Setup the matplotlib figure and axes"""
        # Extract map dimensions
        map_size = self.sim_data['scenario']['map_size']
        self.width, self.height = map_size[0], map_size[1]
        
        # Detect walls and passable areas
        self.detect_walls()
        
        # Create figure with subplots
        self.fig, (self.ax_main, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Main map subplot
        self.ax_main.set_xlim(-0.5, self.width - 0.5)
        self.ax_main.set_ylim(-0.5, self.height - 0.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.invert_yaxis()  # Invert Y-axis to match typical grid layout
        self.ax_main.set_title("HMAS-2 Simulation Visualization")
        self.ax_main.grid(True, alpha=0.3)
        
        # Stats subplot
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.ax_stats.set_title("Turn Statistics")
        
        # Add control buttons
        self.setup_controls()
        
        # Color scheme for agents
        self.agent_colors = plt.get_cmap('Set1')(np.linspace(0, 1, 10))
        
    def setup_controls(self):
        """Setup interactive controls for the visualization"""
        # Make room for controls at the bottom
        plt.subplots_adjust(bottom=0.2)
        
        # Play/Pause button
        ax_play = plt.axes((0.1, 0.05, 0.1, 0.04))
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)
        
        # Previous turn button
        ax_prev = plt.axes((0.25, 0.05, 0.1, 0.04))
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_turn)
        
        # Next turn button
        ax_next = plt.axes((0.4, 0.05, 0.1, 0.04))
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_turn)
        
        # Turn slider
        ax_slider = plt.axes((0.55, 0.05, 0.3, 0.04))
        self.slider_turn = Slider(ax_slider, 'Turn', 0, max(0, len(self.sim_data['turns']) - 1), 
                                  valinit=0, valfmt='%d')
        self.slider_turn.on_changed(self.update_turn_from_slider)
        
    def toggle_play(self, event):
        """Toggle play/pause for animation"""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
        
        if self.is_playing:
            self.start_animation()
        else:
            self.stop_animation()
    
    def prev_turn(self, event):
        """Go to previous turn"""
        if self.current_turn > 0:
            self.current_turn -= 1
            self.update_visualization()
    
    def next_turn(self, event):
        """Go to next turn"""
        max_turn = len(self.sim_data['turns']) - 1
        if self.current_turn < max_turn:
            self.current_turn += 1
            self.update_visualization()
    
    def update_turn_from_slider(self, val):
        """Update turn from slider value"""
        if not self._updating_slider:  # Prevent recursive updates
            self.current_turn = int(self.slider_turn.val)
            self.update_visualization()
    
    def start_animation(self):
        """Start automatic turn progression"""
        def animate(frame):
            if self.is_playing and self.current_turn < len(self.sim_data['turns']) - 1:
                self.current_turn += 1
                self.update_visualization()
                return []
            else:
                self.is_playing = False
                self.btn_play.label.set_text('Play')
                return []
        
        self.animation = FuncAnimation(self.fig, animate, interval=2000, blit=False)
        
    def stop_animation(self):
        """Stop automatic turn progression"""
        if hasattr(self, 'animation') and self.animation:
            self.animation.pause()
    
    def update_visualization(self):
        """Update the visualization for the current turn"""
        try:
            # Clear the main plot
            self.ax_main.clear()
            self.ax_main.set_xlim(-0.5, self.width - 0.5)
            self.ax_main.set_ylim(-0.5, self.height - 0.5)
            self.ax_main.set_aspect('equal')
            self.ax_main.invert_yaxis()
            self.ax_main.grid(True, alpha=0.3)
            
            # Get current turn data
            if self.current_turn < len(self.sim_data['turns']):
                turn_data = self.sim_data['turns'][self.current_turn]
                
                # Update title with turn info
                title = f"Turn {turn_data['turn']} - {turn_data['type'].title()}"
                if turn_data.get('negotiation_occurred', False):
                    title += " (Negotiation)"
                self.ax_main.set_title(title)
                
                # Draw map elements
                self.draw_map_grid()
                self.draw_walls()
                self.draw_targets(turn_data)
                self.draw_boxes(turn_data)
                self.draw_agents(turn_data)
                self.draw_agent_paths(turn_data)
                self.draw_legend()
                
                # Update statistics
                self.update_statistics(turn_data)
                
                # Update slider (with flag to prevent recursion)
                self._updating_slider = True
                self.slider_turn.set_val(self.current_turn)
                self._updating_slider = False
            
            self.fig.canvas.draw_idle()  # Use draw_idle instead of plt.draw()
            
        except Exception as e:
            print(f"❌ Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_map_grid(self):
        """Draw the basic map grid"""
        # Draw grid lines
        for i in range(self.width + 1):
            self.ax_main.axvline(i - 0.5, color='gray', alpha=0.3)
        for i in range(self.height + 1):
            self.ax_main.axhline(i - 0.5, color='gray', alpha=0.3)
    
    def draw_walls(self):
        """Draw wall cells"""
        for wall_x, wall_y in self.walls:
            wall = patches.Rectangle((wall_x-0.5, wall_y-0.5), 1.0, 1.0, 
                                   linewidth=1, edgecolor='black', 
                                   facecolor='darkgray', alpha=0.8)
            self.ax_main.add_patch(wall)
        
        # Highlight passable corridor areas lightly
        for pass_x, pass_y in self.passable:
            pass_area = patches.Rectangle((pass_x-0.5, pass_y-0.5), 1.0, 1.0, 
                                        linewidth=0, facecolor='lightblue', alpha=0.1)
            self.ax_main.add_patch(pass_area)
    
    def draw_targets(self, turn_data):
        """Draw target positions"""
        if 'map_state' in turn_data:
            targets = turn_data['map_state'].get('targets', {})
            for target_id, pos in targets.items():
                if pos and len(pos) >= 2:  # Ensure position is valid
                    x, y = pos[0], pos[1]
                    target = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                             linewidth=2, edgecolor='red', 
                                             facecolor='lightcoral', alpha=0.6)
                    self.ax_main.add_patch(target)
                    self.ax_main.text(x, y, f'T{target_id}', ha='center', va='center', 
                                    fontweight='bold', fontsize=8)
    
    def draw_boxes(self, turn_data):
        """Draw box positions"""
        if 'map_state' in turn_data:
            boxes = turn_data['map_state'].get('boxes', {})
            for box_id, pos in boxes.items():
                if pos and len(pos) >= 2:  # Ensure position is valid
                    x, y = pos[0], pos[1]
                    box = patches.Rectangle((x-0.3, y-0.3), 0.6, 0.6, 
                                          linewidth=2, edgecolor='brown', 
                                          facecolor='burlywood', alpha=0.8)
                    self.ax_main.add_patch(box)
                    self.ax_main.text(x, y, f'B{box_id}', ha='center', va='center', 
                                    fontweight='bold', fontsize=7)
    
    def draw_agents(self, turn_data):
        """Draw agent positions with status indicators"""
        agent_states = turn_data.get('agent_states', {})
        
        for i, (agent_id, state) in enumerate(agent_states.items()):
            if 'position' in state and state['position'] and len(state['position']) >= 2:
                x, y = state['position'][0], state['position'][1]
                color = self.agent_colors[i % len(self.agent_colors)]
                
                # Agent circle
                circle = patches.Circle((x, y), 0.25, facecolor=color, 
                                      edgecolor='black', linewidth=2, alpha=0.8)
                self.ax_main.add_patch(circle)
                
                # Agent ID label
                self.ax_main.text(x, y, str(agent_id), ha='center', va='center', 
                                fontweight='bold', fontsize=9, color='white')
                
                # Status indicators
                status_text = []
                if state.get('carrying_box'):
                    status_text.append(f"[B{state.get('box_id', '?')}]")
                if state.get('is_waiting'):
                    status_text.append(f"Wait:{state.get('wait_turns_remaining', 0)}")
                if state.get('has_negotiated_path'):
                    status_text.append("NEG")
                
                if status_text:
                    self.ax_main.text(x, y-0.5, ' '.join(status_text), ha='center', va='top', 
                                    fontsize=8, bbox=dict(boxstyle="round,pad=0.1", 
                                    facecolor='white', alpha=0.8))
    
    def draw_agent_paths(self, turn_data):
        """Draw agent planned paths"""
        agent_states = turn_data.get('agent_states', {})
        
        for i, (agent_id, state) in enumerate(agent_states.items()):
            if 'planned_path' in state and state['planned_path']:
                path = state['planned_path']
                color = self.agent_colors[i % len(self.agent_colors)]
                
                # Draw path line
                path_x = [pos[0] for pos in path]
                path_y = [pos[1] for pos in path]
                
                line_style = '--' if state.get('has_negotiated_path') else ':'
                alpha = 0.8 if state.get('has_negotiated_path') else 0.4
                
                self.ax_main.plot(path_x, path_y, color=color, linewidth=2, 
                                linestyle=line_style, alpha=alpha, 
                                label=f"Agent {agent_id} path")
                
                # Mark path direction with arrows
                if len(path) > 1:
                    for j in range(min(3, len(path)-1)):  # Show first 3 steps
                        dx = path[j+1][0] - path[j][0]
                        dy = path[j+1][1] - path[j][1]
                        self.ax_main.arrow(path[j][0], path[j][1], dx*0.3, dy*0.3, 
                                         head_width=0.1, head_length=0.1, 
                                         fc=color, ec=color, alpha=alpha)
    
    def draw_legend(self):
        """Draw a legend explaining the visualization elements"""
        legend_elements = []
        
        # Add legend items
        legend_elements.append(patches.Patch(color='darkgray', alpha=0.8, label='Walls'))
        legend_elements.append(patches.Patch(color='lightblue', alpha=0.1, label='Passable Areas'))
        legend_elements.append(patches.Patch(color='lightcoral', alpha=0.6, label='Targets'))
        legend_elements.append(patches.Patch(color='burlywood', alpha=0.8, label='Boxes'))
        
        # Add agent colors
        for i, (agent_id, _) in enumerate(self.sim_data['scenario']['initial_agents'].items()):
            if i < len(self.agent_colors):
                color = self.agent_colors[i]
                legend_elements.append(patches.Patch(color=color, alpha=0.8, label=f'Agent {agent_id}'))
        
        self.ax_main.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                           fontsize=8, framealpha=0.9)
    
    def update_statistics(self, turn_data):
        """Update the statistics panel"""
        self.ax_stats.clear()
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        
        stats_text = []
        stats_text.append(f"Turn {turn_data['turn']}")
        stats_text.append(f"Type: {turn_data['type'].title()}")
        stats_text.append("")
        
        # Agent statistics
        agent_states = turn_data.get('agent_states', {})
        stats_text.append("Agent Status:")
        for agent_id, state in agent_states.items():
            status = f"Agent {agent_id}: "
            if state.get('carrying_box'):
                status += f"Carrying Box {state.get('box_id')}"
            elif state.get('is_waiting'):
                status += f"Waiting ({state.get('wait_turns_remaining')} turns)"
            else:
                status += "Moving"
            
            if state.get('has_negotiated_path'):
                status += " (Negotiated)"
            
            stats_text.append(status)
        
        stats_text.append("")
        
        # Conflict information
        if turn_data.get('negotiation_occurred'):
            stats_text.append("NEGOTIATION Turn")
            stats_text.append("Conflicts resolved via HMAS-2")
        else:
            stats_text.append("ROUTINE Turn")
            stats_text.append("No conflicts detected")
        
        # Simulation summary
        if 'summary' in self.sim_data:
            summary = self.sim_data['summary']
            stats_text.append("")
            stats_text.append("Simulation Summary:")
            stats_text.append(f"Total Turns: {summary.get('total_turns', 'N/A')}")
            stats_text.append(f"Negotiations: {summary.get('total_negotiations', 'N/A')}")
            stats_text.append(f"Conflicts: {summary.get('total_conflicts', 'N/A')}")
        
        # Display statistics text
        y_pos = 0.95
        for line in stats_text:
            self.ax_stats.text(0.05, y_pos, line, transform=self.ax_stats.transAxes, 
                             fontsize=10, verticalalignment='top')
            y_pos -= 0.06
    
    def show(self):
        """Show the visualization"""
        self.update_visualization()
        plt.show()

def find_simulation_logs():
    """Find all simulation log files in the logs directory"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return []
    
    log_files = []
    for filename in os.listdir(logs_dir):
        if filename.startswith("simulation_log_") and filename.endswith(".json"):
            log_files.append(os.path.join(logs_dir, filename))
    
    return sorted(log_files, reverse=True)  # Most recent first

def select_log_file():
    """Interactive selection of log file"""
    log_files = find_simulation_logs()
    
    if not log_files:
        print("❌ No simulation log files found in logs/ directory")
        print("   Run a simulation first with test_negotiation.py")
        sys.exit(1)
    
    print("📁 Available simulation logs:")
    for i, log_file in enumerate(log_files, 1):
        # Extract timestamp from filename
        basename = os.path.basename(log_file)
        timestamp_str = basename.replace("simulation_log_", "").replace(".json", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp_str
        
        print(f"   {i}. {basename} ({formatted_time})")
    
    while True:
        try:
            choice = input("\nSelect log file (number): ").strip()
            if choice.lower() in ['q', 'quit', 'exit']:
                sys.exit(0)
            
            index = int(choice) - 1
            if 0 <= index < len(log_files):
                return log_files[index]
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(log_files)}")
        except ValueError:
            print("❌ Please enter a valid number")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Visualize HMAS-2 simulation results")
    parser.add_argument("log_file", nargs='?', help="Path to simulation log JSON file")
    parser.add_argument("--list", action="store_true", help="List available log files")
    
    args = parser.parse_args()
    
    if args.list:
        log_files = find_simulation_logs()
        if log_files:
            print("📁 Available simulation logs:")
            for log_file in log_files:
                print(f"   {log_file}")
        else:
            print("❌ No simulation log files found")
        return
    
    # Determine log file to use
    if args.log_file:
        if not os.path.exists(args.log_file):
            print(f"❌ Log file not found: {args.log_file}")
            sys.exit(1)
        log_file = args.log_file
    else:
        log_file = select_log_file()
    
    print(f"🎬 Starting visualization for: {log_file}")
    
    # Create and show visualization
    try:
        visualizer = SimulationVisualizer(log_file)
        visualizer.show()
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()