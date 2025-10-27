#!/usr/bin/env python3
"""
Interactive Layout Editor - Terminal-based UI for creating/editing warehouse layouts
"""

import os
import sys
import json
from typing import Dict, Optional, Set, List, Tuple, cast

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.map_generator.layout_manager import LayoutManager
from src.map_generator.layout_validator import LayoutValidator
from src.map_generator.constants import EMPTY_LAYOUT_TEMPLATE, CellType


class LayoutEditor:
    """Interactive terminal editor for warehouse layouts"""

    def __init__(self):
        self.manager = LayoutManager()
        self.validator = LayoutValidator()
        self.current_layout: Optional[Dict] = None
        self.current_filename: Optional[str] = None
        self.is_custom = False
        self.saved = True

    def run(self):
        """Main editor loop"""
        self._print_header()

        while True:
            print("\n" + "=" * 60)
            print("MAIN MENU")
            print("=" * 60)
            print("1. Create new layout")
            print("2. Load existing layout")
            print("3. View layouts")
            print("4. Delete layout")
            print("5. Exit")
            print()

            choice = input("Select option (1-5): ").strip()

            if choice == "1":
                self._create_new_layout()
            elif choice == "2":
                self._load_layout()
            elif choice == "3":
                self._view_layouts()
            elif choice == "4":
                self._delete_layout()
            elif choice == "5":
                if self.current_layout and not self.saved:
                    response = input("Unsaved changes. Exit anyway? (y/N): ").strip().lower()
                    if response != 'y':
                        continue
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")

    def _print_header(self):
        """Print ASCII header"""
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║" + " WAREHOUSE LAYOUT EDITOR ".center(58) + "║")
        print("║" + " Interactive Terminal UI for Custom Layouts ".center(58) + "║")
        print("╚" + "═" * 58 + "╝")
        print()

    def _create_new_layout(self):
        """Create a new layout from scratch"""
        print("\n" + "=" * 60)
        print("CREATE NEW LAYOUT")
        print("=" * 60)

        try:
            width = int(input("Enter grid width (5-50, default 8): ") or "8")
            height = int(input("Enter grid height (5-50, default 6): ") or "6")

            if not (5 <= width <= 50 and 5 <= height <= 50):
                print("❌ Invalid dimensions")
                return

            self.current_layout = self.manager.create_empty_layout(width, height)
            self.current_filename = None
            self.saved = False

            print(f"\n✅ Layout created: {width}x{height}")
            self._edit_layout()

        except ValueError:
            print("❌ Invalid input. Please enter numbers.")

    def _load_layout(self):
        """Load an existing layout for editing"""
        print("\n" + "=" * 60)
        print("LOAD LAYOUT")
        print("=" * 60)

        layouts = self.manager.list_available_layouts()

        if not layouts['prebuilt'] and not layouts['custom']:
            print("❌ No layouts available.")
            return

        print("\nAvailable layouts:")
        print("-" * 40)
        print("PREBUILT:")
        for i, name in enumerate(sorted(layouts['prebuilt']), 1):
            print(f"  {i:2}. {name}")

        print("\nCUSTOM:")
        offset = len(layouts['prebuilt'])
        for i, name in enumerate(sorted(layouts['custom']), 1):
            print(f"  {i + offset:2}. {name}")

        try:
            choice = int(input("\nSelect layout number: "))
            all_layouts = sorted(layouts['prebuilt']) + sorted(layouts['custom'])

            if 1 <= choice <= len(all_layouts):
                layout_name = all_layouts[choice - 1]
                is_custom = choice > len(layouts['prebuilt'])

                loaded = self.manager.load_layout(layout_name, is_custom)
                if loaded:
                    self.current_layout = loaded
                    self.current_filename = layout_name
                    self.is_custom = is_custom
                    self.saved = True
                    self._edit_layout()
                else:
                    print("❌ Failed to load layout")
            else:
                print("❌ Invalid selection")

        except ValueError:
            print("❌ Invalid input")

    def _view_layouts(self):
        """View details of all available layouts"""
        print(self.manager.list_layout_details())

    def _delete_layout(self):
        """Delete a layout"""
        print("\n" + "=" * 60)
        print("DELETE LAYOUT")
        print("=" * 60)

        layouts = self.manager.list_available_layouts()

        if not layouts['custom']:
            print("❌ No custom layouts to delete (cannot delete prebuilt).")
            return

        print("Custom layouts:")
        for i, name in enumerate(sorted(layouts['custom']), 1):
            print(f"  {i}. {name}")

        try:
            choice = int(input("\nSelect layout to delete: "))
            custom_list = sorted(layouts['custom'])

            if 1 <= choice <= len(custom_list):
                layout_name = custom_list[choice - 1]
                self.manager.delete_layout(layout_name, is_custom=True)
            else:
                print("❌ Invalid selection")

        except ValueError:
            print("❌ Invalid input")

    def _edit_layout(self):
        """Main editing loop for a layout"""
        while True:
            self._display_grid()
            print("\n" + "=" * 60)
            print("EDITING COMMANDS")
            print("=" * 60)
            print("w <x> <y>           - Toggle wall at position")
            print("a <id> <x> <y>      - Place/move agent")
            print("b <id> <x> <y>      - Place/move box")
            print("t <id> <x> <y>      - Place/move target")
            print("goal <agent> <tgt>  - Set agent goal to target")
            print("rand a <count>      - Random agent placement")
            print("rand b <count>      - Random box placement")
            print("rand t <count>      - Random target placement")
            print("clear               - Clear all entities (keep walls)")
            print("info                - Show layout info")
            print("validate            - Check for errors")
            print("save                - Save layout")
            print("back                - Return to main menu")
            print()

            command = input("Enter command: ").strip().lower().split()

            if not command:
                continue

            action = command[0]

            try:
                if action == "w" and len(command) == 3:
                    x, y = int(command[1]), int(command[2])
                    self._toggle_wall(x, y)
                    self.saved = False

                elif action == "a" and len(command) == 4:
                    agent_id, x, y = int(command[1]), int(command[2]), int(command[3])
                    self._place_agent(agent_id, x, y)
                    self.saved = False

                elif action == "b" and len(command) == 4:
                    box_id, x, y = int(command[1]), int(command[2]), int(command[3])
                    self._place_box(box_id, x, y)
                    self.saved = False

                elif action == "t" and len(command) == 4:
                    target_id, x, y = int(command[1]), int(command[2]), int(command[3])
                    self._place_target(target_id, x, y)
                    self.saved = False

                elif action == "goal" and len(command) == 3:
                    agent_id, target_id = int(command[1]), int(command[2])
                    self._set_goal(agent_id, target_id)
                    self.saved = False

                elif action == "rand" and len(command) >= 3:
                    mode = command[1]
                    count = int(command[2])
                    if mode in ['a', 'b', 't']:
                        self._random_placement(mode, count)
                        self.saved = False
                    else:
                        print("❌ Invalid mode. Use 'a', 'b', or 't'")

                elif action == "clear":
                    self._clear_entities()
                    self.saved = False

                elif action == "info":
                    self._show_info()

                elif action == "validate":
                    self._validate_layout()

                elif action == "save":
                    self._save_layout()

                elif action == "back":
                    if not self.saved:
                        response = input("Unsaved changes. Go back anyway? (y/N): ").strip().lower()
                        if response == 'y':
                            break
                    else:
                        break

                else:
                    print("❌ Invalid command or parameters")

            except (ValueError, IndexError):
                print("❌ Invalid parameters")
            except Exception as e:
                print(f"❌ Error: {e}")

    def _display_grid(self):
        """Display the current grid"""
        if not self.current_layout:
            print("❌ No layout loaded")
            return

        layout = cast(Dict, self.current_layout)
        grid = layout.get('grid', [])
        width = layout.get('dimensions', {}).get('width', 0)
        height = layout.get('dimensions', {}).get('height', 0)
        agents = layout.get('agents', [])
        boxes = layout.get('boxes', [])
        targets = layout.get('targets', [])

        print("\n🗺️  CURRENT LAYOUT:\n")

        # Print x-axis labels (centered in 3-char columns)
        print("    ", end="")  # Space for y-axis labels
        for x in range(width):
            print(f"{x:^3}", end="")
        print()

        # Create entity position map for quick lookup
        agent_positions = {(a['x'], a['y']): a['id'] for a in agents}
        box_positions = {(b['x'], b['y']): b['id'] for b in boxes}
        target_positions = {(t['x'], t['y']): t['id'] for t in targets}

        # Print grid with y-axis labels
        for y in range(height):
            print(f"{y:2}: ", end="")
            for x in range(width):
                cell = grid[y][x]
                
                # Check if there's an entity at this position
                if (x, y) in agent_positions:
                    symbol = f"A{agent_positions[(x, y)]}"
                elif (x, y) in box_positions:
                    symbol = f"B{box_positions[(x, y)]}"
                elif (x, y) in target_positions:
                    symbol = f"T{target_positions[(x, y)]}"
                elif cell == '#':
                    symbol = "##"
                else:  # cell == '.'
                    symbol = "·"
                
                # Print with consistent 3-char width, centered
                print(f"{symbol:^3}", end="")
            print()

        # Print legend
        print("\nLegend: # = Wall | · = Empty | A = Agent | B = Box | T = Target")

    def _toggle_wall(self, x: int, y: int):
        """Toggle a wall at position"""
        if not self.current_layout:
            return
        
        layout = cast(Dict, self.current_layout)
        grid = layout['grid']
        width = layout['dimensions']['width']
        height = layout['dimensions']['height']

        if not (0 <= x < width and 0 <= y < height):
            print(f"❌ Position out of bounds: ({x}, {y})")
            return

        # Convert grid to list of lists for editing
        grid_list = [list(row) for row in grid]

        current = grid_list[y][x]
        if current == '#':
            grid_list[y][x] = '.'
            print(f"✅ Removed wall at ({x}, {y})")
        else:
            grid_list[y][x] = '#'
            print(f"✅ Placed wall at ({x}, {y})")

        # Update grid
        layout = cast(Dict, self.current_layout)
        layout['grid'] = [''.join(row) for row in grid_list]

    def _place_agent(self, agent_id: int, x: int, y: int):
        """Place or move an agent"""
        if not self._is_valid_entity_position(x, y):
            print(f"❌ Cannot place agent at ({x}, {y})")
            return

        layout = cast(Dict, self.current_layout)
        agents = layout['agents']

        # Remove existing agent with same ID
        agents[:] = [a for a in agents if a['id'] != agent_id]

        # Add new agent
        agents.append({'id': agent_id, 'x': x, 'y': y})
        print(f"✅ Placed agent {agent_id} at ({x}, {y})")

    def _place_box(self, box_id: int, x: int, y: int):
        """Place or move a box"""
        if not self._is_valid_entity_position(x, y):
            print(f"❌ Cannot place box at ({x}, {y})")
            return

        layout = cast(Dict, self.current_layout)
        boxes = layout['boxes']

        # Remove existing box with same ID
        boxes[:] = [b for b in boxes if b['id'] != box_id]

        # Add new box
        boxes.append({'id': box_id, 'x': x, 'y': y})
        print(f"✅ Placed box {box_id} at ({x}, {y})")

    def _place_target(self, target_id: int, x: int, y: int):
        """Place or move a target"""
        if not self._is_valid_entity_position(x, y):
            print(f"❌ Cannot place target at ({x}, {y})")
            return

        layout = cast(Dict, self.current_layout)
        targets = layout['targets']

        # Remove existing target with same ID
        targets[:] = [t for t in targets if t['id'] != target_id]

        # Add new target
        targets.append({'id': target_id, 'x': x, 'y': y})
        print(f"✅ Placed target {target_id} at ({x}, {y})")

    def _is_valid_entity_position(self, x: int, y: int) -> bool:
        """Check if position is valid for entity placement"""
        if not self.current_layout:
            return False
            
        layout = cast(Dict, self.current_layout)
        grid = layout['grid']
        width = layout['dimensions']['width']
        height = layout['dimensions']['height']

        if not (0 <= x < width and 0 <= y < height):
            print(f"Position out of bounds: ({x}, {y})")
            return False

        cell = grid[y][x]
        if cell == '#':
            print(f"Position ({x}, {y}) is a wall")
            return False

        return True

    def _set_goal(self, agent_id: int, target_id: int):
        """Set an agent's goal to a target"""
        if not self.current_layout:
            return
            
        layout = cast(Dict, self.current_layout)
        agents = {a['id']: a for a in layout['agents']}
        targets = {t['id']: t for t in layout['targets']}

        if agent_id not in agents:
            print(f"❌ Agent {agent_id} not found")
            return

        if target_id not in targets:
            print(f"❌ Target {target_id} not found")
            return

        goals = layout['agent_goals']
        goals[str(agent_id)] = target_id
        print(f"✅ Set agent {agent_id} goal to target {target_id}")

    def _random_placement(self, mode: str, count: int):
        """Randomly place entities"""
        if not self.current_layout:
            return
            
        import random

        layout = cast(Dict, self.current_layout)
        grid = layout['grid']
        width = layout['dimensions']['width']
        height = layout['dimensions']['height']

        valid_positions = []
        for y in range(height):
            for x in range(width):
                if grid[y][x] == '.':
                    valid_positions.append((x, y))

        if len(valid_positions) < count:
            print(f"❌ Not enough empty spaces. Available: {len(valid_positions)}, Requested: {count}")
            return

        selected = random.sample(valid_positions, count)

        if mode == 'a':
            layout['agents'] = []
            for i, (x, y) in enumerate(selected):
                layout['agents'].append({'id': i, 'x': x, 'y': y})
            print(f"✅ Randomly placed {count} agents")

        elif mode == 'b':
            layout['boxes'] = []
            for i, (x, y) in enumerate(selected):
                layout['boxes'].append({'id': i, 'x': x, 'y': y})
            print(f"✅ Randomly placed {count} boxes")

        elif mode == 't':
            layout['targets'] = []
            for i, (x, y) in enumerate(selected):
                layout['targets'].append({'id': i, 'x': x, 'y': y})
            print(f"✅ Randomly placed {count} targets")

    def _clear_entities(self):
        """Clear all entities (agents, boxes, targets) but keep walls"""
        if not self.current_layout:
            return
            
        response = input("Clear all entities? (y/N): ").strip().lower()
        if response == 'y':
            layout = cast(Dict, self.current_layout)
            layout['agents'] = []
            layout['boxes'] = []
            layout['targets'] = []
            layout['agent_goals'] = {}
            print("✅ Entities cleared")

    def _show_info(self):
        """Show layout information"""
        if not self.current_layout:
            print("❌ No layout loaded")
            return
            
        layout = cast(Dict, self.current_layout)
        dims = layout['dimensions']
        print("\n" + "-" * 60)
        print("LAYOUT INFORMATION")
        print("-" * 60)
        print(f"Name:          {layout.get('name', 'Unnamed')}")
        print(f"Description:   {layout.get('description', 'No description')}")
        print(f"Dimensions:    {dims['width']}x{dims['height']}")
        print(f"Agents:        {len(layout['agents'])}")
        print(f"Boxes:         {len(layout['boxes'])}")
        print(f"Targets:       {len(layout['targets'])}")
        print(f"Agent Goals:   {len(layout['agent_goals'])}")
        print("-" * 60)

    def _validate_layout(self):
        """Validate the current layout"""
        if not self.current_layout:
            print("❌ No layout loaded")
            return
            
        print("\n🔍 Validating layout...")
        layout = cast(Dict, self.current_layout)
        is_valid, errors, warnings = self.validator.validate(layout)

        if is_valid:
            print("✅ " + self.validator.get_error_summary())
        else:
            print("❌ " + self.validator.get_error_summary())

    def _save_layout(self):
        """Save the current layout"""
        if not self.current_layout:
            print("❌ No layout to save")
            return
            
        print("\n" + "=" * 60)
        print("SAVE LAYOUT")
        print("=" * 60)

        layout = cast(Dict, self.current_layout)
        
        # Get layout name
        default_name = self.current_filename or "my_layout"
        name = input(f"Enter layout name (default: {default_name}): ").strip()
        if not name:
            name = default_name

        # Get description
        description = input("Enter description (optional): ").strip()
        if description:
            layout['description'] = description

        layout['name'] = name

        # Choose location
        print("\nSave to:")
        print("1. Custom layouts (my_layouts)")
        print("2. Prebuilt layouts (shared)")

        choice = input("Select (1-2, default 1): ").strip() or "1"

        if choice == "1":
            is_custom = True
        elif choice == "2":
            response = input("Save to prebuilt? This affects all users. (y/N): ").strip().lower()
            is_custom = response != 'y'
        else:
            print("❌ Invalid choice")
            return

        # Save
        if self.manager.save_layout(layout, name, is_custom):
            self.current_filename = name
            self.saved = True
            print("✅ Layout saved successfully!")
        else:
            print("❌ Failed to save layout")


def main():
    """Main entry point"""
    try:
        editor = LayoutEditor()
        editor.run()
    except KeyboardInterrupt:
        print("\n\n👋 Editor closed.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
