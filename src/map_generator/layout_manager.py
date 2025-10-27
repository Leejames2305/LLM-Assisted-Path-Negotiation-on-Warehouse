"""
Layout Manager - Handles loading, saving, and listing warehouse layouts
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .layout_validator import LayoutValidator
from .constants import (
    PREBUILT_LAYOUTS,
    PREBUILT_LAYOUT_DIR,
    CUSTOM_LAYOUT_DIR,
    EMPTY_LAYOUT_TEMPLATE,
    LAYOUT_SCHEMA_VERSION,
)


class LayoutManager:
    """Manages layout files - load, save, list, validate"""

    def __init__(self):
        self.validator = LayoutValidator()
        self._ensure_directories()

    @staticmethod
    def _ensure_directories():
        """Ensure layout directories exist"""
        os.makedirs(PREBUILT_LAYOUT_DIR, exist_ok=True)
        os.makedirs(CUSTOM_LAYOUT_DIR, exist_ok=True)

    def list_available_layouts(self) -> Dict[str, List[str]]:
        """
        List all available layouts (prebuilt and custom)

        Returns:
            Dict with 'prebuilt' and 'custom' keys containing layout names
        """
        layouts = {'prebuilt': [], 'custom': []}

        # Get prebuilt layouts
        if os.path.exists(PREBUILT_LAYOUT_DIR):
            for file in os.listdir(PREBUILT_LAYOUT_DIR):
                if file.endswith('.json'):
                    layouts['prebuilt'].append(file.replace('.json', ''))

        # Get custom layouts
        if os.path.exists(CUSTOM_LAYOUT_DIR):
            for file in os.listdir(CUSTOM_LAYOUT_DIR):
                if file.endswith('.json'):
                    layouts['custom'].append(file.replace('.json', ''))

        return layouts

    def load_layout(self, layout_name: str, is_custom: bool = False) -> Optional[Dict]:
        """
        Load a layout from file

        Args:
            layout_name: Name of layout (without .json extension)
            is_custom: If True, load from custom dir; if False, load from prebuilt

        Returns:
            Layout dict if successful, None otherwise
        """
        if is_custom:
            filepath = os.path.join(CUSTOM_LAYOUT_DIR, f'{layout_name}.json')
        else:
            filepath = os.path.join(PREBUILT_LAYOUT_DIR, f'{layout_name}.json')

        if not os.path.exists(filepath):
            print(f"❌ Layout file not found: {filepath}")
            return None

        try:
            with open(filepath, 'r') as f:
                layout = json.load(f)

            # Validate layout
            is_valid, errors, warnings = self.validator.validate(layout)

            if not is_valid:
                print(f"❌ Layout validation failed: {layout_name}")
                print(self.validator.get_error_summary())
                return None

            if warnings:
                print(f"⚠️  Layout loaded with warnings:")
                for warning in warnings:
                    print(f"   - {warning}")

            print(f"✅ Layout loaded successfully: {layout_name}")
            return layout

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in {filepath}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading layout {layout_name}: {e}")
            return None

    def save_layout(self, layout: Dict, layout_name: str, is_custom: bool = True, overwrite: bool = False) -> bool:
        """
        Save a layout to file with validation

        Args:
            layout: Layout dictionary
            layout_name: Name to save as (without .json extension)
            is_custom: If True, save to custom dir; if False, save to prebuilt
            overwrite: If True, overwrite existing files

        Returns:
            True if successful, False otherwise
        """
        # Validate layout first
        is_valid, errors, warnings = self.validator.validate(layout)

        if not is_valid:
            print("❌ Layout validation failed before saving:")
            print(self.validator.get_error_summary())
            return False

        if warnings:
            print("⚠️  Layout has warnings:")
            for warning in warnings:
                print(f"   - {warning}")

        # Determine save path
        if is_custom:
            save_dir = CUSTOM_LAYOUT_DIR
        else:
            save_dir = PREBUILT_LAYOUT_DIR

        self._ensure_directories()
        filepath = os.path.join(save_dir, f'{layout_name}.json')

        # Check if file exists
        if os.path.exists(filepath) and not overwrite:
            print(f"⚠️  File already exists: {filepath}")
            response = input("Overwrite? (y/N): ").strip().lower()
            if response != 'y':
                print("Save cancelled.")
                return False

        try:
            with open(filepath, 'w') as f:
                json.dump(layout, f, indent=2)

            print(f"✅ Layout saved successfully: {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error saving layout: {e}")
            return False

    def delete_layout(self, layout_name: str, is_custom: bool = True) -> bool:
        """
        Delete a layout file

        Args:
            layout_name: Name of layout to delete (without .json extension)
            is_custom: If True, delete from custom dir; if False, delete from prebuilt

        Returns:
            True if successful, False otherwise
        """
        if is_custom:
            filepath = os.path.join(CUSTOM_LAYOUT_DIR, f'{layout_name}.json')
        else:
            filepath = os.path.join(PREBUILT_LAYOUT_DIR, f'{layout_name}.json')

        if not os.path.exists(filepath):
            print(f"❌ Layout file not found: {filepath}")
            return False

        try:
            response = input(f"Delete {layout_name}? (y/N): ").strip().lower()
            if response != 'y':
                print("Deletion cancelled.")
                return False

            os.remove(filepath)
            print(f"✅ Layout deleted: {layout_name}")
            return True

        except Exception as e:
            print(f"❌ Error deleting layout: {e}")
            return False

    def create_empty_layout(self, width: int = 8, height: int = 6) -> Dict:
        """
        Create an empty layout template with specified dimensions

        Args:
            width: Grid width
            height: Grid height

        Returns:
            Empty layout dictionary
        """
        layout = EMPTY_LAYOUT_TEMPLATE.copy()
        layout['dimensions'] = {'width': width, 'height': height}

        # Create empty grid
        layout['grid'] = ['#' + '.' * (width - 2) + '#' for _ in range(height)]
        layout['grid'][0] = '#' * width
        layout['grid'][-1] = '#' * width

        return layout

    def get_layout_info(self, layout_name: str, is_custom: bool = False) -> Optional[Dict]:
        """
        Get information about a layout without full loading

        Args:
            layout_name: Name of layout
            is_custom: If True, look in custom dir; if False, look in prebuilt

        Returns:
            Dict with layout metadata or None if not found
        """
        if is_custom:
            filepath = os.path.join(CUSTOM_LAYOUT_DIR, f'{layout_name}.json')
        else:
            filepath = os.path.join(PREBUILT_LAYOUT_DIR, f'{layout_name}.json')

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                layout = json.load(f)

            return {
                'name': layout.get('name', layout_name),
                'description': layout.get('description', ''),
                'width': layout.get('dimensions', {}).get('width', 0),
                'height': layout.get('dimensions', {}).get('height', 0),
                'num_agents': len(layout.get('agents', [])),
                'num_boxes': len(layout.get('boxes', [])),
                'num_targets': len(layout.get('targets', [])),
                'filepath': filepath,
            }

        except Exception as e:
            print(f"⚠️  Could not read layout info: {e}")
            return None

    def duplicate_layout(self, source_name: str, dest_name: str, is_custom: bool = True) -> bool:
        """
        Duplicate an existing layout

        Args:
            source_name: Name of layout to copy from
            dest_name: Name to save copy as
            is_custom: If True, duplicate in custom dir; if False, duplicate in prebuilt

        Returns:
            True if successful, False otherwise
        """
        layout = self.load_layout(source_name, is_custom)
        if layout is None:
            return False

        layout['name'] = dest_name
        return self.save_layout(layout, dest_name, is_custom, overwrite=False)

    def list_layout_details(self) -> str:
        """
        Get a formatted string with details of all available layouts

        Returns:
            Formatted string listing all layouts with details
        """
        layouts = self.list_available_layouts()
        output = "\n" + "=" * 70 + "\n"
        output += "AVAILABLE LAYOUTS\n"
        output += "=" * 70 + "\n"

        if layouts['prebuilt']:
            output += f"\n📦 PREBUILT LAYOUTS ({len(layouts['prebuilt'])}):\n"
            output += "-" * 70 + "\n"

            for name in sorted(layouts['prebuilt']):
                info = self.get_layout_info(name, is_custom=False)
                if info:
                    output += f"  • {info['name']:<20} ({info['width']}x{info['height']})"
                    output += f" - {info['num_agents']} agents, {info['num_boxes']} boxes, {info['num_targets']} targets\n"
                    if info['description']:
                        output += f"    Description: {info['description']}\n"

        if layouts['custom']:
            output += f"\n💾 CUSTOM LAYOUTS ({len(layouts['custom'])}):\n"
            output += "-" * 70 + "\n"

            for name in sorted(layouts['custom']):
                info = self.get_layout_info(name, is_custom=True)
                if info:
                    output += f"  • {info['name']:<20} ({info['width']}x{info['height']})"
                    output += f" - {info['num_agents']} agents, {info['num_boxes']} boxes, {info['num_targets']} targets\n"
                    if info['description']:
                        output += f"    Description: {info['description']}\n"

        if not layouts['prebuilt'] and not layouts['custom']:
            output += "❌ No layouts found. Please create one using the layout editor.\n"

        output += "=" * 70 + "\n"
        return output

    @staticmethod
    def validate_layout_file(filepath: str) -> Tuple[bool, str]:
        """
        Validate a layout file without loading it fully

        Args:
            filepath: Path to JSON layout file

        Returns:
            Tuple of (is_valid, error_message)
        """
        validator = LayoutValidator()

        if not os.path.exists(filepath):
            return False, f"File not found: {filepath}"

        try:
            with open(filepath, 'r') as f:
                layout = json.load(f)

            is_valid, errors, warnings = validator.validate(layout)

            if not is_valid:
                return False, validator.get_error_summary()

            return True, validator.get_error_summary()

        except json.JSONDecodeError as e:
            return False, f"JSON parsing error: {e}"
        except Exception as e:
            return False, f"Error validating layout: {e}"
