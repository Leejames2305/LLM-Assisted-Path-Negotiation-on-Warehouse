"""
Layout Selection Helper - Interactive UI for choosing layouts in games
"""

from typing import Optional, Dict

from .layout_manager import LayoutManager


def select_layout_interactive() -> Optional[Dict]:
    """
    Interactive layout selection menu for game startup
    
    Returns:
        Layout dict if successful, None if cancelled
    """
    manager = LayoutManager()
    layouts = manager.list_available_layouts()

    total_layouts = len(layouts['prebuilt']) + len(layouts['custom'])

    if total_layouts == 0:
        print("❌ No layouts available!")
        print("🛠️  Use the layout editor to create one:")
        print("   python -m src.tools.layout_editor")
        return None

    print("\n" + "=" * 60)
    print("SELECT WAREHOUSE LAYOUT")
    print("=" * 60)

    all_names = []
    options_text = []

    # Add prebuilt layouts
    if layouts['prebuilt']:
        print("\n📦 PREBUILT LAYOUTS:")
        for i, name in enumerate(sorted(layouts['prebuilt']), 1):
            all_names.append((name, False))
            info = manager.get_layout_info(name, is_custom=False)
            if info:
                dims = f"({info['width']}x{info['height']})"
                ents = f"{info['num_agents']}a/{info['num_boxes']}b/{info['num_targets']}t"
                print(f"  {i:2}. {info['name']:<20} {dims:<10} {ents}")
            else:
                print(f"  {i:2}. {name}")

    # Add custom layouts
    if layouts['custom']:
        offset = len(layouts['prebuilt'])
        print("\n💾 CUSTOM LAYOUTS:")
        for i, name in enumerate(sorted(layouts['custom']), 1):
            all_names.append((name, True))
            info = manager.get_layout_info(name, is_custom=True)
            if info:
                dims = f"({info['width']}x{info['height']})"
                ents = f"{info['num_agents']}a/{info['num_boxes']}b/{info['num_targets']}t"
                print(f"  {i + offset:2}. {info['name']:<20} {dims:<10} {ents}")
            else:
                print(f"  {i + offset:2}. {name}")

    print("\n(Enter 'e' to edit/create layouts)")
    print()

    try:
        choice = input("Select layout (1-{}) or 'e' for editor: ".format(total_layouts)).strip().lower()

        if choice == 'e':
            print("\n🛠️  Use the layout editor to manage layouts:")
            print("   python -m src.tools.layout_editor")
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_names):
                layout_name, is_custom = all_names[idx]
                layout = manager.load_layout(layout_name, is_custom)
                if layout:
                    print()
                    return layout
                else:
                    print("❌ Failed to load layout")
                    return None
            else:
                print("❌ Invalid selection")
                return None
        except ValueError:
            print("❌ Please enter a number or 'e'")
            return None

    except KeyboardInterrupt:
        print("\n👋 Cancelled")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def get_layout_for_game(allow_selection: bool = True) -> Optional[Dict]:
    """
    Get a layout for starting a game
    
    Args:
        allow_selection: If True, allow user to select layout interactively
        
    Returns:
        Layout dict if successful, None otherwise
    """
    if allow_selection:
        return select_layout_interactive()
    else:
        # Fallback to default/first available layout
        manager = LayoutManager()
        layouts = manager.list_available_layouts()

        if layouts['prebuilt']:
            return manager.load_layout(layouts['prebuilt'][0], is_custom=False)
        elif layouts['custom']:
            return manager.load_layout(layouts['custom'][0], is_custom=True)
        else:
            print("❌ No layouts available")
            return None
