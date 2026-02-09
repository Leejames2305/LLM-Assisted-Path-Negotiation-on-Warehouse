# Random Map Generation Feature

## Overview
The benchmark tool now supports randomly generating warehouse layouts with customizable dimensions and wall density. This feature allows users to test the multi-robot navigation system on procedurally generated maps.

## Usage

When running the benchmark tool:

```bash
python benchmark_tool.py
```

You will be prompted to choose a layout mode:
1. Select existing layout
2. Generate random map

### Random Map Generation Parameters

If you select option 2, you will be asked for:

1. **Map Width** (8-50, default: 20)
   - The width of the generated map in cells
   - Minimum: 8, Maximum: 50

2. **Map Height** (8-50, default: 15)
   - The height of the generated map in cells
   - Minimum: 8, Maximum: 50

3. **Wall Specification** (default: 15%)
   - Can be specified as:
     - **Percentage**: e.g., `20%` - percentage of internal cells to fill with walls (max 70%)
     - **Absolute count**: e.g., `50` - exact number of walls to place

### Example Interaction

```
Choose layout mode:
  1. Select existing layout
  2. Generate random map

Select mode (1/2, default 1): 2

============================================================
🎲 RANDOM MAP GENERATION
============================================================

Map width (8-50, default 20): 15
Map height (8-50, default 15): 12
Walls (e.g., '20%' or '50', default 15%): 25%

🔨 Generating random map (15x12 with ~32 walls)...

✅ Map generated successfully!

Preview:
  ###############
  #...#..#.....##
  #.##..#.......#
  ...
```

## Features

- **Guaranteed Connectivity**: The algorithm ensures all walkable cells are reachable from any other walkable cell
- **Seeded Generation**: Uses the benchmark seed for reproducible results
- **Persistent Map**: The generated map persists throughout all benchmark rounds
- **Random Entity Placement**: Agents, boxes, and targets are still randomly placed using the existing benchmark method

## Implementation Details

### Key Functions

1. **`generate_random_grid(width, height, wall_count, seed)`**
   - Generates a random grid with specified dimensions and wall count
   - Ensures connectivity by validating each wall placement
   - Returns the grid as a list of strings

2. **`bfs_all_reachable(grid, start)`**
   - Performs breadth-first search to find all reachable cells
   - Used to validate connectivity after placing walls

3. **`create_layout_from_grid(grid, name)`**
   - Converts a generated grid into a layout dictionary
   - Compatible with existing benchmark infrastructure

4. **`prompt_for_random_map_generation(seed)`**
   - Interactive prompt for user to specify map parameters
   - Validates input and generates the map

### Algorithm

The random map generation algorithm:

1. Creates a grid with walls on all borders
2. Fills interior with walkable floor cells
3. Randomly places walls one at a time
4. After each wall placement, validates connectivity
5. If connectivity is broken, rejects the wall
6. Continues until requested wall count is reached

This ensures the generated maps are always solvable while maintaining randomness.

## Limitations

- Maximum map size: 50x50 cells
- Minimum map size: 8x8 cells
- Maximum wall density: 70% of internal cells
- Wall placement is randomized but connectivity-constrained

## Testing

The feature has been tested with:
- Various map sizes (8x8 to 50x50)
- Different wall densities (0% to 70%)
- Both percentage and absolute wall count specifications
- Edge cases (minimum size, maximum density, default values)
- Connectivity validation for all generated maps

All tests pass successfully with proper validation and error handling.
