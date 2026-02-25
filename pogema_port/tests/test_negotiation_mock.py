"""
Mock negotiation test — no real OpenRouter API calls.
Tests CentralNegotiator and AgentValidator with mocked responses.
"""

import json
from unittest.mock import patch, MagicMock


def test_central_negotiator_mock():
    """Test CentralNegotiator returns resolution dict with mocked API."""
    from pogema_port.negotiation.central_negotiator import CentralNegotiator

    canned_response = json.dumps({
        "resolution": "priority",
        "agent_actions": {
            "0": {"action": "move", "path": [[1, 4], [1, 5], [1, 6]], "priority": 1},
            "1": {"action": "wait", "path": [[1, 4]], "priority": 2}
        },
        "reasoning": "Agent 0 moves first; Agent 1 waits at conflict point."
    })

    conflict_data = {
        "turn": 1,
        "agents": [
            {"id": 0, "current_pos": [1, 3], "target_pos": [1, 7], "planned_path": [[1, 3], [1, 4], [1, 5], [1, 6], [1, 7]]},
            {"id": 1, "current_pos": [1, 5], "target_pos": [1, 0], "planned_path": [[1, 5], [1, 4], [1, 3], [1, 2], [1, 1], [1, 0]]},
        ],
        "conflict_points": [[1, 4]],
        "map_state": {
            "agents": {0: [1, 3], 1: [1, 5]},
            "targets": {0: [1, 7], 1: [1, 0]},
            "grid": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
        }
    }

    with patch.object(
        CentralNegotiator.__module__ + '.OpenRouterClient',
        'send_request',
        return_value=canned_response
    ) if False else patch(
        'pogema_port.negotiation.openrouter_client.OpenRouterClient.send_request',
        return_value=canned_response
    ):
        negotiator = CentralNegotiator(model='mock-model')
        plan, history, prompts = negotiator.negotiate_path_conflict(conflict_data)

    assert isinstance(plan, dict), f"Expected dict, got {type(plan)}"
    assert '0' in plan or 0 in plan, f"Agent 0 missing from plan: {plan}"
    print(f"✅ test_central_negotiator_mock passed: plan keys = {list(plan.keys())}")


def test_conflict_detection():
    """Test ConflictDetector identifies head-on conflict."""
    from pogema_port.negotiation.conflict_detector import ConflictDetector

    cd = ConflictDetector(num_rows=3, num_cols=8)

    # Two agents heading toward each other on row 1
    paths = {
        0: [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
        1: [(1, 4), (1, 3), (1, 2), (1, 1), (1, 0)],
    }
    conflicts = cd.detect_path_conflicts(paths)

    assert conflicts['has_conflicts'], "Expected conflict for head-on agents"
    assert 0 in conflicts['conflicting_agents'] or 1 in conflicts['conflicting_agents']
    print(f"✅ test_conflict_detection passed: conflict at {conflicts['conflict_points']}")


def test_pathfinder_actions():
    """Test that A* produces valid POGEMA action sequences."""
    from pogema_port.pathfinding.astar import SimplePathfinder, path_to_pogema_actions

    pf = SimplePathfinder(num_rows=8, num_cols=8)
    walls = {(2, 2), (2, 3), (3, 2), (3, 3)}

    path = pf.find_path((0, 0), (7, 7), walls)
    assert path, "Expected non-empty path"
    assert path[0] == (0, 0) and path[-1] == (7, 7)

    actions = path_to_pogema_actions(path)
    assert all(0 <= a <= 4 for a in actions), f"Invalid action in {actions}"
    print(f"✅ test_pathfinder_actions passed: path length={len(path)}, actions={actions}")


def test_config_load():
    """Test that all example map configs load and create valid POGEMA envs."""
    import os
    from pogema_port.config import load_grid_config
    from pogema import pogema_v0

    maps_dir = os.path.join(os.path.dirname(__file__), '..', 'maps')
    for map_name in ['corridor.json', 'open_warehouse.json', 'bottleneck.json']:
        path = os.path.join(maps_dir, map_name)
        cfg = load_grid_config(path)
        env = pogema_v0(grid_config=cfg)
        obs, _ = env.reset()
        assert len(obs) == cfg.num_agents, f"Mismatch for {map_name}"
        print(f"✅ test_config_load: {map_name} loaded OK ({cfg.num_agents} agents)")


if __name__ == "__main__":
    test_conflict_detection()
    test_pathfinder_actions()
    test_config_load()
    test_central_negotiator_mock()
    print("\n✅ All mock tests passed.")
