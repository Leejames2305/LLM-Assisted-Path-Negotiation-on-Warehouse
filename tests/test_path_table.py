"""Unit tests for PathTable reservations and conflict lookups."""

from src.navigation.path_table import PathTable


def test_path_table_vertex_and_edge_conflicts():
    table = PathTable()

    # Agent 1: (1,1)->(2,1)
    table.insert_path(1, [(1, 1), (2, 1)], hold_until=4)

    # Vertex conflict at t=1
    assert table.constrained((0, 1), (2, 1), 1)

    # Edge-swap conflict at t=1 (other moves 1,1->2,1)
    assert table.constrained((2, 1), (1, 1), 1)



def test_path_table_target_hold_conflict_and_delete():
    table = PathTable()

    table.insert_path(1, [(1, 1), (1, 2)], hold_until=5)

    # Goal hold should block entering goal location at later timesteps.
    assert table.constrained((1, 1), (1, 2), 4)

    conflicts = table.get_conflicting_agents(2, (1, 1), (1, 2), 4)
    assert 1 in conflicts

    table.delete_path(1)
    assert not table.constrained((1, 1), (1, 2), 4)


def test_path_table_bulk_build_and_conflict_helpers():
    table = PathTable()
    paths = {
        1: [(0, 0), (1, 0), (2, 0)],
        2: [(2, 0), (1, 0), (0, 0)],
        3: [(2, 2), (2, 1), (1, 1), (2, 1)],
    }
    table.build_from_paths(paths)

    points = table.get_conflict_points()
    assert points
    assert ((1, 0), 1) in points

    conflicting_agents = table.get_conflicting_agents_set()
    assert 1 in conflicting_agents
    assert 2 in conflicting_agents

    assert table.conflict_count() > 0
