#!/usr/bin/env python3
"""
Test for validator key handling bug fix
Ensures _validate_plan can handle both string and integer keys in plan dictionaries
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm.central_negotiator import CentralNegotiator


def test_validate_plan_with_string_keys():
    """Test that _validate_plan works with string keys"""
    print("\n🧪 Test 1: Plan with string keys (e.g., '0', '1')")

    negotiator = CentralNegotiator()

    # Create a plan with string keys (as returned by JSON parsing)
    plan = {
        "0": {
            "action": "move",
            "path": [[1, 1], [2, 1], [3, 1]]
        },
        "1": {
            "action": "wait",
            "wait_turns": 1
        }
    }

    # Create mock conflict data
    conflict_data = {
        "map_state": {
            "grid": [
                [".", ".", ".", "."],
                [".", ".", ".", "."],
                [".", ".", ".", "."],
            ],
            "agents": {
                0: [1, 1],
                1: [1, 2]
            }
        }
    }

    # Create mock validators that always return valid
    def mock_validator(agent_id, proposed_action, current_state):
        return {
            "valid": True,
            "reason": "Mock validation passed"
        }

    agent_validators = {
        0: mock_validator,
        1: mock_validator
    }

    # This should NOT raise "No action provided" error
    results = negotiator._validate_plan(plan, conflict_data, agent_validators)

    # Check that both agents were validated
    assert 0 in results, "Agent 0 should have validation results"
    assert 1 in results, "Agent 1 should have validation results"
    assert results[0]["valid"] == True, "Agent 0 should be valid"
    assert results[1]["valid"] == True, "Agent 1 should be valid"
    assert "No action provided" not in results[0]["reason"], "Should not have 'No action provided' error"
    assert "No action provided" not in results[1]["reason"], "Should not have 'No action provided' error"

    print("✅ Test passed: String keys handled correctly")


def test_validate_plan_with_integer_keys():
    """Test that _validate_plan works with integer keys"""
    print("\n🧪 Test 2: Plan with integer keys (e.g., 0, 1)")

    negotiator = CentralNegotiator()

    # Create a plan with integer keys (as might be created by fallback methods)
    plan = {
        0: {
            "action": "move",
            "path": [[1, 1], [2, 1], [3, 1]]
        },
        1: {
            "action": "wait",
            "wait_turns": 1
        }
    }

    # Create mock conflict data
    conflict_data = {
        "map_state": {
            "grid": [
                [".", ".", ".", "."],
                [".", ".", ".", "."],
                [".", ".", ".", "."],
            ],
            "agents": {
                0: [1, 1],
                1: [1, 2]
            }
        }
    }

    # Create mock validators that always return valid
    def mock_validator(agent_id, proposed_action, current_state):
        return {
            "valid": True,
            "reason": "Mock validation passed"
        }

    agent_validators = {
        0: mock_validator,
        1: mock_validator
    }

    # This should NOT raise "No action provided" error
    results = negotiator._validate_plan(plan, conflict_data, agent_validators)

    # Check that both agents were validated
    assert 0 in results, "Agent 0 should have validation results"
    assert 1 in results, "Agent 1 should have validation results"
    assert results[0]["valid"] == True, "Agent 0 should be valid"
    assert results[1]["valid"] == True, "Agent 1 should be valid"
    assert "No action provided" not in results[0]["reason"], "Should not have 'No action provided' error"
    assert "No action provided" not in results[1]["reason"], "Should not have 'No action provided' error"

    print("✅ Test passed: Integer keys handled correctly")


def test_validate_plan_with_mixed_keys():
    """Test that _validate_plan works with mixed string and integer keys"""
    print("\n🧪 Test 3: Plan with mixed keys (string and integer)")

    negotiator = CentralNegotiator()

    # Create a plan with mixed keys
    plan = {
        "0": {
            "action": "move",
            "path": [[1, 1], [2, 1], [3, 1]]
        },
        1: {  # Integer key
            "action": "wait",
            "wait_turns": 1
        }
    }

    # Create mock conflict data
    conflict_data = {
        "map_state": {
            "grid": [
                [".", ".", ".", "."],
                [".", ".", ".", "."],
                [".", ".", ".", "."],
            ],
            "agents": {
                0: [1, 1],
                1: [1, 2]
            }
        }
    }

    # Create mock validators that always return valid
    def mock_validator(agent_id, proposed_action, current_state):
        return {
            "valid": True,
            "reason": "Mock validation passed"
        }

    agent_validators = {
        0: mock_validator,
        1: mock_validator
    }

    # This should NOT raise "No action provided" error
    results = negotiator._validate_plan(plan, conflict_data, agent_validators)

    # Check that both agents were validated
    assert 0 in results, "Agent 0 should have validation results"
    assert 1 in results, "Agent 1 should have validation results"
    assert results[0]["valid"] == True, "Agent 0 should be valid"
    assert results[1]["valid"] == True, "Agent 1 should be valid"
    assert "No action provided" not in results[0]["reason"], "Should not have 'No action provided' error"
    assert "No action provided" not in results[1]["reason"], "Should not have 'No action provided' error"

    print("✅ Test passed: Mixed keys handled correctly")


def test_validate_plan_missing_agent():
    """Test that _validate_plan correctly identifies missing agents"""
    print("\n🧪 Test 4: Plan with missing agent (should fail with 'No action provided')")

    negotiator = CentralNegotiator()

    # Create a plan with only agent 0
    plan = {
        "0": {
            "action": "move",
            "path": [[1, 1], [2, 1], [3, 1]]
        }
        # Agent 1 is missing
    }

    # Create mock conflict data
    conflict_data = {
        "map_state": {
            "grid": [
                [".", ".", ".", "."],
                [".", ".", ".", "."],
                [".", ".", ".", "."],
            ],
            "agents": {
                0: [1, 1],
                1: [1, 2]
            }
        }
    }

    # Create mock validators that always return valid
    def mock_validator(agent_id, proposed_action, current_state):
        return {
            "valid": True,
            "reason": "Mock validation passed"
        }

    agent_validators = {
        0: mock_validator,
        1: mock_validator  # Validator for agent 1 exists but no action in plan
    }

    # This SHOULD raise "No action provided" error for agent 1
    results = negotiator._validate_plan(plan, conflict_data, agent_validators)

    # Check that agent 0 is valid but agent 1 has error
    assert 0 in results, "Agent 0 should have validation results"
    assert 1 in results, "Agent 1 should have validation results"
    assert results[0]["valid"] == True, "Agent 0 should be valid"
    assert results[1]["valid"] == False, "Agent 1 should be invalid"
    assert "No action provided" in results[1]["reason"], "Agent 1 should have 'No action provided' error"

    print("✅ Test passed: Missing agent correctly identified")


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 VALIDATOR KEY HANDLING TESTS")
    print("=" * 60)

    try:
        test_validate_plan_with_string_keys()
        test_validate_plan_with_integer_keys()
        test_validate_plan_with_mixed_keys()
        test_validate_plan_missing_agent()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
