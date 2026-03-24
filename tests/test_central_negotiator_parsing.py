"""
Unit tests for CentralNegotiator JSON response parsing.
"""

import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm.central_negotiator import CentralNegotiator


def _make_negotiator() -> CentralNegotiator:
    return CentralNegotiator(model='dummy/test-model')


def test_parse_valid_json_payload():
    negotiator = _make_negotiator()
    response = """
    {
      "resolution": "reroute",
      "agent_actions": {
        "3": {"action": "move", "path": [[1, 1], [1, 2]]}
      },
      "reasoning": "basic"
    }
    """

    parsed = negotiator._parse_negotiation_response(response)
    assert parsed["resolution"] == "reroute"
    assert "3" in parsed["agent_actions"]


def test_parse_json_inside_markdown_fence():
    negotiator = _make_negotiator()
    response = """The plan is below.
```json
{
  "resolution": "wait",
  "agent_actions": {
    "1": {"action": "wait", "path": [[5, 5], [5, 5]]}
  },
  "reasoning": "safe wait"
}
```
Use this strictly.
"""

    parsed = negotiator._parse_negotiation_response(response)
    assert parsed["resolution"] == "wait"
    assert "1" in parsed["agent_actions"]


def test_parse_truncated_json_with_missing_closers():
    negotiator = _make_negotiator()
    response = (
        '{"resolution":"reroute","agent_actions":'
        '{"7":{"action":"move","path":[[1,1],[1,2],[1,3]]}'
    )

    parsed = negotiator._parse_negotiation_response(response)
    assert parsed.get("resolution") == "reroute"
    assert "7" in parsed.get("agent_actions", {})


def test_parse_truncated_json_with_unclosed_string_and_escape():
    negotiator = _make_negotiator()
    response = (
        '{"resolution":"reroute","agent_actions":{"2":{"action":"move","path":[[0,0],[0,1]]}},'
        '"reasoning":"route through corner\\'
    )

    parsed = negotiator._parse_negotiation_response(response)
    assert parsed.get("resolution") == "reroute"
    assert "2" in parsed.get("agent_actions", {})


def test_parse_handles_trailing_comma_before_recovery_close():
    negotiator = _make_negotiator()
    response = '{"resolution":"wait","agent_actions":{"4":{"action":"wait","path":[[3,3],[3,3]]}},'

    parsed = negotiator._parse_negotiation_response(response)
    assert parsed.get("resolution") == "wait"
    assert "4" in parsed.get("agent_actions", {})


def test_parse_returns_safe_wait_fallback_on_non_json_text():
    negotiator = _make_negotiator()
    parsed = negotiator._parse_negotiation_response("I cannot provide that right now.")

    assert parsed.get("resolution") == "wait"
    assert parsed.get("agent_actions") == {}


def test_validate_plan_missing_action_marks_fallback_required_not_rejected():
    negotiator = _make_negotiator()

    def always_valid(**kwargs):
        return {"valid": True, "reason": "ok"}

    results = negotiator._validate_plan(
        plan={"1": {"action": "wait", "path": [[1, 1], [1, 1]]}},
        conflict_data={"map_state": {}},
        agent_validators={1: always_valid, 2: always_valid}
    )

    assert results[1]["valid"] is True
    assert results[2]["valid"] is True
    assert results[2].get("fallback_required") is True
