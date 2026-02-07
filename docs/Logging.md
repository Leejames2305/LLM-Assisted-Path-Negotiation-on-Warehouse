# Logging

The format of the `logs/sim_log_*.json` files are as follows:

```json
{
  "scenario": {
    "type": "interactive_simulation",
    "map_size": [10, 8],
    "grid": [[...grid array...]],
    "initial_agents": {"0": [4, 1], "1": [5, 6]},
    "initial_targets": {"0": [8, 6], "1": [2, 1]},
    "initial_boxes": {"0": [3, 1], "1": [6, 6]},
    "agent_goals": {"0": 0, "1": 1},
    "timestamp": "2025-11-30T16:57:17.694698"
  },
  "turns": [
    {
      "turn": 0,
      "timestamp": "2025-11-30T16:57:17.694698",
      "type": "routine",
      "agent_states": {
        "0": {
          "position": [4, 1],
          "target_position": [3, 1],
          "planned_path": [[4, 1], [3, 1]],
          "executed_path": [[4, 1]],
          "is_waiting": false,
          "wait_turns_remaining": 0,
          "has_negotiated_path": false,
          "carrying_box": false,
          "box_id": null,
          "priority": 1
        }
      },
      "map_state": {
        "boxes": {"0": [3, 1], "1": [6, 6]},
        "targets": {"0": [8, 6], "1": [2, 1]},
        "dimensions": [10, 8]
      },
      "negotiation": null
    },
    {
      "turn": 1,
      "timestamp": "2025-11-30T17:01:17.312133",
      "type": "negotiation",
      "agent_states": {
        "0": {
          "position": [3, 1],
          "target_position": [8, 6],
          "planned_path": [[4, 1], [5, 1], ...],
          "executed_path": [[4, 1], [3, 1]],
          "has_negotiated_path": true,
          "carrying_box": true,
          "box_id": 0
        }
      },
      "map_state": {
        "boxes": {},
        "targets": {"0": [8, 6], "1": [2, 1]},
        "dimensions": [10, 8]
      },
      "negotiation": {
        "conflict_data": {
          "agents": [
            {"id": 0, "current_pos": [3, 1], "target_pos": [8, 6], "planned_path": [[3, 1], [4, 1], [5, 1], ...]},
            {"id": 1, "current_pos": [6, 6], "target_pos": [2, 1], "planned_path": [[6, 6], [5, 6], [4, 6], ...]}
          ],
          "conflict_points": [[5, 4]],
          "map_state": {
            "agents": {"0": [3, 1], "1": [6, 6]},
            "boxes": {},
            "targets": {"0": [8, 6], "1": [2, 1]},
            "agent_goals": {"0": 0, "1": 1},
            "grid": [...]
          },
          "turn": 1
        },
        "hmas2_stages": {
          "central_negotiation": {
            "system_prompt": "You are an expert robot conflict resolver...",
            "user_prompt": "TURN 1 - PATH CONFLICT DETECTED\n\nAGENTS IN CONFLICT:\n- Agent 0: At (3, 1), going to (8, 6)...",
            "llm_response": {
              "0": {"action": "move", "path": [[3, 1], [4, 1], ...], "priority": 1},
              "1": {"action": "move", "path": [[6, 6], [5, 6], ...], "priority": 1},
              "refinement_history": [{"iteration": 0, "stage": "initial_negotiation", ...}]
            },
            "model_used": "x-ai/grok-4.1-fast:free"
          },
          "agent_validations": {
            "0": {
              "agent_id": 0,
              "validation_result": {"valid": true, "reason": "path_valid_orthogonal_moves_no_walls", "alternative": null},
              "alternative_suggested": null
            },
            "1": {
              "agent_id": 1,
              "validation_result": {"valid": true, "reason": "path_valid_orthogonal_moves_no_walls", "alternative": null},
              "alternative_suggested": null
            }
          },
          "refinement_loop": {
            "total_iterations": 2,
            "final_status": "unknown",
            "iterations": [...]
          },
          "final_actions": {"0": {"action": "move", "path": [...]}, "1": {"action": "move", "path": [...]}},
          "validation_overrides": {}
        }
      }
    }
  ],
  "summary": {
    "total_turns": 31,
    "routine_turns": 30,
    "negotiation_turns": 1,
    "total_conflicts": 1,
    "hmas2_metrics": {
      "total_validations": 2,
      "approvals": 2,
      "rejections": 0,
      "alternatives_suggested": 0,
      "total_refinement_iterations": 2,
      "disagreement_rate": 0.0
    },
    "completion_timestamp": "2025-11-30T17:01:30.796715"
  }
}
```

## Example console output

```
🎯 Starting conflict negotiation (max 5 refinement iterations)

📋 STAGE 1: INITIAL NEGOTIATION
   🤖 System prompt (captured for logging)
   📝 User prompt describing agents 0/1, conflict points, and spatial hints

🔄 STAGE 2: VALIDATION (Iteration 1/5)
   Results: 2 agents, 1 rejection
   📞 Requesting LLM refinement for agent 1

🔄 STAGE 2: VALIDATION (Iteration 2/5)
   Results: 2 agents, 0 rejections
   ✅ All agents approved! Plan accepted.
```
