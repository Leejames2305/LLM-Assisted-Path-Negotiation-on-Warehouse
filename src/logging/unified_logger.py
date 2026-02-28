"""
Unified Logger for Multi-Robot Warehouse Simulation

Provides a single, comprehensive logging format that captures:
- Simulation scenario metadata
- Turn-by-turn agent states (planned_path + executed_path)
- HMAS-2 negotiation data (prompts, responses, validations, refinements)
- Map state changes

Output: logs/sim_log_[datetime].json
"""

import json
import os
import signal
import atexit
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Unified logger to capture simulation data
class UnifiedLogger:
    
    _instance = None  # Singleton for signal handling
    
    def __init__(self):
        self.log_data = {
            'scenario': {},
            'turns': [],
            'summary': {}
        }
        self._unsaved_data = False
        self._initialized = False
        self._log_file_path = None
        
        # Set singleton instance for signal handling
        UnifiedLogger._instance = self
        
        # Register signal handlers for emergency save
        self._setup_signal_handlers()
    
    # Setup signal handlers for emergency save on interrupt
    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            print(f"\n\n🛑 PROCESS INTERRUPTED! (Signal {signum})")
            print("💾 Saving simulation data before exit...")
            
            if UnifiedLogger._instance is not None:
                try:
                    log_file = UnifiedLogger._instance.finalize(emergency=True)
                    if log_file:
                        print(f"✅ Emergency save completed: {log_file}")
                except Exception as e:
                    print(f"❌ Error during emergency save: {e}")
            
            print("👋 Exiting gracefully...")
            import sys
            sys.exit(0)
        
        # Cleanup function to save on normal exit
        def cleanup_on_exit():
            if UnifiedLogger._instance is not None and UnifiedLogger._instance._unsaved_data:
                print("💾 Auto-saving simulation data on exit...")
                UnifiedLogger._instance.finalize()
        
        # Register handlers
        try:
            signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        except (ValueError, OSError):
            # Signal handling may fail in some environments (e.g., threads)
            pass
        
        atexit.register(cleanup_on_exit)
        
        # Safe print with fallback for encoding issues
        try:
            print("🛡️  Emergency save protection enabled (Ctrl+C safe)")
        except UnicodeEncodeError:
            print("[LOGGER] Emergency save protection enabled (Ctrl+C safe)")
    
    # Initialize logging with scenario metadata
    def initialize(self, scenario_data: Dict) -> None:
        self.log_data['scenario'] = {
            'type': scenario_data.get('type', 'simulation'),
            'simulation_mode': scenario_data.get('simulation_mode', 'turn_based'),
            'map_size': scenario_data.get('map_size', [0, 0]),
            'grid': scenario_data.get('grid', []),
            'initial_agents': self._to_json_safe(scenario_data.get('initial_agents', {})),
            'initial_targets': self._to_json_safe(scenario_data.get('initial_targets', {})),
            'initial_boxes': self._to_json_safe(scenario_data.get('initial_boxes', {})),
            'agent_goals': self._to_json_safe(scenario_data.get('agent_goals', {})),
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_data['turns'] = []
        self.log_data['task_completions'] = []
        self.log_data['summary'] = {}
        self._initialized = True
        self._unsaved_data = True
    
    # Log a single turn's data
    def log_turn(
        self,
        turn_num: int,
        agent_states: Dict,
        map_state: Dict,
        negotiation_data: Optional[Dict] = None
        ) -> None:

        turn_entry = {
            'turn': turn_num,
            'timestamp': datetime.now().isoformat(),
            'type': 'negotiation' if negotiation_data else 'routine',
            'agent_states': self._to_json_safe(agent_states),
            'map_state': self._to_json_safe(map_state),
            'negotiation': self._to_json_safe(negotiation_data) if negotiation_data else None
        }
        
        self.log_data['turns'].append(turn_entry)
        self._unsaved_data = True

    # Record a single task completion for lifelong throughput tracking
    def log_task_completion(self, agent_id: int, task_turn: int, task_duration_turns: int) -> None:
        if 'task_completions' not in self.log_data:
            self.log_data['task_completions'] = []
        self.log_data['task_completions'].append({
            'agent_id': agent_id,
            'turn': task_turn,
            'task_duration_turns': task_duration_turns,
            'timestamp': datetime.now().isoformat()
        })
        self._unsaved_data = True
    
    # Finalize and save the log to a file
    def finalize(self, emergency: bool = False, performance_metrics: Optional[Dict] = None) -> Optional[str]:

        if not self.log_data['turns']:
            print("⚠️  No turn data to save")
            return None
        
        simulation_mode = self.log_data.get('scenario', {}).get('simulation_mode', 'turn_based')
        
        if simulation_mode == 'lifelong':
            self.log_data['summary'] = self._compute_lifelong_summary(performance_metrics)
        else:
            self.log_data['summary'] = self._compute_turnbased_summary(performance_metrics)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "emergency_" if emergency else ""
        filename = f"{prefix}sim_log_{timestamp}.json"
        
        # Save to logs directory
        log_path = os.path.join("logs", filename)
        os.makedirs("logs", exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, default=str)
        
        self._unsaved_data = False
        self._log_file_path = log_path
        
        turns = self.log_data['turns']
        negotiation_turns = [t for t in turns if t.get('type') == 'negotiation']
        print(f"📝 Simulation log saved to: {log_path}")
        print(f"📊 Summary: {len(turns)} turns, {len(negotiation_turns)} negotiations")
        
        return log_path

    # Build turn-based summary dictionary
    def _compute_turnbased_summary(self, performance_metrics: Optional[Dict] = None) -> Dict:
        turns = self.log_data['turns']
        negotiation_turns = [t for t in turns if t.get('type') == 'negotiation']
        routine_turns = [t for t in turns if t.get('type') == 'routine']
        hmas2_metrics = self._calculate_hmas2_metrics(negotiation_turns)
        return {
            'total_turns': len(turns),
            'routine_turns': len(routine_turns),
            'negotiation_turns': len(negotiation_turns),
            'total_conflicts': len(negotiation_turns),
            'hmas2_metrics': hmas2_metrics,
            'performance_metrics': performance_metrics or {},
            'completion_timestamp': datetime.now().isoformat()
        }

    # Build lifelong summary dictionary with throughput analytics
    def _compute_lifelong_summary(self, performance_metrics: Optional[Dict] = None) -> Dict:
        task_completions = self.log_data.get('task_completions', [])
        turns = self.log_data['turns']
        negotiation_turns = [t for t in turns if t.get('type') == 'negotiation']

        throughput_timeline = self._compute_throughput_timeline(task_completions)

        per_agent: Dict[int, Dict] = {}
        for tc in task_completions:
            aid = tc['agent_id']
            per_agent.setdefault(aid, {'tasks': 0, 'task_durations': []})
            per_agent[aid]['tasks'] += 1
            per_agent[aid]['task_durations'].append(tc['task_duration_turns'])

        pm = performance_metrics or {}
        return {
            'simulation_mode': 'lifelong',
            'total_turns': len(turns),
            'total_tasks_completed': len(task_completions),
            'throughput_tasks_per_second': pm.get('throughput_tasks_per_second', 0),
            'throughput_tasks_per_turn': pm.get('throughput_tasks_per_turn', 0),
            'throughput_timeline': throughput_timeline,
            'per_agent_stats': {
                str(aid): {
                    'tasks': data['tasks'],
                    'avg_task_turns': (
                        sum(data['task_durations']) / len(data['task_durations'])
                        if data['task_durations'] else 0
                    )
                }
                for aid, data in per_agent.items()
            },
            'negotiation_metrics': {
                'total_negotiations': len(negotiation_turns),
                'hmas2_metrics': self._calculate_hmas2_metrics(negotiation_turns)
            },
            'llm_cost': {
                'total_tokens': pm.get('total_tokens_used', 0),
                'tokens_per_task': (
                    pm.get('total_tokens_used', 0) / max(len(task_completions), 1)
                )
            },
            'performance_metrics': pm,
            'completion_timestamp': datetime.now().isoformat()
        }

    # Compute throughput in 30-second windows from task completion records
    def _compute_throughput_timeline(self, task_completions: List[Dict]) -> List[Dict]:
        if not task_completions:
            return []

        try:
            first_ts = datetime.fromisoformat(task_completions[0]['timestamp'])
            last_ts = datetime.fromisoformat(task_completions[-1]['timestamp'])
            total_seconds = max((last_ts - first_ts).total_seconds(), 1)
        except (KeyError, ValueError):
            return []

        window_size = 30  # seconds
        num_windows = int(total_seconds / window_size) + 1
        timeline = []

        for w in range(num_windows):
            w_start = w * window_size
            w_end = (w + 1) * window_size
            count = 0
            for tc in task_completions:
                try:
                    ts = datetime.fromisoformat(tc['timestamp'])
                    offset = (ts - first_ts).total_seconds()
                    if w_start <= offset < w_end:
                        count += 1
                except (KeyError, ValueError):
                    continue
            if count > 0:
                timeline.append({
                    'window_start_seconds': w_start,
                    'window_end_seconds': w_end,
                    'tasks_completed': count,
                    'throughput_per_second': round(count / window_size, 4)
                })

        return timeline
    
    # Calculate negotiation metrics
    def _calculate_hmas2_metrics(self, negotiation_turns: List[Dict]) -> Dict:
        total_validations = 0
        approvals = 0
        rejections = 0
        alternatives_suggested = 0
        total_refinement_iterations = 0
        
        for turn in negotiation_turns:
            negotiation = turn.get('negotiation')
            if not negotiation:
                continue
            
            hmas2 = negotiation.get('hmas2_stages', {})
            
            # Count agent validations
            validations = hmas2.get('agent_validations', {})
            for agent_id, val_data in validations.items():
                total_validations += 1
                val_result = val_data.get('validation_result', {})
                if isinstance(val_result, dict):
                    if val_result.get('valid', False):
                        approvals += 1
                    else:
                        rejections += 1
                    if val_data.get('alternative_suggested'):
                        alternatives_suggested += 1
            
            # Count refinement iterations
            refinement = hmas2.get('refinement_loop', {})
            total_refinement_iterations += refinement.get('total_iterations', 0)
        
        disagreement_rate = rejections / total_validations if total_validations > 0 else 0.0
        
        return {
            'total_validations': total_validations,
            'approvals': approvals,
            'rejections': rejections,
            'alternatives_suggested': alternatives_suggested,
            'total_refinement_iterations': total_refinement_iterations,
            'disagreement_rate': round(disagreement_rate, 3)
        }
    
    # Recursive conversion to JSON-serializable format
    def _to_json_safe(self, obj: Any, visited: Optional[set] = None) -> Any:
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        
        # Prevent circular references
        if obj_id in visited:
            return None
        
        if obj is None:
            return None
        
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        if isinstance(obj, dict):
            visited_copy = visited.copy()
            visited_copy.add(obj_id)
            result = {}
            for k, v in obj.items():
                # Convert keys to strings for JSON compatibility
                key = str(k) if not isinstance(k, str) else k
                result[key] = self._to_json_safe(v, visited_copy)
            return result
        
        if isinstance(obj, (list, tuple)):
            visited_copy = visited.copy()
            visited_copy.add(obj_id)
            return [self._to_json_safe(item, visited_copy) for item in obj]
        
        # Handle numpy arrays
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        
        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            visited_copy = visited.copy()
            visited_copy.add(obj_id)
            return self._to_json_safe(obj.__dict__, visited_copy)
        
        # Fallback: convert to string
        try:
            return str(obj)
        except:
            return None
    
    def get_log_data(self) -> Dict:
        return self.log_data
    
    def has_unsaved_data(self) -> bool:
        return self._unsaved_data
    
    def get_last_saved_path(self) -> Optional[str]:
        return self._log_file_path

# Helper function to create negotiation data structure for logging
def create_negotiation_data(
    conflict_data: Dict,
    system_prompt: str,
    user_prompt: str,
    llm_response: Dict,
    model_used: str,
    agent_validations: Optional[Dict] = None,
    refinement_loop: Optional[Dict] = None,
    final_actions: Optional[Dict] = None,
    validation_overrides: Optional[Dict] = None
    ) -> Dict:

    return {
        'conflict_data': conflict_data,
        'hmas2_stages': {
            'central_negotiation': {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'llm_response': llm_response,
                'model_used': model_used
            },
            'agent_validations': agent_validations or {},
            'refinement_loop': refinement_loop or {
                'total_iterations': 0,
                'final_status': 'none',
                'iterations': []
            },
            'final_actions': final_actions or {},
            'validation_overrides': validation_overrides or {}
        }
    }
