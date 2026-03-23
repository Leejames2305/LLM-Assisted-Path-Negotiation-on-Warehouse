"""
Parallel Negotiator Manager for Multi-Group Conflict Resolution
Coordinates multiple central negotiators running in parallel for independent conflict groups
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Callable
from .central_negotiator import CentralNegotiator

logger = logging.getLogger(__name__)

class ParallelNegotiatorManager:
    def __init__(self, model: Optional[str] = None, enable_spatial_hints: bool = True):
        """
        Initialize the parallel negotiator manager.
        Creates a pool of negotiators for handling multiple conflict groups simultaneously.
        """
        self.model = model
        self.enable_spatial_hints = enable_spatial_hints
        self.max_parallel_negotiations = 32  # Maximum concurrent negotiations

        logger.info(f"Parallel Negotiator Manager initialized (max workers: {self.max_parallel_negotiations})")

    def negotiate_parallel_conflicts(
        self,
        conflict_groups: List[Dict],
        all_agent_validators: Dict[int, Callable],
        silent_mode: bool = False
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Negotiate multiple independent conflict groups in parallel.

        Args:
            conflict_groups: List of conflict group dicts, each with conflicting_agents, conflict_points, etc.
            all_agent_validators: Dict mapping agent_id -> validator function for ALL agents
            silent_mode: Suppress print output

        Returns:
            Tuple of (resolutions, negotiation_logs) where:
            - resolutions: List of resolution dicts, one per group
            - negotiation_logs: List of negotiation log data, one per group
        """
        if not conflict_groups:
            return [], []

        num_groups = len(conflict_groups)

        if not silent_mode:
            print(f"\n🎯 PARALLEL NEGOTIATION MODE")
            print(f"   Total conflict groups: {num_groups}")
            total_agents = sum(len(group['conflicting_agents']) for group in conflict_groups)
            print(f"   Total agents involved: {total_agents}")
            for i, group in enumerate(conflict_groups):
                agents = group['conflicting_agents']
                points = group['conflict_points']
                print(f"   Group {i+1}: {len(agents)} agent(s) at {len(points)} conflict point(s)")

        # Start parallel negotiations
        start_time = time.time()
        resolutions = []
        negotiation_logs = []

        max_workers = min(num_groups, self.max_parallel_negotiations)

        if not silent_mode:
            print(f"\n⚡ Starting {num_groups} parallel negotiation(s) with {max_workers} worker(s)...")

        # Define negotiation task for a single group
        def negotiate_single_group(group_index: int, conflict_data: Dict) -> Tuple[int, Dict, Dict]:
            """
            Negotiate a single conflict group.
            Returns (group_index, resolution, log_data)
            """
            # Create a dedicated negotiator for this group
            negotiator = CentralNegotiator(
                model=self.model,
                enable_spatial_hints=self.enable_spatial_hints
            )

            # Extract validators only for agents in this group
            group_agents = conflict_data['conflicting_agents']
            group_validators = {
                agent_id: all_agent_validators[agent_id]
                for agent_id in group_agents
                if agent_id in all_agent_validators
            }

            if not silent_mode:
                logger.info(f"Group {group_index+1}: Starting negotiation for agents {group_agents}")

            # Run negotiation
            result = negotiator.negotiate_path_conflict(
                conflict_data,
                agent_validators=group_validators
            )

            # Unpack result
            if isinstance(result, tuple):
                if len(result) >= 3:
                    resolution, refinement_history, prompts_data = result
                else:
                    resolution, refinement_history = result
                    prompts_data = {}
            else:
                resolution = result
                refinement_history = []
                prompts_data = {}

            # Build log data for this negotiation
            log_data = {
                'group_index': group_index,
                'conflict_data': conflict_data,
                'resolution': resolution,
                'refinement_history': refinement_history,
                'prompts_data': prompts_data,
                'agents_in_group': group_agents
            }

            if not silent_mode:
                if resolution:
                    logger.info(f"Group {group_index+1}: ✓ Negotiation successful")
                else:
                    logger.warning(f"Group {group_index+1}: ✗ Negotiation failed (deadlock)")

            return group_index, resolution, log_data

        # Execute negotiations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(negotiate_single_group, i, group): i
                for i, group in enumerate(conflict_groups)
            }

            # Collect results as they complete
            results_by_index = {}
            for future in as_completed(futures):
                group_index, resolution, log_data = future.result()
                group_index = futures[future]
                try:
                    _, resolution, log_data = future.result()
                except Exception as exc:
                    logger.exception(
                        "Parallel negotiation for group %d failed with exception", group_index
                    )
                    resolution = {}
                    log_data = {
                        "group_index": group_index,
                        "error": str(exc),
                    }

        # Sort results by group index to maintain order
        for i in range(num_groups):
            resolution, log_data = results_by_index[i]
            resolutions.append(resolution)
            negotiation_logs.append(log_data)

        # Report timing
        end_time = time.time()
        duration = end_time - start_time

        if not silent_mode:
            successful = sum(1 for r in resolutions if r)
            failed = num_groups - successful
            print(f"\n📊 PARALLEL NEGOTIATION COMPLETE")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Successful: {successful}/{num_groups}")
            if failed > 0:
                print(f"   Failed (deadlock): {failed}/{num_groups}")

        return resolutions, negotiation_logs

    def merge_resolutions(
        self,
        resolutions: List[Dict]
    ) -> Dict:
        """
        Merge multiple group resolutions into a single combined resolution dict.

        Args:
            resolutions: List of resolution dicts from parallel negotiations

        Returns:
            Combined resolution dict with all agent actions merged
        """
        merged = {
            'resolution': 'parallel_negotiation',
            'agent_actions': {},
            'reasoning': f'Parallel negotiation across {len(resolutions)} independent conflict group(s)',
            'num_groups': len(resolutions),
            'group_resolutions': []
        }

        for i, resolution in enumerate(resolutions):
            if not resolution:
                # Empty resolution (deadlock)
                merged['group_resolutions'].append({
                    'group_index': i,
                    'status': 'deadlock',
                    'agent_actions': {}
                })
                continue

            # Extract agent actions from this group
            if 'agent_actions' in resolution:
                actions = resolution['agent_actions']
            elif isinstance(resolution, dict):
                # Resolution might directly contain agent IDs as keys
                actions = {
                    str(k): v for k, v in resolution.items()
                    if str(k).isdigit() or isinstance(k, int)
                }
            else:
                actions = {}

            # Merge into combined agent_actions
            merged['agent_actions'].update(actions)

            # Track group-level resolution
            merged['group_resolutions'].append({
                'group_index': i,
                'status': 'resolved',
                'agent_actions': actions,
                'resolution_type': resolution.get('resolution', 'unknown')
            })

        return merged
