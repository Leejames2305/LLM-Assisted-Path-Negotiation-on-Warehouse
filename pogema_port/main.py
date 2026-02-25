"""
POGEMA Port — Main simulation loop.

Usage:
    python -m pogema_port.main [--map PATH] [--obs-radius N] [--max-steps N] [--no-render] [--seed N]
"""

import argparse
import json
import os
import sys
from dotenv import load_dotenv
from pogema import pogema_v0

from .config import load_grid_config
from .agent_controller import LLMNegotiationController

DEFAULT_MAP = os.path.join(os.path.dirname(__file__), 'maps', 'corridor.json')


def run_pogema_simulation(
    map_path: str = DEFAULT_MAP,
    obs_radius: int = None,
    max_steps: int = None,
    seed: int = None,
    render: bool = True,
    enable_negotiation: bool = True,
    enable_spatial_hints: bool = True,
) -> dict:
    """
    Run a POGEMA simulation with LLM-assisted path negotiation.
    Returns a dict with final metrics.
    """
    # 1. Load environment config
    load_dotenv()

    # 2. Load map config and apply optional overrides
    grid_config = load_grid_config(map_path)

    # Apply overrides if provided
    if obs_radius is not None or max_steps is not None or seed is not None:
        from pogema import GridConfig
        grid_config = GridConfig(
            map='\n'.join(''.join('#' if c else '.' for c in row) for row in grid_config.map),
            agents_xy=list(grid_config.agents_xy),
            targets_xy=list(grid_config.targets_xy),
            obs_radius=obs_radius if obs_radius is not None else grid_config.obs_radius,
            max_episode_steps=max_steps if max_steps is not None else grid_config.max_episode_steps,
            seed=seed if seed is not None else grid_config.seed,
        )

    # 3. Create POGEMA environment
    env = pogema_v0(grid_config=grid_config)
    print(f"\n🗺️  Map: {map_path}")
    print(f"   Agents: {grid_config.num_agents}, Grid: {grid_config.height}×{grid_config.width}")
    print(f"   Max steps: {grid_config.max_episode_steps}, Seed: {grid_config.seed}")

    # 4. Create controller
    controller = LLMNegotiationController(
        grid_config=grid_config,
        enable_negotiation=enable_negotiation,
        enable_spatial_hints=enable_spatial_hints,
    )

    # 5. Reset environment
    obs, info = env.reset()
    terminated = [False] * grid_config.num_agents
    truncated = [False] * grid_config.num_agents
    step = 0

    print(f"\n🚀 Starting simulation...\n")

    # 6. Main loop
    while True:
        actions = controller.get_actions(obs, terminated, step)
        obs, rewards, terminated, truncated, info = env.step(actions)
        controller.update_positions(env, actions)

        if render:
            try:
                env.render()
            except Exception:
                pass  # Rendering may not work in all environments

        step += 1

        # Print step summary
        done_count = sum(terminated)
        print(f"  Step {step:4d}: actions={actions} | done={done_count}/{grid_config.num_agents} | rewards={[round(r,2) for r in rewards]}")

        if all(terminated) or all(truncated):
            break

        if step >= grid_config.max_episode_steps:
            break

    # 7. Print final metrics
    print(f"\n{'='*60}")
    print(f"🏁 SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Steps taken:     {step}")
    print(f"  Agents reached target: {sum(terminated)}/{grid_config.num_agents}")

    llm_stats = controller.get_llm_stats()
    print(f"\n📊 LLM NEGOTIATION STATS:")
    print(f"  Negotiations triggered: {llm_stats['negotiation_count']}")
    print(f"  Total tokens used:      {llm_stats['total_tokens_used']}")
    if llm_stats['negotiation_count'] > 0:
        print(f"  Avg negotiation time:   {llm_stats['avg_negotiation_duration_sec']:.2f}s")

    metrics = {
        'steps_taken': step,
        'agents_reached_target': sum(terminated),
        'num_agents': grid_config.num_agents,
        'llm_stats': llm_stats,
        'terminated': terminated,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Run POGEMA simulation with LLM-assisted path negotiation'
    )
    parser.add_argument(
        '--map',
        type=str,
        default=DEFAULT_MAP,
        help='Path to map config JSON (default: pogema_port/maps/corridor.json)',
    )
    parser.add_argument(
        '--obs-radius',
        type=int,
        default=None,
        help='Observation radius override',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Max episode steps override',
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable POGEMA rendering',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed override',
    )
    parser.add_argument(
        '--no-negotiate',
        action='store_true',
        help='Disable LLM negotiation (use A* only)',
    )
    parser.add_argument(
        '--no-spatial-hints',
        action='store_true',
        help='Disable spatial hints in negotiation',
    )
    parser.add_argument(
        '--save-metrics',
        type=str,
        default=None,
        help='Save final metrics to this JSON file',
    )

    args = parser.parse_args()

    metrics = run_pogema_simulation(
        map_path=args.map,
        obs_radius=args.obs_radius,
        max_steps=args.max_steps,
        seed=args.seed,
        render=not args.no_render,
        enable_negotiation=not args.no_negotiate,
        enable_spatial_hints=not args.no_spatial_hints,
    )

    if args.save_metrics:
        with open(args.save_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved to {args.save_metrics}")

    return 0 if metrics['agents_reached_target'] == metrics['num_agents'] else 1


if __name__ == '__main__':
    sys.exit(main())
