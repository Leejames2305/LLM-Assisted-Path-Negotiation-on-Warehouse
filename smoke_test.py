"""
Smoke test for POGEMA installation and basic functionality.
Run: python pogema_port/smoke_test.py
"""

from pogema import pogema_v0, GridConfig


def main():
    print("=== POGEMA Smoke Test ===")

    # Create a simple 8x8 env with 2 agents and random obstacles
    config = GridConfig(num_agents=2, size=8, density=0.3, seed=42)
    env = pogema_v0(grid_config=config)

    obs, info = env.reset()
    print(f"✅ Environment created. Observation shape: {len(obs)} agents")

    for step in range(10):
        actions = env.sample_actions()
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"  Step {step + 1}: rewards={rewards}, terminated={terminated}")

        if all(terminated) or all(truncated):
            print("  All agents done early.")
            break

    print("✅ Smoke test passed — POGEMA is working correctly.")


if __name__ == "__main__":
    main()
