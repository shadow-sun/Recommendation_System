"""Run the recommendation simulation with the API server."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.simulator import run_simulation


def main():
    print("Starting recommendation simulation...")
    from src.config.settings import config
    print(f"Make sure the API server is running: uvicorn src.services.gateway:app --port {config.serving.port}")
    print()

    # Wait for server to be ready
    import requests
    for _ in range(10):
        try:
            r = requests.get(f"http://localhost:{config.serving.port}/health", timeout=2.0)
            if r.status_code == 200:
                print("API server is ready.")
                break
        except requests.RequestException:
            time.sleep(1.0)
    else:
        print("Warning: API server not responding. Attempting simulation anyway...")

    results = run_simulation(
        api_url=f"http://localhost:{config.serving.port}",
        n_users=50,
        n_rounds=5,
    )

    print("\n" + "=" * 50)
    print("  Simulation Results")
    print("=" * 50)
    summary = results["simulation_summary"]
    print(f"  Users: {summary['n_users']}")
    print(f"  Rounds: {summary['n_rounds']}")
    print(f"  Final click rate: {summary['final_click_rate']:.4f}")
    print(f"  Avg latency: {summary['avg_latency_ms']:.1f}ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
