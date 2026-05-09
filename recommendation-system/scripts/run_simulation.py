from src.config import get_settings
from src.simulation import run_closed_loop


if __name__ == "__main__":
    output = get_settings().project_root / "data" / "processed" / "simulation_report.json"
    print(run_closed_loop(output=output))

