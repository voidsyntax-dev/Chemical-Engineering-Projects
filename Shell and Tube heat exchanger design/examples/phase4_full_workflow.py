"""Run the complete design and reporting workflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from any directory without PYTHONPATH

from she.cli import load_input
from she.rating import solve_design
from she.reporting import generate_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    case, trial, passes, constraints = load_input(root / "sample_input.json")
    solution = solve_design(case, trial, passes, constraints)
    generate_report(solution, case, root / "phase4_report.txt", root / "phase4_report.xlsx")
    print(f"Rated duty: {solution.rating.duty_w:.0f} W")
    print(f"Reports written to {root / 'phase4_report.txt'} and {root / 'phase4_report.xlsx'}")


if __name__ == "__main__":
    main()
