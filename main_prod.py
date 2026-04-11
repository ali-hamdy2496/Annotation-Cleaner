"""
Production execution script for object placement optimization.

Same pipeline as main.py — reads input JSON, runs the full region-splitting +
greedy placement optimization, and writes output JSON — but with all debugging
plots and visualizations stripped out.

Usage:
    python3 main_prod.py [input.json] [output.json]

Defaults:
    input.json  = AnnotationCleaner_CurveLoops.json
    output.json = output.json
"""

import sys
import json

from json_helper import load_problem_data
from main import _run_optimization_core


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "AnnotationCleaner_CurveLoops.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"

    print(f"Loading data from {input_path}...")
    movables, fixed_obstacles, placement_bounds = load_problem_data(input_path)

    result = _run_optimization_core(movables, fixed_obstacles, placement_bounds)

    print(f"\nSaving output to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(result["output_data"], f, indent=2)
    print(f"Saved {output_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total time:          {result['total_time']:.2f}s")
    print(f"Average displacement: {result['avg_displacement']:.4f}")
    print(f"Final overlaps:       {result['num_overlaps']}")

    return result["num_overlaps"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
