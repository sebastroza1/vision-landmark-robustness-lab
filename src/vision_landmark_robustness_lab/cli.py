from __future__ import annotations

import argparse
import json

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic landmark robustness")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", default="reports/quick-cpu")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.mode, arguments.seed)
    print(json.dumps({"status": result["status"], "output": arguments.output}, indent=2))


if __name__ == "__main__":
    main()

