"""
SkyGuard AI — Demo Scenario Evaluation Runner
Executes the comprehensive 504-observation detection pipeline evaluation.
Implementation maintained in Backend/tests/eval_demo_scenarios.py
"""

from pathlib import Path
import sys

backend_tests = Path(__file__).resolve().parent / "Backend" / "tests"
if str(backend_tests) not in sys.path:
    sys.path.insert(0, str(backend_tests))

# pyrefly: ignore [missing-import]
from eval_demo_scenarios import run_evaluation

if __name__ == "__main__":
    run_evaluation()
