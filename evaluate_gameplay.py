"""Deterministic gameplay evaluation for trained and analytical predictors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ai import Vec2
from main import make_city
from marl import DecentralizedController
from prediction import load_neural_trajectory_model
from simulation import PoliceSimulation


ROOT = Path(__file__).resolve().parent
SPAWNS = [Vec2(5.5, 15.5), Vec2(25.5, 15.5), Vec2(15.5, 5.5), Vec2(25.5, 5.5)]
PATROLS = [
    [(5, 15), (5, 10), (10, 10), (10, 15)],
    [(25, 15), (20, 15), (20, 10), (25, 10)],
    [(15, 5), (10, 5), (10, 0), (15, 0)],
    [(25, 5), (20, 5), (20, 0), (25, 0)],
]


def player_input(name: str, seconds: float) -> Vec2:
    if name == "east":
        return Vec2(1, 0)
    if name == "zigzag":
        return Vec2(1, 1 if int(seconds / 1.4) % 2 == 0 else -1)
    if name == "square":
        return (Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0), Vec2(0, -1))[int(seconds / 2.2) % 4]
    angle = seconds * 1.3
    return Vec2(math.cos(angle), math.sin(angle))


def run_suite(use_neural: bool, duration: float) -> dict:
    controller = DecentralizedController.load(ROOT / "marl_policy.pt", 4)
    model = load_neural_trajectory_model(ROOT / "trajectory_model.pt") if use_neural else None
    outcomes = []
    for scenario in ("east", "zigzag", "square", "circle"):
        simulation = PoliceSimulation(make_city(), controller, model, SPAWNS, PATROLS)
        steps = int(duration * 60)
        for step in range(steps):
            simulation.step(1 / 60, player_input(scenario, step / 60))
            if simulation.arrested:
                break
        outcomes.append(
            {
                "scenario": scenario,
                "arrested": simulation.arrested,
                "seconds": round(simulation.elapsed, 3),
            }
        )
    captured = [item["seconds"] for item in outcomes if item["arrested"]]
    return {
        "mode": "trained_trajectory_ensemble" if use_neural else "analytical_only",
        "captures": len(captured),
        "scenarios": len(outcomes),
        "mean_capture_seconds": round(sum(captured) / len(captured), 3) if captured else None,
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=ROOT / "gameplay_report.json")
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    report = {
        "physics_hz": 60,
        "decision_hz": 5,
        "duration_limit_seconds": args.duration,
        "analytical": run_suite(False, args.duration),
        "trained": run_suite(True, args.duration),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
