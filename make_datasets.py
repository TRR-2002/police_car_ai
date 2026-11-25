"""Generate reproducible sample datasets for MARL roles and path prediction."""

from __future__ import annotations

import csv
import argparse
import math
from pathlib import Path
import random

from prediction import PREDICTION_HORIZONS


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WIDTH, HEIGHT = 30.0, 20.0
NUM_AGENTS = 4


def make_role_dataset(rows: int, seed: int) -> Path:
    random.seed(seed)
    path = DATA / "team_roles.csv"
    fields = [f"obs_{index}" for index in range(7 + NUM_AGENTS)] + ["action"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for _ in range(rows // NUM_AGENTS):
            player_x, player_y = random.uniform(0, WIDTH), random.uniform(0, HEIGHT)
            player_vx, player_vy = random.uniform(-5, 5), random.uniform(-5, 5)
            for agent in range(NUM_AGENTS):
                car_x, car_y = random.uniform(0, WIDTH), random.uniform(0, HEIGHT)
                car_vx, car_vy = random.uniform(-3.8, 3.8), random.uniform(-3.8, 3.8)
                relative_x = (player_x - car_x) / WIDTH
                relative_y = (player_y - car_y) / HEIGHT
                detected = float(math.hypot(player_x - car_x, player_y - car_y) <= 16)
                identity = [float(index == agent) for index in range(NUM_AGENTS)]
                observation = [
                    relative_x,
                    relative_y,
                    player_vx / 5,
                    player_vy / 5,
                    car_vx / 3.8,
                    car_vy / 3.8,
                    detected,
                    *identity,
                ]
                # Expert team plan: direct, lead, left flank, right flank.
                writer.writerow([*observation, agent])
    return path


def make_trajectory_dataset(rows: int, seed: int) -> Path:
    random.seed(seed)
    path = DATA / "trajectories.csv"
    inputs = ["x", "y", "vx", "vy", "ax", "ay"]
    targets = [
        f"delta_{axis}_{str(horizon).replace('.', '_')}"
        for horizon in PREDICTION_HORIZONS
        for axis in ("x", "y")
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(inputs + targets)
        for _ in range(rows):
            x, y = random.uniform(0, WIDTH), random.uniform(0, HEIGHT)
            speed = random.uniform(0, 5)
            heading = random.uniform(-math.pi, math.pi)
            vx, vy = math.cos(heading) * speed, math.sin(heading) * speed
            # Sample braking, acceleration, and turns.
            tangent_acceleration = random.uniform(-2.5, 2.5)
            turning_acceleration = random.uniform(-3.5, 3.5)
            ax = math.cos(heading) * tangent_acceleration - math.sin(heading) * turning_acceleration
            ay = math.sin(heading) * tangent_acceleration + math.cos(heading) * turning_acceleration
            features = [x / WIDTH, y / HEIGHT, vx / 7, vy / 7, ax / 10, ay / 10]
            outputs = []
            for horizon in PREDICTION_HORIZONS:
                future_x = min(WIDTH - 0.01, max(0.0, x + vx * horizon + 0.5 * ax * horizon**2))
                future_y = min(HEIGHT - 0.01, max(0.0, y + vy * horizon + 0.5 * ay * horizon**2))
                outputs.extend(((future_x - x) / WIDTH, (future_y - y) / HEIGHT))
            writer.writerow(features + outputs)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate reproducible police AI datasets")
    parser.add_argument("--rows", type=int, default=50_000, help="rows per dataset")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.rows < 1_000:
        raise ValueError("--rows must be at least 1000")
    DATA.mkdir(parents=True, exist_ok=True)
    # Role rows are emitted in complete four-agent teams.
    rows = args.rows - args.rows % NUM_AGENTS
    role_path = make_role_dataset(rows, seed=args.seed)
    trajectory_path = make_trajectory_dataset(rows, seed=args.seed + 1)
    print(f"created: {role_path} ({sum(1 for _ in role_path.open(encoding='utf-8')) - 1} rows)")
    print(
        f"created: {trajectory_path} "
        f"({sum(1 for _ in trajectory_path.open(encoding='utf-8')) - 1} rows)"
    )


if __name__ == "__main__":
    main()
