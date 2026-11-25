"""Decentralized motion prediction and utility-based interception."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from ai import GridMap, PoliceCarAI, Vec2, astar


def _clamp_vector(vector: Vec2, maximum: float) -> Vec2:
    return vector if vector.length() <= maximum else vector.normalized() * maximum


PREDICTION_HORIZONS = (0.35, 0.7, 1.1, 1.6, 2.2)


class NeuralTrajectoryModel(nn.Module):
    """Supervised predictor trained on sampled position/velocity/acceleration data."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(PREDICTION_HORIZONS) * 2),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def load_neural_trajectory_model(path: str | Path) -> NeuralTrajectoryModel:
    model = NeuralTrajectoryModel()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


@dataclass
class AlphaBetaGammaPredictor:
    """Noise-tolerant position, velocity, and acceleration estimator."""

    alpha: float = 0.80
    beta: float = 0.35
    gamma: float = 0.06
    position: Vec2 | None = None
    velocity: Vec2 = Vec2(0.0, 0.0)
    acceleration: Vec2 = Vec2(0.0, 0.0)
    confidence: float = 0.0

    def observe(self, measurement: Vec2, dt: float, reported_velocity: Vec2) -> None:
        dt = min(max(dt, 1e-3), 0.2)
        if self.position is None:
            self.position = measurement
            self.velocity = reported_velocity
            self.confidence = 0.3
            return
        predicted = self.position + self.velocity * dt + self.acceleration * (0.5 * dt * dt)
        residual = measurement - predicted
        self.position = predicted + residual * self.alpha
        self.velocity = _clamp_vector(
            self.velocity + residual * (self.beta / dt), 7.0
        )
        self.acceleration = _clamp_vector(
            self.acceleration + residual * (2.0 * self.gamma / (dt * dt)), 10.0
        )
        # Blend sensor-reported velocity to limit drift after abrupt turns.
        self.velocity = self.velocity * 0.7 + reported_velocity * 0.3
        self.confidence = min(1.0, self.confidence + 0.08)

    def forecast(self, seconds: float) -> Vec2 | None:
        if self.position is None:
            return None
        seconds = min(max(seconds, 0.0), 3.0)
        return self.position + self.velocity * seconds + self.acceleration * (
            0.5 * seconds * seconds
        )


@dataclass
class PathPredictionEnsemble:
    """Combines filtered dynamics, reported velocity, roads, and intercept utility."""

    grid: GridMap
    motion: AlphaBetaGammaPredictor = field(default_factory=AlphaBetaGammaPredictor)
    last_measurement: Vec2 | None = None
    reported_velocity: Vec2 = Vec2(0.0, 0.0)
    neural_model: NeuralTrajectoryModel | None = None

    def observe(self, position: Vec2, velocity: Vec2, dt: float) -> None:
        self.last_measurement = position
        self.reported_velocity = velocity
        self.motion.observe(position, dt, velocity)

    def _road_project(self, position: Vec2) -> Vec2:
        cell = self.grid.nearest_passable(self.grid.world_to_cell(position))
        return self.grid.cell_center(cell)

    def candidates(self) -> list[tuple[float, Vec2, float]]:
        """Return horizon, road-projected prediction, and confidence."""
        if self.motion.position is None:
            return []
        output = []
        neural_predictions: list[Vec2] | None = None
        if self.neural_model is not None:
            state = torch.tensor(
                [[
                    self.motion.position.x / self.grid.width,
                    self.motion.position.y / self.grid.height,
                    self.motion.velocity.x / 7.0,
                    self.motion.velocity.y / 7.0,
                    self.motion.acceleration.x / 10.0,
                    self.motion.acceleration.y / 10.0,
                ]],
                dtype=torch.float32,
            )
            with torch.no_grad():
                deltas = self.neural_model(state)[0].view(-1, 2)
            neural_predictions = [
                self.motion.position + _clamp_vector(
                    Vec2(float(delta[0]) * self.grid.width, float(delta[1]) * self.grid.height),
                    (self.motion.velocity.length() + 7.0 * PREDICTION_HORIZONS[index])
                    * PREDICTION_HORIZONS[index],
                )
                for index, delta in enumerate(deltas)
            ]
        for prediction_index, horizon in enumerate(PREDICTION_HORIZONS):
            dynamics = self.motion.forecast(horizon)
            linear = self.motion.position + self.reported_velocity * horizon
            blended = dynamics * self.motion.confidence + linear * (1 - self.motion.confidence)
            output.append((horizon, self._road_project(blended), self.motion.confidence))
            if neural_predictions is not None:
                # Straight-line acceleration is already handled exactly by the
                # analytical filter. During turns, offer the learned forecast as
                # a separate candidate; route utility chooses it only when its
                # travel timing is better than the analytical candidate.
                speed = self.motion.velocity.length()
                acceleration = self.motion.acceleration.length()
                cross = abs(
                    self.motion.velocity.x * self.motion.acceleration.y
                    - self.motion.velocity.y * self.motion.acceleration.x
                )
                turning = cross / max(speed * acceleration, 1e-6) > 0.15
                if turning:
                    output.append(
                        (
                            horizon,
                            self._road_project(neural_predictions[prediction_index]),
                            self.motion.confidence * 0.8,
                        )
                    )
        return output

    def best_intercept(self, car: PoliceCarAI) -> Vec2 | None:
        """Choose a reachable prediction using travel-time mismatch as utility."""
        start = self.grid.nearest_passable(self.grid.world_to_cell(car.position))
        best: tuple[float, Vec2] | None = None
        for horizon, position, confidence in self.candidates():
            goal = self.grid.world_to_cell(position)
            route = astar(self.grid, start, goal)
            if not route:
                continue
            travel_time = max(0, len(route) - 1) / max(car.max_speed, 0.1)
            timing_error = abs(travel_time - horizon)
            lateness = max(0.0, travel_time - horizon)
            utility_cost = timing_error + 1.5 * lateness - 0.25 * confidence * horizon
            if best is None or utility_cost < best[0]:
                best = (utility_cost, position)
        return best[1] if best else self.motion.position
