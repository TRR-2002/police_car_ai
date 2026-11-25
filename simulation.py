"""Headless fixed-step police pursuit simulation.

This module owns game state and model scheduling.  The Tk UI in main.py is only
an input/render adapter, which makes pursuit behavior deterministic and directly
testable without opening a window.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai import GridMap, PoliceCarAI, PoliceState, Vec2
from marl import DecentralizedController, action_offset
from prediction import PathPredictionEnsemble, NeuralTrajectoryModel


@dataclass
class RadioReport:
    position: Vec2
    velocity: Vec2
    remaining: float = 3.0


class PoliceSimulation:
    """Authoritative game state with 60 Hz physics and 5 Hz AI decisions."""

    decision_interval = 0.20

    def __init__(
        self,
        grid: GridMap,
        controller: DecentralizedController,
        trajectory_model: NeuralTrajectoryModel | None,
        spawn_points: list[Vec2],
        patrol_routes: list[list[tuple[int, int]]],
    ):
        self.grid = grid
        self.controller = controller
        self.trajectory_model = trajectory_model
        self.spawn_points = spawn_points
        self.patrol_routes = patrol_routes
        self.reset()

    def reset(self) -> None:
        self.player_position = Vec2(5.5, 5.5)
        self.player_velocity = Vec2(0.0, 0.0)
        self.police_cars = [
            PoliceCarAI(self.grid, spawn, route)
            for spawn, route in zip(self.spawn_points, self.patrol_routes)
        ]
        self.predictors = [
            PathPredictionEnsemble(self.grid, neural_model=self.trajectory_model)
            for _ in self.police_cars
        ]
        self.last_actions = [0] * len(self.police_cars)
        self.tactical_offsets = [Vec2(0.0, 0.0)] * len(self.police_cars)
        self.intercepts: list[Vec2 | None] = [None] * len(self.police_cars)
        self.radio: RadioReport | None = None
        self.decision_remaining = 0.0
        self.elapsed = 0.0

    @property
    def arrested(self) -> bool:
        return any(car.state == PoliceState.ARREST for car in self.police_cars)

    def _move_player(self, movement: Vec2, dt: float) -> None:
        target_velocity = movement.normalized() * 4.5
        blend = min(1.0, 9.0 * dt)
        self.player_velocity = self.player_velocity + (
            target_velocity - self.player_velocity
        ) * blend
        self.player_position, blocked = self.grid.move_with_slide(
            self.player_position, self.player_velocity * dt, radius=0.24
        )
        if blocked.x:
            self.player_velocity = Vec2(0.0, self.player_velocity.y)
        if blocked.y:
            self.player_velocity = Vec2(self.player_velocity.x, 0.0)

    def _update_perception(self, dt: float) -> list[tuple[bool, bool]]:
        detections = [
            car.perception.detect(
                self.grid, car.position, self.player_position, self.player_velocity
            )
            for car in self.police_cars
        ]
        if any(visible or heard for visible, heard in detections):
            self.radio = RadioReport(self.player_position, self.player_velocity)
        elif self.radio is not None:
            self.radio.remaining -= dt
            if self.radio.remaining <= 0:
                self.radio = None
        return detections

    def _observed_motion(
        self, detection: tuple[bool, bool]
    ) -> tuple[Vec2, Vec2, bool]:
        if detection[0] or detection[1]:
            return self.player_position, self.player_velocity, True
        if self.radio is not None:
            return self.radio.position, self.radio.velocity, True
        return self.player_position, self.player_velocity, False

    def _update_decisions(
        self, detections: list[tuple[bool, bool]], dt: float
    ) -> None:
        for index, (police, predictor, detection) in enumerate(
            zip(self.police_cars, self.predictors, detections)
        ):
            observed_position, observed_velocity, has_report = self._observed_motion(
                detection
            )
            if has_report:
                predictor.observe(observed_position, observed_velocity, dt)
            if self.decision_remaining > 0 or not has_report:
                continue
            action = self.controller.choose_action(
                police, observed_position, observed_velocity, index
            )
            self.last_actions[index] = action
            self.tactical_offsets[index] = action_offset(
                action, police, observed_velocity
            )
            self.intercepts[index] = predictor.best_intercept(police)

    def step(self, dt: float, movement: Vec2 = Vec2(0.0, 0.0)) -> None:
        dt = min(max(dt, 1e-4), 0.1)
        self.elapsed += dt
        self._move_player(movement, dt)
        detections = self._update_perception(dt)
        self.decision_remaining -= dt
        self._update_decisions(detections, dt)
        if self.decision_remaining <= 0:
            self.decision_remaining += self.decision_interval

        positions = tuple(car.position for car in self.police_cars)
        containment_support = sum(
            position.distance_to(self.player_position) <= 1.5 for position in positions
        )
        radio_position = self.radio.position if self.radio else None
        radio_velocity = self.radio.velocity if self.radio else Vec2(0.0, 0.0)
        for index, police in enumerate(self.police_cars):
            teammates = tuple(
                position for other, position in enumerate(positions) if other != index
            )
            police.update(
                dt,
                self.player_position,
                self.player_velocity,
                tactical_offset=self.tactical_offsets[index],
                predicted_target=self.intercepts[index],
                avoid_positions=teammates,
                radio_position=radio_position,
                radio_velocity=radio_velocity,
                containment_support=containment_support,
            )

