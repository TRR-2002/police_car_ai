"""Classical police-car AI: perception, FSM, A*, and steering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
import math


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        length = self.length()
        return Vec2(0.0, 0.0) if length == 0 else self * (1.0 / length)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).length()


Cell = tuple[int, int]


class GridMap:
    def __init__(self, width: int, height: int, blocked: set[Cell] | None = None):
        self.width = width
        self.height = height
        self.blocked = blocked or set()

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.blocked

    def neighbors(self, cell: Cell) -> list[Cell]:
        x, y = cell
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [candidate for candidate in candidates if self.passable(candidate)]

    def world_to_cell(self, position: Vec2) -> Cell:
        return (
            min(self.width - 1, max(0, int(position.x))),
            min(self.height - 1, max(0, int(position.y))),
        )

    def contains_world(self, position: Vec2, radius: float = 0.0) -> bool:
        return (
            radius <= position.x <= self.width - radius
            and radius <= position.y <= self.height - radius
        )

    def is_drivable_position(self, position: Vec2, radius: float = 0.20) -> bool:
        """Check world bounds and all corners of a vehicle-sized footprint."""
        if not self.contains_world(position, radius):
            return False
        probes = (
            Vec2(position.x - radius, position.y - radius),
            Vec2(position.x + radius, position.y - radius),
            Vec2(position.x - radius, position.y + radius),
            Vec2(position.x + radius, position.y + radius),
        )
        return all(self.world_to_cell(probe) not in self.blocked for probe in probes)

    def move_with_slide(
        self, position: Vec2, displacement: Vec2, radius: float = 0.20
    ) -> tuple[Vec2, Vec2]:
        """Move with independent-axis collision, returning position and blocked axes."""
        target = position + displacement
        if self.is_drivable_position(target, radius):
            return target, Vec2(0.0, 0.0)
        result = position
        blocked_x = 1.0
        blocked_y = 1.0
        x_target = Vec2(target.x, position.y)
        if self.is_drivable_position(x_target, radius):
            result = x_target
            blocked_x = 0.0
        y_target = Vec2(result.x, target.y)
        if self.is_drivable_position(y_target, radius):
            result = y_target
            blocked_y = 0.0
        return result, Vec2(blocked_x, blocked_y)

    @staticmethod
    def cell_center(cell: Cell) -> Vec2:
        return Vec2(cell[0] + 0.5, cell[1] + 0.5)

    def nearest_passable(self, cell: Cell) -> Cell:
        if self.passable(cell):
            return cell
        frontier = [cell]
        visited = {cell}
        while frontier:
            current = frontier.pop(0)
            x, y = current
            for candidate in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if candidate in visited or not self.in_bounds(candidate):
                    continue
                if self.passable(candidate):
                    return candidate
                visited.add(candidate)
                frontier.append(candidate)
        raise ValueError("Map contains no passable cells")

    def has_line_of_sight(self, start: Vec2, end: Vec2) -> bool:
        """Sample a ray densely enough to detect every crossed grid obstacle."""
        distance = start.distance_to(end)
        steps = max(1, math.ceil(distance * 4))
        for index in range(1, steps):
            amount = index / steps
            point = start + (end - start) * amount
            if self.world_to_cell(point) in self.blocked:
                return False
        return True


def astar(grid: GridMap, start: Cell, goal: Cell) -> list[Cell] | None:
    """Return an optimal four-directional path including start and goal."""
    if not grid.passable(start) or not grid.passable(goal):
        return None
    frontier: list[tuple[float, int, Cell]] = [(0.0, 0, start)]
    came_from: dict[Cell, Cell | None] = {start: None}
    cost_so_far = {start: 0.0}
    order = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in grid.neighbors(current):
            new_cost = cost_so_far[current] + 1.0
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heuristic = abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
                order += 1
                heapq.heappush(frontier, (new_cost + heuristic, order, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return None
    path = []
    current: Cell | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    return list(reversed(path))


class PoliceState(Enum):
    PATROL = auto()
    CHASE = auto()
    SEARCH = auto()
    ARREST = auto()


@dataclass
class Perception:
    sight_range: float = 16.0
    hearing_range: float = 8.0
    noisy_speed: float = 2.0

    def detect(
        self,
        grid: GridMap,
        police_position: Vec2,
        player_position: Vec2,
        player_velocity: Vec2,
    ) -> tuple[bool, bool]:
        distance = police_position.distance_to(player_position)
        visible = distance <= self.sight_range and grid.has_line_of_sight(
            police_position, player_position
        )
        heard = distance <= self.hearing_range and player_velocity.length() >= self.noisy_speed
        return visible, heard


@dataclass
class PoliceCarAI:
    grid: GridMap
    position: Vec2
    patrol_points: list[Cell]
    max_speed: float = 4.9
    acceleration: float = 10.0
    arrest_range: float = 0.7
    state: PoliceState = PoliceState.PATROL
    velocity: Vec2 = Vec2(0.0, 0.0)
    perception: Perception = field(default_factory=Perception)
    path: list[Cell] = field(default_factory=list)
    patrol_index: int = 0
    last_seen: Vec2 | None = None
    lost_time: float = 0.0
    search_time: float = 0.0
    capture_progress: float = 0.0
    replan_time: float = 0.0
    target_cell: Cell | None = None

    def _transition(self, state: PoliceState) -> None:
        if self.state == state:
            return
        self.state = state
        self.path.clear()
        self.target_cell = None
        if state == PoliceState.SEARCH:
            self.search_time = 0.0
        if state == PoliceState.CHASE:
            self.lost_time = 0.0

    def _plan(self, destination: Vec2) -> None:
        start = self.grid.nearest_passable(self.grid.world_to_cell(self.position))
        goal = self.grid.nearest_passable(self.grid.world_to_cell(destination))
        if goal != self.target_cell or not self.path or self.replan_time <= 0:
            self.path = astar(self.grid, start, goal) or []
            if self.path and self.path[0] == start:
                self.path.pop(0)
            self.target_cell = goal
            self.replan_time = 0.35

    def _steer(self, dt: float, avoid_positions: tuple[Vec2, ...] = ()) -> None:
        if not self.path:
            desired_velocity = Vec2(0.0, 0.0)
        else:
            waypoint = self.grid.cell_center(self.path[0])
            if self.position.distance_to(waypoint) < 0.22:
                self.path.pop(0)
                if not self.path:
                    desired_velocity = Vec2(0.0, 0.0)
                else:
                    waypoint = self.grid.cell_center(self.path[0])
                    desired_velocity = (waypoint - self.position).normalized() * self.max_speed
            else:
                desired_velocity = (waypoint - self.position).normalized() * self.max_speed

        # Local reciprocal-style separation prevents several police cars stacking.
        for other in avoid_positions:
            away = self.position - other
            distance = away.length()
            if 0.001 < distance < 1.15:
                desired_velocity = desired_velocity + away.normalized() * (
                    (1.15 - distance) * self.max_speed * 1.8
                )
        if desired_velocity.length() > self.max_speed:
            desired_velocity = desired_velocity.normalized() * self.max_speed

        steering = desired_velocity - self.velocity
        maximum_change = self.acceleration * dt
        if steering.length() > maximum_change:
            steering = steering.normalized() * maximum_change
        self.velocity = self.velocity + steering
        self.position, blocked_axes = self.grid.move_with_slide(
            self.position, self.velocity * dt, radius=0.22
        )
        if blocked_axes.x:
            self.velocity = Vec2(0.0, self.velocity.y)
        if blocked_axes.y:
            self.velocity = Vec2(self.velocity.x, 0.0)
        if blocked_axes.x and blocked_axes.y:
            self.path.clear()

    def update(
        self,
        dt: float,
        player_position: Vec2,
        player_velocity: Vec2,
        tactical_offset: Vec2 = Vec2(0.0, 0.0),
        predicted_target: Vec2 | None = None,
        avoid_positions: tuple[Vec2, ...] = (),
        radio_position: Vec2 | None = None,
        radio_velocity: Vec2 = Vec2(0.0, 0.0),
        search_target: Vec2 | None = None,
    ) -> PoliceState:
        dt = min(max(dt, 1e-4), 0.1)
        visible, heard = self.perception.detect(
            self.grid, self.position, player_position, player_velocity
        )
        local_contact = visible or heard
        contact = local_contact or radio_position is not None
        observed_position = player_position if local_contact else radio_position
        observed_velocity = player_velocity if local_contact else radio_velocity
        distance = self.position.distance_to(player_position)
        self.replan_time -= dt

        if self.state == PoliceState.PATROL:
            if contact:
                self.last_seen = observed_position
                self._transition(PoliceState.CHASE)
            else:
                destination = self.grid.cell_center(self.patrol_points[self.patrol_index])
                if self.position.distance_to(destination) < 0.5:
                    self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
                    destination = self.grid.cell_center(self.patrol_points[self.patrol_index])
                self._plan(destination)

        if self.state == PoliceState.CHASE:
            # Capture has one clear rule: crossing inside arrest range.
            if distance <= self.arrest_range:
                self.capture_progress = 0.0
                self._transition(PoliceState.ARREST)
            else:
                self.capture_progress = 0.0
            if self.state == PoliceState.CHASE and contact:
                self.lost_time = 0.0
                self.last_seen = observed_position
                # Lead a moving target, with prediction bounded to avoid wild routes.
                lead_time = min(1.2, distance / max(self.max_speed, 0.1))
                predicted = (
                    predicted_target
                    if predicted_target is not None
                    else observed_position + observed_velocity * lead_time
                ) + tactical_offset
                self._plan(predicted)
            elif self.state == PoliceState.CHASE:
                self.lost_time += dt
                if self.last_seen is not None:
                    self._plan(self.last_seen)
                if self.lost_time >= 1.0:
                    self._transition(PoliceState.SEARCH)

        elif self.state == PoliceState.SEARCH:
            if contact:
                self.last_seen = observed_position
                self._transition(PoliceState.CHASE)
            else:
                self.search_time += dt
                destination = search_target if search_target is not None else self.last_seen
                if destination is not None:
                    self._plan(destination)
                if self.search_time >= 5.0:
                    self._transition(PoliceState.PATROL)

        elif self.state == PoliceState.ARREST:
            self.path.clear()
            if distance > self.arrest_range * 1.7:
                self._transition(PoliceState.CHASE)

        self._steer(dt, avoid_positions)
        return self.state
