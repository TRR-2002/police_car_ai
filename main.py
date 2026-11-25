"""Smooth, resizable Tkinter demonstration for the production police AI."""

from __future__ import annotations

import math
from pathlib import Path
import time
import tkinter as tk

from ai import GridMap, PoliceCarAI, PoliceState, Vec2
from marl import ACTION_NAMES, DecentralizedController, action_offset
from prediction import PathPredictionEnsemble, load_neural_trajectory_model


WIDTH, HEIGHT = 30, 20
PHYSICS_DT = 1.0 / 60.0
MAX_FRAME_TIME = 0.10


def make_city() -> GridMap:
    blocked = {
        (x, y)
        for x in range(WIDTH)
        for y in range(HEIGHT)
        if x % 5 in {2, 3} and y % 5 in {1, 2, 3}
    }
    return GridMap(WIDTH, HEIGHT, blocked)


class PoliceDemo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Police Pursuit — CTDE MARL")
        self.root.minsize(720, 480)
        self.root.geometry("960x640")
        self.root.configure(bg="#0c1117")
        self.canvas = tk.Canvas(self.root, bg="#0c1117", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.grid = make_city()
        self.spawn_points = [
            Vec2(5.5, 15.5), Vec2(25.5, 15.5),
            Vec2(15.5, 5.5), Vec2(25.5, 5.5),
        ]
        self.patrol_routes = [
            [(5, 15), (5, 10), (10, 10), (10, 15)],
            [(25, 15), (20, 15), (20, 10), (25, 10)],
            [(15, 5), (10, 5), (10, 0), (15, 0)],
            [(25, 5), (20, 5), (20, 0), (25, 0)],
        ]

        root = Path(__file__).resolve().parent
        policy_path = root / "marl_policy.pt"
        trajectory_path = root / "trajectory_model.pt"
        if not policy_path.is_file() or not trajectory_path.is_file():
            raise FileNotFoundError(
                "Trained models missing. Run make_datasets.py, then train_models.py."
            )
        self.controller = DecentralizedController.load(policy_path, 4)
        self.trajectory_model = load_neural_trajectory_model(trajectory_path)

        self.keys: set[str] = set()
        self.hud_visible = True
        self.hud_deadline = time.perf_counter() + 4.0
        self.last_clock = time.perf_counter()
        self.accumulator = 0.0
        self.running = True
        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.reset_world()
        self.root.after_idle(self.tick)
        self.root.focus_force()

    def reset_world(self):
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
        self.dispatch_timer = 0.0
        self.hud_visible = True
        self.hud_deadline = time.perf_counter() + 4.0

    def close(self):
        self.running = False
        self.root.destroy()

    def key_down(self, event):
        key = event.keysym.lower()
        self.keys.add(key)
        if key == "r":
            self.reset_world()
        elif key == "h":
            self.hud_visible = not self.hud_visible
            self.hud_deadline = float("inf") if self.hud_visible else 0.0

    def key_up(self, event):
        self.keys.discard(event.keysym.lower())

    def move_player(self, dt: float):
        x = int("d" in self.keys or "right" in self.keys) - int(
            "a" in self.keys or "left" in self.keys
        )
        y = int("s" in self.keys or "down" in self.keys) - int(
            "w" in self.keys or "up" in self.keys
        )
        target_velocity = Vec2(x, y).normalized() * 4.5
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

    def update_physics(self, dt: float):
        self.move_player(dt)
        detections = [
            car.perception.detect(
                self.grid, car.position, self.player_position, self.player_velocity
            )
            for car in self.police_cars
        ]
        if any(visible or heard for visible, heard in detections):
            self.dispatch_timer = 3.0
        else:
            self.dispatch_timer = max(0.0, self.dispatch_timer - dt)
        radio_alert = self.dispatch_timer > 0.0

        positions = tuple(car.position for car in self.police_cars)
        decisions = []
        for index, police in enumerate(self.police_cars):
            action = self.controller.choose_action(
                police, self.player_position, self.player_velocity, index
            )
            self.last_actions[index] = action
            visible, heard = detections[index]
            if visible or heard:
                self.predictors[index].observe(
                    self.player_position, self.player_velocity, dt
                )
            decisions.append(
                (
                    action_offset(action, police, self.player_velocity),
                    self.predictors[index].best_intercept(police),
                    tuple(position for other, position in enumerate(positions) if other != index),
                )
            )
        for police, (offset, prediction, teammates) in zip(self.police_cars, decisions):
            police.update(
                dt,
                self.player_position,
                self.player_velocity,
                tactical_offset=offset,
                predicted_target=prediction,
                avoid_positions=teammates,
                radio_alert=radio_alert,
            )

    def viewport(self) -> tuple[float, float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = min(width / WIDTH, height / HEIGHT)
        return scale, (width - WIDTH * scale) / 2, (height - HEIGHT * scale) / 2

    def screen(self, point: Vec2) -> tuple[float, float]:
        scale, offset_x, offset_y = self.viewport()
        return offset_x + point.x * scale, offset_y + point.y * scale

    def draw_car(self, position: Vec2, velocity: Vec2, color: str):
        px, py = self.screen(position)
        scale, _, _ = self.viewport()
        angle = math.atan2(velocity.y, velocity.x) if velocity.length() > 0.05 else 0
        forward = Vec2(math.cos(angle), math.sin(angle))
        side = Vec2(-forward.y, forward.x)
        points = []
        for longitudinal, lateral in ((0.40, 0), (-0.30, 0.23), (-0.30, -0.23)):
            points.extend(
                (
                    px + (forward.x * longitudinal + side.x * lateral) * scale,
                    py + (forward.y * longitudinal + side.y * lateral) * scale,
                )
            )
        self.canvas.create_polygon(points, fill=color, outline="#f4f7fb", width=1)

    def draw(self):
        self.canvas.delete("all")
        scale, offset_x, offset_y = self.viewport()
        map_right, map_bottom = offset_x + WIDTH * scale, offset_y + HEIGHT * scale
        self.canvas.create_rectangle(
            offset_x, offset_y, map_right, map_bottom,
            fill="#20262d", outline="#5a6673", width=2
        )
        for y in range(HEIGHT):
            for x in range(WIDTH):
                x0, y0 = offset_x + x * scale, offset_y + y * scale
                if (x, y) in self.grid.blocked:
                    inset = max(1.0, scale * 0.07)
                    self.canvas.create_rectangle(
                        x0 + inset, y0 + inset,
                        x0 + scale - inset, y0 + scale - inset,
                        fill="#505a65", outline="#687481"
                    )
                else:
                    self.canvas.create_rectangle(
                        x0, y0, x0 + scale, y0 + scale, outline="#29313a"
                    )

        route_colors = ("#ffd166", "#06d6a0", "#c77dff", "#f78c6b")
        for index, police in enumerate(self.police_cars):
            if len(police.path) > 1:
                points = [
                    coordinate
                    for cell in police.path
                    for coordinate in self.screen(self.grid.cell_center(cell))
                ]
                self.canvas.create_line(
                    *points, fill=route_colors[index], width=max(1, scale * 0.07), smooth=True
                )
        self.draw_car(self.player_position, self.player_velocity, "#3aa7ff")
        for police in self.police_cars:
            color = "#ff334f" if police.state != PoliceState.PATROL else "#ff7b32"
            self.draw_car(police.position, police.velocity, color)

        arrested = any(car.state == PoliceState.ARREST for car in self.police_cars)
        if arrested:
            self.canvas.create_rectangle(
                offset_x + 3, offset_y + 3, map_right - 3, map_bottom - 3,
                outline="#ff334f", width=max(4, scale * 0.18)
            )

        if self.hud_visible and time.perf_counter() <= self.hud_deadline:
            states = "   ".join(
                f"P{index + 1} {car.state.name} · {ACTION_NAMES[self.last_actions[index]]}"
                for index, car in enumerate(self.police_cars)
            )
            self.canvas.create_rectangle(
                offset_x + 12, offset_y + 12,
                min(map_right - 12, offset_x + 700), offset_y + 68,
                fill="#0b1118", outline="#495562", width=1
            )
            self.canvas.create_text(
                offset_x + 24, offset_y + 24, anchor="nw",
                fill="#f1f5f9", font=("Segoe UI", 10, "bold"), text=states
            )
            self.canvas.create_text(
                offset_x + 24, offset_y + 47, anchor="nw",
                fill="#9eabb8", font=("Segoe UI", 9),
                text="WASD/arrows drive · R reset · H HUD · status auto-hides"
            )

    def tick(self):
        if not self.running:
            return
        now = time.perf_counter()
        frame_time = min(MAX_FRAME_TIME, now - self.last_clock)
        self.last_clock = now
        self.accumulator += frame_time
        steps = 0
        while self.accumulator >= PHYSICS_DT and steps < 6:
            self.update_physics(PHYSICS_DT)
            self.accumulator -= PHYSICS_DT
            steps += 1
        if self.hud_visible and now > self.hud_deadline:
            self.hud_visible = False
        self.draw()
        self.root.after(8, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    PoliceDemo().run()
