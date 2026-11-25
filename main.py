"""Text-free, resizable Tkinter front end for the headless simulation."""

from __future__ import annotations

import math
from pathlib import Path
import time
import tkinter as tk

from ai import GridMap, PoliceState, Vec2
from marl import ACTION_NAMES, DecentralizedController
from prediction import load_neural_trajectory_model
from simulation import PoliceSimulation


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
        self.root.title("Police Pursuit - CTDE MARL")
        self.root.minsize(720, 480)
        self.root.geometry("960x640")
        self.root.configure(bg="#0c1117")
        self.canvas = tk.Canvas(self.root, bg="#0c1117", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        # Diagnostics never cover gameplay and are hidden by default.
        self.status = tk.Label(
            self.root,
            bg="#0b1118",
            fg="#cbd5e1",
            anchor="w",
            padx=10,
            pady=4,
            font=("Segoe UI", 9),
        )

        self.grid = make_city()
        spawn_points = [
            Vec2(5.5, 15.5),
            Vec2(25.5, 15.5),
            Vec2(15.5, 5.5),
            Vec2(25.5, 5.5),
        ]
        patrol_routes = [
            [(5, 15), (5, 10), (10, 10), (10, 15)],
            [(25, 15), (20, 15), (20, 10), (25, 10)],
            [(15, 5), (10, 5), (10, 0), (15, 0)],
            [(25, 5), (20, 5), (20, 0), (25, 0)],
        ]
        project = Path(__file__).resolve().parent
        policy_path = project / "marl_policy.pt"
        trajectory_path = project / "trajectory_model.pt"
        if not policy_path.is_file() or not trajectory_path.is_file():
            raise FileNotFoundError(
                "Trained models missing. Run make_datasets.py, then train_models.py."
            )
        controller = DecentralizedController.load(policy_path, 4)
        trajectory_model = load_neural_trajectory_model(trajectory_path)
        self.simulation = PoliceSimulation(
            self.grid,
            controller,
            trajectory_model,
            spawn_points,
            patrol_routes,
        )

        self.keys: set[str] = set()
        self.debug_visible = False
        self.last_clock = time.perf_counter()
        self.accumulator = 0.0
        self.running = True
        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after_idle(self.tick)
        self.root.focus_force()

    def close(self):
        self.running = False
        self.root.destroy()

    def key_down(self, event):
        key = event.keysym.lower()
        self.keys.add(key)
        if key == "r":
            self.simulation.reset()
        elif key == "h":
            self.debug_visible = not self.debug_visible
            if self.debug_visible:
                self.status.pack(fill="x", side="bottom")
            else:
                self.status.pack_forget()
        elif key == "escape":
            self.close()

    def key_up(self, event):
        self.keys.discard(event.keysym.lower())

    def movement_input(self) -> Vec2:
        x = int("d" in self.keys or "right" in self.keys) - int(
            "a" in self.keys or "left" in self.keys
        )
        y = int("s" in self.keys or "down" in self.keys) - int(
            "w" in self.keys or "up" in self.keys
        )
        return Vec2(x, y)

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
            offset_x,
            offset_y,
            map_right,
            map_bottom,
            fill="#20262d",
            outline="#5a6673",
            width=2,
        )
        for y in range(HEIGHT):
            for x in range(WIDTH):
                x0, y0 = offset_x + x * scale, offset_y + y * scale
                if (x, y) in self.grid.blocked:
                    inset = max(1.0, scale * 0.07)
                    self.canvas.create_rectangle(
                        x0 + inset,
                        y0 + inset,
                        x0 + scale - inset,
                        y0 + scale - inset,
                        fill="#505a65",
                        outline="#687481",
                    )
                else:
                    self.canvas.create_rectangle(
                        x0, y0, x0 + scale, y0 + scale, outline="#29313a"
                    )

        if self.debug_visible:
            route_colors = ("#ffd166", "#06d6a0", "#c77dff", "#f78c6b")
            for index, police in enumerate(self.simulation.police_cars):
                if len(police.path) > 1:
                    points = [
                        coordinate
                        for cell in police.path
                        for coordinate in self.screen(self.grid.cell_center(cell))
                    ]
                    self.canvas.create_line(
                        *points,
                        fill=route_colors[index],
                        width=max(1, scale * 0.07),
                        smooth=True,
                    )

        self.draw_car(
            self.simulation.player_position,
            self.simulation.player_velocity,
            "#3aa7ff",
        )
        for police in self.simulation.police_cars:
            color = "#ff334f" if police.state != PoliceState.PATROL else "#ff7b32"
            self.draw_car(police.position, police.velocity, color)

        if self.simulation.arrested:
            self.canvas.create_rectangle(
                offset_x + 3,
                offset_y + 3,
                map_right - 3,
                map_bottom - 3,
                outline="#ff334f",
                width=max(4, scale * 0.18),
            )

        if self.debug_visible:
            states = "   ".join(
                f"P{index + 1} {car.state.name} / "
                f"{ACTION_NAMES[self.simulation.last_actions[index]]}"
                for index, car in enumerate(self.simulation.police_cars)
            )
            radio = "radio active" if self.simulation.radio else "radio quiet"
            self.status.configure(
                text=f"{states}   |   {radio}   |   H hides diagnostics"
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
            self.simulation.step(PHYSICS_DT, self.movement_input())
            self.accumulator -= PHYSICS_DT
            steps += 1
        self.draw()
        self.root.after(16, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    PoliceDemo().run()
