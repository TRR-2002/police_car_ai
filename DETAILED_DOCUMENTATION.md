# Police Car AI — Complete Technical Documentation

> Project location: `G:\Desktop\AI_dev\police_car_ai`  
> Runtime: Python 3.12, PyTorch 2.7.1 CPU, Tkinter  
> Architecture: classical navigation + finite-state control + CTDE multi-agent reinforcement learning + learned trajectory prediction

## 1. Purpose and scope

This project is a small but complete reference implementation of cooperative police-car AI. Four police agents patrol a grid city, detect a player through sight or hearing, broadcast a short-lived radio alert, predict the player's future path, plan routes around buildings, select cooperative roles, steer without stacking on teammates, search after losing contact, and perform a sustained-proximity arrest.

The project deliberately separates responsibilities:

- **Classical game AI** handles dependable state transitions, perception, collision-safe motion, and A* navigation.
- **Multi-agent reinforcement learning (MARL)** chooses cooperative tactical roles.
- **Supervised trajectory learning** predicts several future player positions.
- **Analytical prediction** provides a stable fallback and ensemble partner for the neural predictor.
- **Tkinter** supplies a lightweight demonstration interface; it is not the recommended rendering layer for a shipped game.

This is a production-oriented prototype, not a finished commercial vehicle system. The algorithms are intentionally readable and engine-independent so they can be ported into Unity, Godot, Unreal Engine, or a custom engine.

## 2. What the AI can do

Each police car can:

1. Patrol an independent district.
2. Detect the player within a 16-cell sight radius when line of sight is clear.
3. Hear a sufficiently fast/noisy player within 8 cells.
4. Share a three-second team radio alert after any unit detects the player.
5. Transition through `PATROL`, `CHASE`, `SEARCH`, and `ARREST` states.
6. Select a learned team role: direct pursuit, lead pursuit, left flank, right flank, or rear block.
7. Estimate player position, velocity, and acceleration with an alpha-beta-gamma filter.
8. Use a learned neural model to predict five future positions.
9. Blend neural, filtered-dynamics, and linear predictions.
10. Project predictions onto drivable roads.
11. Evaluate reachable interception points with A* travel-cost utility.
12. Follow A* routes with acceleration-limited steering.
13. Separate from nearby police vehicles to reduce stacking.
14. Slide along buildings and map boundaries instead of leaving the world.
15. Require sustained close proximity before completing an arrest.

## 3. Project filesystem

```text
police_car_ai/
├── ai.py                         Core vectors, grid, A*, perception, FSM, steering
├── marl.py                       Decentralized actor, centralized critic, CTDE training
├── prediction.py                 Analytical and learned trajectory prediction
├── main.py                       Smooth, resizable Tkinter simulation
├── make_datasets.py              Reproducible CSV dataset generator
├── train_models.py               Supervised training + CTDE fine-tuning pipeline
├── train_marl.py                 Small standalone CTDE training entry point
├── marl_policy.pt                Trained decentralized actor + training-only critic
├── trajectory_model.pt           Trained neural trajectory predictor
├── training_report.json          Final training and held-out test metrics
├── pytest.ini                    Test discovery and import configuration
├── README.md                     Short setup and usage guide
├── DETAILED_DOCUMENTATION.md     This complete manual
├── build_documentation.py        Rebuilds full source listings in this manual
├── data/
│   ├── team_roles.csv            50,000 local-observation/role examples
│   └── trajectories.csv          50,000 motion/future-position examples
└── tests/
    ├── test_ai.py                Navigation, perception, FSM, collision tests
    ├── test_marl.py              CTDE and local-runtime-policy tests
    ├── test_prediction.py        Prediction, road projection, intercept tests
    └── test_training_data.py     Dataset and network-shape tests
```

Generated cache directories such as `__pycache__/` and `.pytest_cache/` are not source code and should not be committed. The `.pt` files are binary PyTorch checkpoints; the CSV files are generated data rather than handwritten source.

## 4. Quick start

The project reuses the virtual environment created beside `mini_gpt`.

### Run the trained simulation

```powershell
cd G:\Desktop\AI_dev\police_car_ai
..\mini_gpt\.venv\Scripts\python.exe main.py
```

### Controls

| Control | Effect |
|---|---|
| `W`, `A`, `S`, `D` | Drive the blue player car |
| Arrow keys | Alternative player controls |
| `R` | Reset the entire simulation and all predictors |
| `H` | Toggle the HUD |
| Window close button | Stop the update loop and close safely |

The HUD automatically disappears after four seconds. A red border means at least one police unit has completed an arrest.

### Regenerate datasets and retrain

```powershell
cd G:\Desktop\AI_dev\police_car_ai
..\mini_gpt\.venv\Scripts\python.exe make_datasets.py --rows 50000
..\mini_gpt\.venv\Scripts\python.exe train_models.py --role-epochs 60 --trajectory-epochs 90 --ctde-iterations 2500
```

### Run tests

```powershell
cd G:\Desktop\AI_dev\police_car_ai
..\mini_gpt\.venv\Scripts\python.exe -m pytest -q
```

Expected current result: `16 passed`.

## 5. Runtime architecture

### 5.1 Per-frame data flow

```text
Keyboard input
    ↓
Player fixed-step movement and collision sliding
    ↓
Per-car sight/hearing checks
    ↓
Three-second radio dispatch if any unit has contact
    ↓
Each car independently constructs its 11-value local observation
    ↓
Shared decentralized actor chooses one tactical action per car
    ↓
Each car updates its own trajectory predictor when it has local contact
    ↓
Prediction ensemble proposes road-valid future positions
    ↓
A*-cost utility selects a reachable intercept
    ↓
FSM chooses patrol/chase/search/arrest behavior
    ↓
A* route planning + acceleration-limited steering + teammate separation
    ↓
Radius-aware collision and world-boundary resolution
    ↓
Responsive rendering
```

### 5.2 Fixed-step simulation

`main.py` uses a 60 Hz physics step (`1/60` second) and an accumulator. Rendering is requested approximately every 8 ms, but game logic advances only in fixed increments. Frame gaps are clamped to 100 ms, and at most six physics steps are processed per UI tick. This avoids unstable movement after window dragging, breakpoints, or temporary stalls.

### 5.3 Responsive viewport

The renderer chooses the smaller of window-width/map-width and window-height/map-height. The map is centered with letterboxing, preserving its aspect ratio at normal and maximized sizes. All roads, buildings, routes, cars, borders, and HUD coordinates are transformed through the same viewport.

## 6. Core game AI (`ai.py`)

### 6.1 `Vec2`

`Vec2` is an immutable two-dimensional vector supporting addition, subtraction, scalar multiplication, length, normalization, and distance. Keeping this type engine-independent makes the AI easy to port.

### 6.2 `GridMap`

`GridMap` stores dimensions and blocked cells. It provides:

- bounds checking;
- passability tests;
- four-directional neighbors;
- world-to-grid conversion;
- cell centers;
- nearest-road projection;
- line-of-sight ray sampling;
- radius-aware vehicle footprint checks;
- independent-axis movement for wall and corner sliding.

The radius-aware check probes all four corners of a vehicle footprint. It prevents a vehicle center from being technically on a road while its body clips a building or world boundary.

### 6.3 A* pathfinding

`astar()` uses Manhattan distance as an admissible heuristic for four-directional movement. It returns an optimal path containing the start and goal or `None` if the destination is unreachable.

Complexity is approximately `O(V log V)` with the heap implementation, where `V` is the number of explored cells. In a production engine, replace it with that engine's navmesh or road-graph query while retaining the same AI-level interface.

### 6.4 Perception

Sight requires both distance within 16 cells and unobstructed line of sight. Hearing requires distance within 8 cells and player speed at or above the configured noise threshold. Any local detection refreshes a three-second team dispatch in `main.py`.

### 6.5 Finite-state machine

| State | Entry/behavior | Exit |
|---|---|---|
| `PATROL` | Follow cyclic district waypoints | Local detection or radio alert → `CHASE` |
| `CHASE` | Predict, intercept, replan, and accumulate capture progress | Lost contact → `SEARCH`; sustained proximity → `ARREST` |
| `SEARCH` | Move toward the last observed position for five seconds | Contact → `CHASE`; timeout → `PATROL` |
| `ARREST` | Stop route following | Player escapes beyond threshold → `CHASE` |

Arrest is not instantaneous. The player must remain within arrest range while moving slowly for 0.65 seconds. Capture progress decays when these conditions are lost.

### 6.6 Steering and collision avoidance

The controller converts the next route waypoint into desired velocity, limits acceleration, adds local separation from teammates, limits final speed, and resolves movement with radius-aware collision sliding. If both axes are blocked, the route is cleared so it can be replanned.

## 7. MARL and CTDE (`marl.py`)

### 7.1 Why CTDE

Centralized training with decentralized execution permits a critic to use global team state during training without giving runtime agents impossible global knowledge. At runtime, each car calls the same actor using only its local observation and identity.

### 7.2 Local actor observation

The actor receives 11 normalized values:

| Index | Value |
|---:|---|
| 0–1 | Player position relative to this car, normalized by map size |
| 2–3 | Player velocity, normalized by expected maximum speed |
| 4–5 | This police car's velocity, normalized by police maximum speed |
| 6 | Detection/contact indicator |
| 7–10 | One-hot identity for the four police agents |

The shared actor is `11 → 64 → 64 → 5`, with `Tanh` activations. It produces logits for five actions:

1. direct pursuit;
2. lead pursuit;
3. left flank;
4. right flank;
5. rear block.

### 7.3 Centralized critic state

The critic receives 20 values: four player values (position and velocity) plus four values for each of four police cars. Its network is `20 → 128 → 64 → 1`. It is saved for reproducibility but is not loaded by the game.

### 7.4 Cooperative reward

Training uses a global reward based on correct complementary roles and a penalty when agents duplicate roles. Actor loss uses the global advantage and entropy regularization. Critic loss is mean squared value error. Combined gradients are clipped to norm 1.0.

### 7.5 Runtime execution

`DecentralizedController` loads only `actor_state`. Every car independently creates its local observation, runs the shared actor, selects `argmax`, and converts the action into a tactical offset. No centralized critic is used during play.

## 8. Prediction system (`prediction.py`)

### 8.1 Analytical motion filter

`AlphaBetaGammaPredictor` estimates position, velocity, and acceleration. It predicts forward, compares the prediction with the newest measurement, applies residual corrections, clamps extreme velocity/acceleration, and blends reported velocity to control drift.

### 8.2 Neural trajectory model

The neural model receives six normalized values:

```text
x, y, velocity_x, velocity_y, acceleration_x, acceleration_y
```

Its network is `6 → 64 → 64 → 10` with ReLU activations. Ten outputs represent X/Y displacement at five horizons:

```text
0.35 s, 0.70 s, 1.10 s, 1.60 s, 2.20 s
```

### 8.3 Prediction ensemble

The runtime predictor blends:

- alpha-beta-gamma constant-acceleration forecast;
- sensor-reported linear-velocity forecast;
- neural future-displacement forecast.

The neural model receives 35% ensemble weight, while analytical predictions retain 65%. This intentionally limits neural extrapolation errors outside the generated training distribution.

### 8.4 Road projection and intercept utility

Every future prediction is projected to its nearest passable cell. For each candidate, A* estimates route length and travel time. Utility penalizes timing mismatch and arriving after the player, while rewarding confident longer-horizon interception. The lowest-cost reachable candidate becomes the chase target.

## 9. Dataset specifications

### 9.1 `data/team_roles.csv`

Current size: 50,000 rows. Each row contains 11 local actor observations plus the expert action label.

```text
obs_0,obs_1,...,obs_10,action
```

Generation randomizes player position/velocity, car position/velocity, detection status, and agent identity. Labels assign complementary direct, lead, left-flank, and right-flank responsibilities.

### 9.2 `data/trajectories.csv`

Current size: 50,000 rows. Inputs are normalized position, velocity, and acceleration. Targets are normalized X/Y displacement at the five forecast horizons.

```text
x,y,vx,vy,ax,ay,
delta_x_0_35,delta_y_0_35,
delta_x_0_7,delta_y_0_7,
delta_x_1_1,delta_y_1_1,
delta_x_1_6,delta_y_1_6,
delta_x_2_2,delta_y_2_2
```

Examples include acceleration, braking, and turning acceleration. Future positions are clipped to world bounds.

### 9.3 Dataset limitations

The generated datasets do not reproduce real vehicle tire forces, traffic, destructible obstacles, network latency, or player-specific driving styles. Before shipping in a real game, collect trajectories from the actual game physics and retrain on those logs. Synthetic data is best treated as bootstrapping data.

## 10. Training pipeline (`train_models.py`)

1. Load both CSV files.
2. Shuffle deterministically under the selected torch seed.
3. Split each dataset into 80% training, 10% validation, and 10% held-out test data.
4. Supervise the decentralized role actor with cross-entropy loss.
5. Train the trajectory model with AdamW and MSE.
6. Clip trajectory gradients to norm 1.0.
7. Early-stop trajectory training after 12 stale validation epochs.
8. Restore the best validation checkpoint.
9. Fine-tune the pretrained actor with centralized actor-critic training.
10. Save both binary checkpoints and `training_report.json`.

Current recorded metrics:

| Metric | Value |
|---|---:|
| Role rows | 50,000 |
| Trajectory rows | 50,000 |
| Role validation accuracy | 1.0000 |
| Role held-out test accuracy | 1.0000 |
| Trajectory validation MSE | 0.0000034386 |
| Trajectory held-out test MSE | 0.0000036374 |
| CTDE final role accuracy | 1.0000 |

The perfect role score reflects a deliberately simple expert role assignment and should not be confused with proof that every pursuit scenario is solved.

## 11. Checkpoint contents

### 11.1 `marl_policy.pt`

Contains:

- `actor_state` — runtime decentralized actor;
- `critic_state` — training-only centralized critic;
- `num_agents` — expected team size;
- textual training/execution metadata.

### 11.2 `trajectory_model.pt`

Contains:

- `model_state` — trajectory network parameters;
- `validation_mse` — best validation result;
- `training_rows` — number of rows in the training split.

Never load untrusted PyTorch checkpoints. PyTorch serialization can execute unsafe pickle content when configured incorrectly. This project loads locally generated files with `weights_only=False` because metadata is present; use a safer deployment format such as ONNX or safetensors for distributed builds.

## 12. Tests

The suite verifies:

- optimal A* routing around obstacles;
- unreachable-goal behavior;
- sight, occlusion, hearing, and range;
- patrol/chase/search/arrest transitions;
- pursuit progress;
- sustained arrest and escape transitions;
- solid world edges and corner sliding;
- local actor versus global critic dimensions;
- decentralized checkpoint loading;
- CTDE checkpoint metadata;
- acceleration-aware forecasting;
- road-valid prediction projection;
- reachable utility interception;
- teammate separation steering;
- generated dataset dimensions;
- network input/output dimensions.

## 13. How to integrate this AI into an actual game

### 13.1 Do not copy the demo renderer

Keep the algorithms but replace Tkinter and `GridMap` with your engine's systems. The engine should own physics, collision, rendering, animation, audio, and networking. The AI should request destinations and steering intentions rather than directly teleporting transforms.

### 13.2 Recommended engine-facing interfaces

Create adapters equivalent to:

```text
NavigationProvider
  find_path(start, goal) -> waypoints
  nearest_drivable(position) -> position
  path_cost(start, goal) -> float

VehicleController
  position() -> Vector2/Vector3
  velocity() -> Vector2/Vector3
  apply_throttle(value)
  apply_brake(value)
  apply_steering(value)

PerceptionProvider
  has_line_of_sight(target) -> bool
  can_hear(target) -> bool

RadioBus
  publish_last_seen(position, timestamp)
  latest_alert() -> optional alert

PolicyRunner
  choose_role(local_observation) -> role
  predict_trajectory(motion_state) -> future points
```

This boundary lets you test AI separately and replace navigation or inference without rewriting behavior code.

### 13.3 Coordinate conversion

The demo uses X/Y grid coordinates. A 3D driving game normally uses X/Z ground coordinates and Y for elevation. Convert engine positions into a local normalized AI space before inference, and convert predictions back afterward. Preserve the training normalization constants or retrain for the new coordinate scale.

### 13.4 Vehicle control

Do not directly set vehicle velocity in a realistic game. Convert the next path/intercept point into:

1. desired heading;
2. signed heading error;
3. steering input through a PID/controller curve;
4. target speed based on curvature, traffic, visibility, and distance;
5. throttle/brake commands.

Add recovery logic for reversing, being overturned, blocked routes, and repeated collision.

### 13.5 Navigation

For city driving, a road-lane graph is often better than a free navmesh. Nodes should describe lanes, intersections, legal directions, speed limits, turn costs, roadblocks, and temporary hazards. Use hierarchical planning for large worlds: district route first, then local lane path.

### 13.6 Perception

Replace distance-only sight with a sensor model:

- field-of-view cone;
- one or more raycasts to vehicle bounds;
- lighting/weather visibility modifiers;
- occlusion by buildings and traffic;
- recognition delay;
- uncertain/noisy last-known position;
- separate hearing events generated by collisions, engines, horns, and gunfire.

Never give the decentralized actor perfect player coordinates unless a local sensor or radio report legitimately supplies them.

### 13.7 Team communication

Represent radio reports as timestamped, uncertain observations. Include delay, expiry, confidence, and source unit. Agents may run independently while consuming messages; decentralized execution does not prohibit explicit in-world communication.

### 13.8 Model deployment

Options, in preferred order:

1. **Export actors to ONNX** and use the engine's ONNX runtime.
2. **Reimplement the tiny MLPs** directly in C#/C++/GDScript using exported weights.
3. Use **TorchScript/libtorch** in a native C++ game.
4. Run Python as a separate process only for prototypes; it is unsuitable for most shipped games.

The decentralized actor is very small: two 64-unit hidden layers and five outputs. It is inexpensive enough to run for many cars, though inference should be batched when possible.

### 13.9 Unity integration suggestion

Use `WheelCollider` or your vehicle controller for physics, a road graph/NavMesh for pathing, and Unity Sentis or ONNX Runtime for inference.

Recommended component split:

```text
PoliceAgent.cs
  FSM and update scheduling
PolicePerception.cs
  Physics.Raycast, trigger/hearing events
PoliceNavigation.cs
  NavMesh/road graph queries
PoliceVehicleController.cs
  steering/throttle/brake
PolicePolicy.cs
  local observation and actor inference
PolicePrediction.cs
  trajectory model inference and intercept scoring
PoliceRadio.cs
  shared timestamped alerts
```

Run physics in `FixedUpdate`. Run neural policy decisions at 5–10 Hz, perception at 10–20 Hz, and route planning only when targets change or a replan timer expires. Interpolate visual elements each rendered frame.

### 13.10 Godot integration suggestion

Use `VehicleBody3D`/custom vehicle physics, `NavigationServer3D` or a road graph, and ray queries from `PhysicsDirectSpaceState3D`. Put FSM logic in a police-agent script and expose role/intercept targets as resources. Use a GDExtension or ONNX binding for inference, or reimplement these small dense networks in GDScript/C#.

Use `_physics_process(delta)` for vehicle control. Do not perform a full A* query every frame; keep the 0.35-second replan cadence or respond to navigation invalidation events.

### 13.11 Unreal Engine integration suggestion

Use a custom `AIController`, Behavior Tree or StateTree for high-level states, Mass AI for large fleets, Chaos Vehicles for car physics, EQS for candidate intercept points, and a road graph/NavMesh for routing. Run model inference through ONNX Runtime or NNE.

Suggested mapping:

- FSM states → StateTree states;
- perception → `UAIPerceptionComponent` plus vehicle-noise stimuli;
- radio dispatch → gameplay message subsystem;
- A* utility candidates → EQS generators/tests;
- actor role → blackboard/state parameter;
- steering target → Chaos vehicle controller.

### 13.12 Custom engine integration suggestion

Port `Vec2`, FSM, utility scoring, and observation construction directly. Replace Python heap-based A* with your navigation module. Export model weights and implement dense layers with matrix multiplication and Tanh/ReLU. Add deterministic fixed-step scheduling and seed all stochastic components for replay testing.

### 13.13 Server-authoritative multiplayer

Run police decisions and physics on the authoritative server. Clients should receive replicated transforms, state, siren state, and optional predicted paths for debug only. Do not trust client-reported player positions. For replays and debugging, log sensor observations, policy action, FSM transition, route target, and model version.

### 13.14 Performance scheduling

Not every subsystem needs frame-rate updates:

| System | Suggested frequency |
|---|---:|
| Vehicle physics | 50–120 Hz |
| Local steering | 30–60 Hz |
| Perception | 10–20 Hz |
| MARL actor | 5–10 Hz |
| Trajectory inference | 5–10 Hz |
| A* replan | On change or 2–4 Hz |
| Strategic coordinator | 1–2 Hz |

Use distance-based AI levels of detail. Far-away agents can update less frequently or use simplified simulation.

### 13.15 Training on real game data

Before production use:

1. Instrument the actual game to record player/police transforms, velocities, accelerations, roads, collisions, actions, and outcomes.
2. Split by entire gameplay session, not random rows, to prevent temporal leakage.
3. Include difficult cases: alleys, corners, jumps, reversals, roadblocks, traffic, and network delay.
4. Retrain the trajectory model on the actual vehicle dynamics.
5. Replace fixed expert roles with scenario-dependent expert labels or self-play rewards.
6. Evaluate capture rate, collision rate, time-to-capture, route efficiency, fairness, and player-rated fun.
7. Maintain a deterministic regression scenario suite.

## 14. Production-hardening checklist

- [ ] Move configuration values into versioned data assets.
- [ ] Add structured logging and model/version identifiers.
- [ ] Export models to ONNX or another safe deployment format.
- [ ] Add checkpoint/data hashes and schema versions.
- [ ] Validate every external model before loading.
- [ ] Train on real engine trajectories.
- [ ] Add traffic and dynamic-obstacle handling.
- [ ] Add stuck/reverse/recovery behavior.
- [ ] Add route cancellation and asynchronous path requests.
- [ ] Add inference batching and AI update LOD.
- [ ] Add scenario-level benchmarks and soak tests.
- [ ] Add multiplayer authority and replay diagnostics.
- [ ] Tune pursuit difficulty for fairness and fun, not only capture rate.
- [ ] Profile CPU, allocations, and path-query spikes in the target engine.

## 15. Known limitations

1. The role dataset assigns mostly fixed roles by identity, making 100% classification accuracy easy.
2. The demo city has static obstacles and no traffic.
3. A* is four-directional and does not model lane curvature or turning radius.
4. Cars are kinematic triangles rather than realistic vehicles.
5. The neural trajectory data is synthetic.
6. The radio alert communicates an exact current position during its active period.
7. The Tkinter demo is single-process and intended for visualization, not deployment.
8. Policies have not been evaluated against large numbers of diverse human players.

## 16. Troubleshooting

### Missing model error

Run:

```powershell
..\mini_gpt\.venv\Scripts\python.exe make_datasets.py --rows 50000
..\mini_gpt\.venv\Scripts\python.exe train_models.py
```

### Cars do not respond

Confirm that at least one car has sight/hearing contact. The current distributed spawn points put one unit within initial sight on the clear horizontal road. Press `R` to reset.

### Window appears frozen after dragging

The simulation clamps frame gaps and processes fixed steps. Release the window and it should continue without a physics explosion. If it does not, inspect the terminal for an exception.

### Cars clip an imported game's geometry

The demo collision footprint does not transfer automatically. Use the target engine's vehicle collider and continuous collision detection. Ensure planned waypoints include adequate vehicle clearance.

### Training is slow

Reduce dataset rows, epoch limits, or CTDE iterations for experiments. Restore the documented values for the final run. PyTorch uses CPU in this project.

### Model performs well on CSV but badly in game

This is domain shift. Record real game trajectories and train/validate by full sessions. Confirm normalization and coordinate conventions match runtime exactly.

## 17. Rebuilding this documentation

Run:

```powershell
..\mini_gpt\.venv\Scripts\python.exe build_documentation.py
```

The builder preserves sections 1–17 and replaces the complete source-listing appendix with the current contents of every textual source, configuration, test, report, and README file.

## Complete source listings

The following listings are generated verbatim from the current project files. Binary model checkpoints and CSV datasets are documented by schema instead of embedded.

### `ai.py`

````python
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
            # Capture has one clear rule: maintain contact inside arrest range.
            if distance <= self.arrest_range:
                self.capture_progress += dt
            else:
                self.capture_progress = max(0.0, self.capture_progress - 2.0 * dt)
            if self.capture_progress >= 0.65:
                self._transition(PoliceState.ARREST)
            elif contact:
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
            else:
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
````

### `marl.py`

````python
"""Small CTDE multi-agent actor-critic for cooperative police roles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ai import PoliceCarAI, Vec2


ACTION_NAMES = ("direct", "lead", "left_flank", "right_flank", "rear_block")
NUM_ACTIONS = len(ACTION_NAMES)


class DecentralizedActor(nn.Module):
    """Runtime policy: consumes one car's local observation only."""

    def __init__(self, num_agents: int = 4):
        super().__init__()
        self.num_agents = num_agents
        self.observation_size = 7 + num_agents
        self.network = nn.Sequential(
            nn.Linear(self.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, local_observation: torch.Tensor) -> torch.Tensor:
        return self.network(local_observation)


class CentralizedCritic(nn.Module):
    """Training-only value model: consumes the complete global team state."""

    def __init__(self, num_agents: int = 4):
        super().__init__()
        self.global_state_size = 4 + 4 * num_agents
        self.network = nn.Sequential(
            nn.Linear(self.global_state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.network(global_state).squeeze(-1)


def local_observation(
    car: PoliceCarAI,
    player_position: Vec2,
    player_velocity: Vec2,
    agent_index: int,
    num_agents: int,
) -> torch.Tensor:
    relative = player_position - car.position
    identity = [1.0 if index == agent_index else 0.0 for index in range(num_agents)]
    visible, heard = car.perception.detect(
        car.grid, car.position, player_position, player_velocity
    )
    values = [
        relative.x / car.grid.width,
        relative.y / car.grid.height,
        player_velocity.x / 5.0,
        player_velocity.y / 5.0,
        car.velocity.x / max(car.max_speed, 0.1),
        car.velocity.y / max(car.max_speed, 0.1),
        1.0 if visible or heard else 0.0,
        *identity,
    ]
    return torch.tensor(values, dtype=torch.float32)


def action_offset(action: int, car: PoliceCarAI, player_velocity: Vec2) -> Vec2:
    forward = player_velocity.normalized()
    if forward.length() == 0:
        forward = (car.position - car.last_seen).normalized() if car.last_seen else Vec2(1, 0)
    side = Vec2(-forward.y, forward.x)
    offsets = (
        Vec2(0, 0),
        forward * 3.0,
        side * 3.0,
        side * -3.0,
        forward * -2.5,
    )
    return offsets[action]


@dataclass
class DecentralizedController:
    actor: DecentralizedActor
    num_agents: int = 4

    @classmethod
    def load(cls, path: str | Path, num_agents: int = 4):
        actor = DecentralizedActor(num_agents)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actor.load_state_dict(payload["actor_state"])
        actor.eval()
        return cls(actor, num_agents)

    @torch.no_grad()
    def choose_action(
        self,
        car: PoliceCarAI,
        player_position: Vec2,
        player_velocity: Vec2,
        agent_index: int,
    ) -> int:
        observation = local_observation(
            car, player_position, player_velocity, agent_index, self.num_agents
        )
        return int(self.actor(observation).argmax().item())


def train_ctde(
    output_path: str | Path,
    iterations: int = 1200,
    batch_size: int = 128,
    num_agents: int = 4,
    seed: int = 7,
    initial_actor_state: dict | None = None,
) -> dict[str, float]:
    """Train cooperative roles with a global reward and centralized critic."""
    torch.manual_seed(seed)
    actor = DecentralizedActor(num_agents)
    if initial_actor_state is not None:
        actor.load_state_dict(initial_actor_state)
    critic = CentralizedCritic(num_agents)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-3
    )
    # Direct, lead, left and right roles make a four-car containment team.
    desired_actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)[:num_agents]
    final_accuracy = 0.0

    for _ in range(iterations):
        player = torch.rand(batch_size, 4) * 2 - 1  # position and velocity
        cars = torch.rand(batch_size, num_agents, 4) * 2 - 1
        identities = torch.eye(num_agents).unsqueeze(0).expand(batch_size, -1, -1)
        relative = player[:, None, :2] - cars[:, :, :2]
        detected = (relative.square().sum(dim=-1, keepdim=True) < 1.2).float()
        local = torch.cat(
            (
                relative,
                player[:, None, 2:].expand(-1, num_agents, -1),
                cars[:, :, 2:],
                detected,
                identities,
            ),
            dim=-1,
        )
        logits = actor(local.reshape(batch_size * num_agents, -1)).view(
            batch_size, num_agents, NUM_ACTIONS
        )
        distributions = torch.distributions.Categorical(logits=logits)
        actions = distributions.sample()
        log_probability = distributions.log_prob(actions).sum(dim=1)
        entropy = distributions.entropy().mean()

        correct = (actions == desired_actions).float().mean(dim=1)
        duplicates = torch.stack(
            [
                (actions[:, left] == actions[:, right]).float()
                for left in range(num_agents)
                for right in range(left + 1, num_agents)
            ],
            dim=1,
        ).mean(dim=1)
        global_reward = correct - 0.25 * duplicates
        global_state = torch.cat((player, cars.reshape(batch_size, -1)), dim=1)
        value = critic(global_state)
        advantage = global_reward - value.detach()
        actor_loss = -(log_probability * advantage).mean() - 0.01 * entropy
        critic_loss = (value - global_reward).square().mean()
        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()), 1.0
        )
        optimizer.step()
        final_accuracy = correct.mean().item()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "num_agents": num_agents,
            "training": "centralized actor-critic with global cooperative reward",
            "execution": "decentralized actor with local observation only",
        },
        output_path,
    )
    return {"final_batch_role_accuracy": final_accuracy}
````

### `prediction.py`

````python
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
````

### `simulation.py`

````python
"""Headless fixed-step police pursuit simulation.

This module owns game state and model scheduling.  The Tk UI in main.py is only
an input/render adapter, which makes pursuit behavior deterministic and directly
testable without opening a window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ai import GridMap, PoliceCarAI, PoliceState, Vec2
from marl import DecentralizedController, action_offset
from prediction import PathPredictionEnsemble, NeuralTrajectoryModel


@dataclass
class RadioReport:
    position: Vec2
    velocity: Vec2
    remaining: float = 3.0
    age: float = 0.0

    def advance(self, dt: float) -> None:
        self.remaining -= dt
        self.age += dt

    def extrapolated_position(self) -> Vec2:
        return self.position + self.velocity * min(self.age, 2.0)


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
        self.lives = 3
        self.reset()

    def reset(self) -> None:
        self.lives = 3
        self.reset_round()

    def reset_round(self) -> None:
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
        self.busted = False

    @property
    def game_over(self) -> bool:
        return self.busted and self.lives <= 0

    def retry(self) -> None:
        if self.game_over:
            self.lives = 3
        self.reset_round()

    @property
    def arrested(self) -> bool:
        return self.busted or any(
            car.state == PoliceState.ARREST for car in self.police_cars
        )

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
            self.radio.advance(dt)
            if self.radio.remaining <= 0:
                self.radio = None
        return detections

    def _observed_motion(
        self, detection: tuple[bool, bool]
    ) -> tuple[Vec2, Vec2, bool]:
        if detection[0] or detection[1]:
            return self.player_position, self.player_velocity, True
        if self.radio is not None:
            predicted = self.radio.extrapolated_position()
            cell = self.grid.nearest_passable(self.grid.world_to_cell(predicted))
            return self.grid.cell_center(cell), self.radio.velocity, True
        return self.player_position, self.player_velocity, False

    def _search_target(self, index: int, police: PoliceCarAI) -> Vec2 | None:
        if police.last_seen is None:
            return None
        angle = index * 1.5707963267948966 + police.search_time * 0.35
        radius = min(6.0, 1.5 + police.search_time * 0.8)
        candidate = police.last_seen + Vec2(
            math.cos(angle) * radius, math.sin(angle) * radius
        )
        cell = self.grid.nearest_passable(self.grid.world_to_cell(candidate))
        return self.grid.cell_center(cell)

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
        if self.busted:
            return
        if any(car.state == PoliceState.ARREST for car in self.police_cars):
            self.busted = True
            self.lives = max(0, self.lives - 1)
            self.player_velocity = Vec2(0.0, 0.0)
            for car in self.police_cars:
                car.velocity = Vec2(0.0, 0.0)
            return
        dt = min(max(dt, 1e-4), 0.1)
        self.elapsed += dt
        self._move_player(movement, dt)
        detections = self._update_perception(dt)
        self.decision_remaining -= dt
        self._update_decisions(detections, dt)
        if self.decision_remaining <= 0:
            self.decision_remaining += self.decision_interval

        positions = tuple(car.position for car in self.police_cars)
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
                search_target=self._search_target(index, police),
            )
        if any(car.state == PoliceState.ARREST for car in self.police_cars):
            self.busted = True
            self.lives = max(0, self.lives - 1)
            self.player_velocity = Vec2(0.0, 0.0)
            for car in self.police_cars:
                car.velocity = Vec2(0.0, 0.0)
````

### `main.py`

````python
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
        self.busted_panel = tk.Frame(self.root, bg="#151b23", padx=10, pady=8)
        self.busted_label = tk.Label(
            self.busted_panel,
            text="BUSTED",
            bg="#151b23",
            fg="#ff5268",
            font=("Segoe UI", 12, "bold"),
        )
        self.busted_label.pack(side="left", padx=(0, 14))
        self.retry_button = tk.Button(
            self.busted_panel,
            text="Retry",
            command=self.retry,
            width=10,
            takefocus=True,
        )
        self.retry_button.pack(side="left", padx=4)
        self.quit_button = tk.Button(
            self.busted_panel,
            text="Quit",
            command=self.close,
            width=10,
            takefocus=True,
        )
        self.quit_button.pack(side="left", padx=4)
        self.busted_hint = tk.Label(
            self.busted_panel,
            text="R or Enter retries  |  Esc quits",
            bg="#151b23",
            fg="#94a3b8",
            font=("Segoe UI", 9),
        )
        self.busted_hint.pack(side="left", padx=(14, 0))

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

    def retry(self):
        self.simulation.retry()
        self.busted_panel.pack_forget()
        if self.debug_visible:
            self.status.pack(fill="x", side="bottom")
        self.canvas.focus_set()

    def key_down(self, event):
        key = event.keysym.lower()
        self.keys.add(key)
        if key in {"r", "return"}:
            self.retry()
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
        if self.simulation.busted and not self.busted_panel.winfo_ismapped():
            if self.debug_visible:
                self.status.pack_forget()
            if self.simulation.game_over:
                self.busted_label.configure(text="GAME OVER - NO LIVES LEFT")
                self.retry_button.configure(text="New Game")
            else:
                suffix = "life" if self.simulation.lives == 1 else "lives"
                self.busted_label.configure(
                    text=f"BUSTED - {self.simulation.lives} {suffix} left"
                )
                self.retry_button.configure(text="Retry")
            self.busted_panel.pack(fill="x", side="bottom")
            self.retry_button.focus_set()
        self.draw()
        self.root.after(16, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    PoliceDemo().run()
````

### `evaluate_gameplay.py`

````python
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
````

### `make_datasets.py`

````python
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
````

### `train_models.py`

````python
"""Train supervised sample-data models, then CTDE-fine-tune the MARL actor."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from marl import DecentralizedActor, train_ctde
from prediction import NeuralTrajectoryModel


ROOT = Path(__file__).resolve().parent


def load_csv(path: Path) -> torch.Tensor:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return torch.tensor([[float(value) for value in row] for row in reader])


def batches(features, targets, batch_size=256):
    order = torch.randperm(len(features))
    for start in range(0, len(features), batch_size):
        indices = order[start : start + batch_size]
        yield features[indices], targets[indices]


def split_data(data: torch.Tensor):
    order = torch.randperm(len(data))
    data = data[order]
    train_end = int(len(data) * 0.8)
    validation_end = int(len(data) * 0.9)
    return data[:train_end], data[train_end:validation_end], data[validation_end:]


def train_role_actor(
    data: torch.Tensor, epochs: int
) -> tuple[DecentralizedActor, float, float]:
    train, validation, test = split_data(data)
    actor = DecentralizedActor(4)
    optimizer = torch.optim.Adam(actor.parameters(), lr=2e-3)
    loss_function = nn.CrossEntropyLoss()
    for _ in range(epochs):
        actor.train()
        for x, y in batches(train[:, :-1], train[:, -1].long()):
            loss = loss_function(actor(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    actor.eval()
    with torch.no_grad():
        accuracy = (actor(validation[:, :-1]).argmax(1) == validation[:, -1]).float().mean().item()
        test_accuracy = (actor(test[:, :-1]).argmax(1) == test[:, -1]).float().mean().item()
    return actor, accuracy, test_accuracy


def train_trajectory_model(
    data: torch.Tensor, epochs: int
) -> tuple[NeuralTrajectoryModel, float, float]:
    train, validation, test = split_data(data)
    model = NeuralTrajectoryModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    best_state = None
    best_validation = float("inf")
    stale_epochs = 0
    for _ in range(epochs):
        model.train()
        for x, y in batches(train[:, :6], train[:, 6:]):
            loss = loss_function(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            current_validation = loss_function(
                model(validation[:, :6]), validation[:, 6:]
            ).item()
        if current_validation < best_validation - 1e-8:
            best_validation = current_validation
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_mse = loss_function(model(validation[:, :6]), validation[:, 6:]).item()
        test_mse = loss_function(model(test[:, :6]), test[:, 6:]).item()
    return model, validation_mse, test_mse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train production police AI models")
    parser.add_argument("--role-epochs", type=int, default=60)
    parser.add_argument("--trajectory-epochs", type=int, default=90)
    parser.add_argument("--ctde-iterations", type=int, default=2500)
    args = parser.parse_args(argv)
    torch.manual_seed(2026)
    role_path = ROOT / "data" / "team_roles.csv"
    trajectory_path = ROOT / "data" / "trajectories.csv"
    if not role_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError("Datasets missing. Run make_datasets.py first.")

    role_data = load_csv(role_path)
    trajectory_data = load_csv(trajectory_path)
    actor, role_accuracy, role_test_accuracy = train_role_actor(
        role_data, args.role_epochs
    )
    trajectory_model, trajectory_mse, trajectory_test_mse = train_trajectory_model(
        trajectory_data, args.trajectory_epochs
    )
    trajectory_destination = ROOT / "trajectory_model.pt"
    torch.save(
        {
            "model_state": trajectory_model.state_dict(),
            "validation_mse": trajectory_mse,
            "training_rows": int(len(trajectory_data) * 0.8),
        },
        trajectory_destination,
    )
    ctde_result = train_ctde(
        ROOT / "marl_policy.pt",
        iterations=args.ctde_iterations,
        batch_size=128,
        initial_actor_state=actor.state_dict(),
    )
    print(f"role validation accuracy: {role_accuracy:.4f}")
    print(f"role test accuracy: {role_test_accuracy:.4f}")
    print(f"trajectory validation MSE: {trajectory_mse:.8f}")
    print(f"trajectory test MSE: {trajectory_test_mse:.8f}")
    print(f"CTDE final role accuracy: {ctde_result['final_batch_role_accuracy']:.4f}")
    print(f"saved: {ROOT / 'marl_policy.pt'}")
    print(f"saved: {trajectory_destination}")
    report = {
        "role_rows": len(role_data),
        "trajectory_rows": len(trajectory_data),
        "role_epochs": args.role_epochs,
        "trajectory_epoch_limit": args.trajectory_epochs,
        "ctde_iterations": args.ctde_iterations,
        "role_validation_accuracy": role_accuracy,
        "role_test_accuracy": role_test_accuracy,
        "trajectory_validation_mse": trajectory_mse,
        "trajectory_test_mse": trajectory_test_mse,
        "ctde_final_role_accuracy": ctde_result["final_batch_role_accuracy"],
        "split": "80% train / 10% validation / 10% held-out test",
    }
    (ROOT / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"saved: {ROOT / 'training_report.json'}")


if __name__ == "__main__":
    main()
````

### `train_marl.py`

````python
"""Train and save the centralized-training/decentralized-execution policy."""

from pathlib import Path

from marl import train_ctde


if __name__ == "__main__":
    destination = Path(__file__).with_name("marl_policy.pt")
    result = train_ctde(destination)
    print(f"saved: {destination}")
    print(f"final batch role accuracy: {result['final_batch_role_accuracy']:.3f}")
````

### `export_models_to_go.py`

````python
"""Export locally trained PyTorch police models as dependency-free Go weights.

The generated file is compiled into Port Halden.  Runtime code therefore never
loads pickle/checkpoint data and does not require Python, PyTorch, or a DLL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def go_values(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().reshape(-1).tolist()
    lines = []
    for start in range(0, len(values), 8):
        chunk = values[start : start + 8]
        lines.append("\t" + ", ".join(f"{value:.9g}" for value in chunk) + ",")
    return "\n".join(lines)


def emit_array(name: str, tensor: torch.Tensor) -> str:
    return f"var {name} = [...]float32{{\n{go_values(tensor)}\n}}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actor_path = args.source / "marl_policy.pt"
    trajectory_path = args.source / "trajectory_model.pt"
    report_path = args.source / "training_report.json"
    actor_payload = torch.load(actor_path, map_location="cpu", weights_only=False)
    trajectory_payload = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    actor = actor_payload["actor_state"]
    trajectory = trajectory_payload["model_state"]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    pieces = [
        "// Code generated by police_car_ai/export_models_to_go.py; DO NOT EDIT.\n",
        "package main\n\n",
        f'const policeActorCheckpointSHA256 = "{checksum(actor_path)}"\n',
        f'const policeTrajectoryCheckpointSHA256 = "{checksum(trajectory_path)}"\n',
        f"const policeTrainingRoleRows = {int(report['role_rows'])}\n",
        f"const policeTrainingTrajectoryRows = {int(report['trajectory_rows'])}\n",
        f"const policeTrainingCTDEIterations = {int(report['ctde_iterations'])}\n",
        f"const policeTrainingRoleTestAccuracy = {float(report['role_test_accuracy']):.9g}\n",
        f"const policeTrainingTrajectoryTestMSE = {float(report['trajectory_test_mse']):.9g}\n\n",
        emit_array("policeActorW1", actor["network.0.weight"]),
        emit_array("policeActorB1", actor["network.0.bias"]),
        emit_array("policeActorW2", actor["network.2.weight"]),
        emit_array("policeActorB2", actor["network.2.bias"]),
        emit_array("policeActorW3", actor["network.4.weight"]),
        emit_array("policeActorB3", actor["network.4.bias"]),
        emit_array("policeTrajectoryW1", trajectory["network.0.weight"]),
        emit_array("policeTrajectoryB1", trajectory["network.0.bias"]),
        emit_array("policeTrajectoryW2", trajectory["network.2.weight"]),
        emit_array("policeTrajectoryB2", trajectory["network.2.bias"]),
        emit_array("policeTrajectoryW3", trajectory["network.4.weight"]),
        emit_array("policeTrajectoryB3", trajectory["network.4.bias"]),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(pieces), encoding="utf-8", newline="\n")
    print(f"exported trained actor and trajectory weights to {args.output}")
    print(f"actor sha256: {checksum(actor_path)}")
    print(f"trajectory sha256: {checksum(trajectory_path)}")


if __name__ == "__main__":
    main()
````

### `build_documentation.py`

````python
"""Append current textual project sources to DETAILED_DOCUMENTATION.md."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCUMENT = ROOT / "DETAILED_DOCUMENTATION.md"
MARKER = "## Complete source listings"
SOURCE_FILES = [
    "ai.py",
    "marl.py",
    "prediction.py",
    "simulation.py",
    "main.py",
    "evaluate_gameplay.py",
    "make_datasets.py",
    "train_models.py",
    "train_marl.py",
    "export_models_to_go.py",
    "build_documentation.py",
    "tests/test_ai.py",
    "tests/test_marl.py",
    "tests/test_prediction.py",
    "tests/test_training_data.py",
    "tests/test_simulation.py",
    "pytest.ini",
    "README.md",
    "training_report.json",
    "gameplay_report.json",
]


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ini": "ini",
    ".md": "markdown",
    ".json": "json",
}


def main() -> None:
    current = DOCUMENT.read_text(encoding="utf-8")
    if MARKER not in current:
        raise ValueError(f"Documentation marker missing: {MARKER}")
    introduction = current.split(MARKER, 1)[0].rstrip()
    parts = [
        introduction,
        "",
        MARKER,
        "",
        "The following listings are generated verbatim from the current project files. "
        "Binary model checkpoints and CSV datasets are documented by schema instead of embedded.",
        "",
    ]
    for relative_name in SOURCE_FILES:
        path = ROOT / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"Documented source file is missing: {path}")
        language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")
        source = path.read_text(encoding="utf-8").rstrip()
        parts.extend(
            [
                f"### `{relative_name}`",
                "",
                f"````{language}",
                source,
                "````",
                "",
            ]
        )
    DOCUMENT.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"rebuilt: {DOCUMENT}")
    print(f"documented source files: {len(SOURCE_FILES)}")
    print(f"bytes: {DOCUMENT.stat().st_size}")


if __name__ == "__main__":
    main()
````

### `tests/test_ai.py`

````python
from ai import GridMap, Perception, PoliceCarAI, PoliceState, Vec2, astar


def test_astar_finds_shortest_path_around_obstacles():
    grid = GridMap(6, 5, {(2, 0), (2, 1), (2, 2), (2, 3)})
    path = astar(grid, (0, 0), (5, 0))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (5, 0)
    assert not set(path) & grid.blocked
    assert len(path) == 14


def test_astar_reports_unreachable_goal():
    grid = GridMap(3, 3, {(1, 0), (1, 1), (1, 2)})
    assert astar(grid, (0, 1), (2, 1)) is None


def test_perception_respects_range_obstacles_and_noise():
    clear = GridMap(10, 10)
    perception = Perception(sight_range=5, hearing_range=3, noisy_speed=2)
    assert perception.detect(clear, Vec2(1, 1), Vec2(4, 1), Vec2(0, 0)) == (True, False)
    blocked = GridMap(10, 10, {(2, 1)})
    assert perception.detect(blocked, Vec2(1, 1), Vec2(4, 1), Vec2(0, 0)) == (False, False)
    assert perception.detect(blocked, Vec2(1, 1), Vec2(3, 1), Vec2(3, 0)) == (False, True)


def test_state_machine_chases_loses_searches_then_patrols():
    grid = GridMap(20, 20)
    police = PoliceCarAI(grid, Vec2(2.5, 2.5), [(2, 2), (2, 10)])
    police.update(0.1, Vec2(5.5, 2.5), Vec2(0, 0))
    assert police.state == PoliceState.CHASE
    far = Vec2(19.5, 19.5)
    for _ in range(11):
        police.update(0.1, far, Vec2(0, 0))
    assert police.state == PoliceState.SEARCH
    for _ in range(51):
        police.update(0.1, far, Vec2(0, 0))
    assert police.state == PoliceState.PATROL


def test_police_uses_path_and_closes_distance():
    grid = GridMap(20, 10, {(8, y) for y in range(9)})
    police = PoliceCarAI(grid, Vec2(2.5, 2.5), [(2, 2), (2, 8)])
    player = Vec2(12.5, 2.5)
    police.perception.sight_range = 50
    police.perception.hearing_range = 50
    police.perception.noisy_speed = 0
    initial = police.position.distance_to(player)
    for _ in range(300):
        police.update(1 / 30, player, Vec2(0, 0))
    assert police.position.distance_to(player) < initial
    assert police.state in {PoliceState.CHASE, PoliceState.ARREST}


def test_arrest_and_escape_transitions():
    grid = GridMap(10, 10)
    police = PoliceCarAI(grid, Vec2(2.5, 2.5), [(2, 2), (8, 8)])
    for _ in range(7):
        police.update(0.1, Vec2(2.8, 2.5), Vec2(0, 0))
    assert police.state == PoliceState.ARREST
    police.update(0.1, Vec2(5.0, 2.5), Vec2(0, 0))
    assert police.state == PoliceState.CHASE


def test_distance_alone_can_arrest_moving_player():
    grid = GridMap(10, 10)
    police = PoliceCarAI(grid, Vec2(2.5, 2.5), [(2, 2), (8, 8)])
    for _ in range(7):
        police.update(
            0.1,
            Vec2(2.8, 2.5),
            Vec2(4.0, 0.0),
        )
    assert police.state == PoliceState.ARREST


def test_world_edges_and_building_corners_are_solid_and_slide():
    grid = GridMap(5, 5, {(2, 2)})
    assert not grid.is_drivable_position(Vec2(-0.1, 1.0))
    assert not grid.is_drivable_position(Vec2(2.1, 2.1))
    position, blocked = grid.move_with_slide(Vec2(1.5, 2.5), Vec2(0.7, 0.4))
    assert position.x < 2.0
    assert position.y > 2.5
    assert blocked.x == 1.0 and blocked.y == 0.0
````

### `tests/test_marl.py`

````python
from pathlib import Path

import torch

from ai import GridMap, PoliceCarAI, Vec2
from marl import (
    CentralizedCritic,
    DecentralizedActor,
    DecentralizedController,
    local_observation,
    train_ctde,
)


def test_actor_is_local_and_critic_is_global():
    actor = DecentralizedActor(4)
    critic = CentralizedCritic(4)
    assert actor.observation_size == 11
    assert critic.global_state_size == 20
    assert actor(torch.zeros(3, 11)).shape == (3, 5)
    assert critic(torch.zeros(3, 20)).shape == (3,)


def test_runtime_controller_uses_one_car_observation(tmp_path: Path):
    checkpoint = tmp_path / "policy.pt"
    train_ctde(checkpoint, iterations=120, batch_size=32)
    controller = DecentralizedController.load(checkpoint, 4)
    grid = GridMap(20, 20)
    car = PoliceCarAI(grid, Vec2(2.5, 2.5), [(2, 2), (10, 10)])
    observation = local_observation(car, Vec2(5, 5), Vec2(1, 0), 0, 4)
    assert observation.shape == (11,)
    action = controller.choose_action(car, Vec2(5, 5), Vec2(1, 0), 0)
    assert 0 <= action < 5


def test_ctde_checkpoint_contains_actor_and_training_only_critic(tmp_path: Path):
    checkpoint = tmp_path / "ctde.pt"
    train_ctde(checkpoint, iterations=10, batch_size=8)
    payload = torch.load(checkpoint, weights_only=False)
    assert "actor_state" in payload and "critic_state" in payload
    assert payload["execution"].startswith("decentralized")
````

### `tests/test_prediction.py`

````python
from ai import GridMap, PoliceCarAI, Vec2
from prediction import AlphaBetaGammaPredictor, PathPredictionEnsemble


def test_acceleration_aware_forecast_moves_ahead():
    predictor = AlphaBetaGammaPredictor()
    predictor.observe(Vec2(1, 1), 0.1, Vec2(2, 0))
    predictor.observe(Vec2(1.2, 1), 0.1, Vec2(2, 0))
    future = predictor.forecast(1.0)
    assert future is not None
    assert future.x > 2.5
    assert abs(future.y - 1.0) < 0.1


def test_predictions_are_projected_to_drivable_cells():
    grid = GridMap(10, 10, {(4, 4), (5, 4)})
    ensemble = PathPredictionEnsemble(grid)
    ensemble.observe(Vec2(3.5, 4.5), Vec2(3, 0), 0.1)
    assert ensemble.candidates()
    for _, predicted, _ in ensemble.candidates():
        assert grid.passable(grid.world_to_cell(predicted))


def test_utility_intercept_is_reachable():
    grid = GridMap(15, 10, {(7, y) for y in range(8)})
    car = PoliceCarAI(grid, Vec2(2.5, 2.5), [(2, 2), (2, 8)])
    ensemble = PathPredictionEnsemble(grid)
    ensemble.observe(Vec2(10.5, 2.5), Vec2(0, 2), 0.1)
    target = ensemble.best_intercept(car)
    assert target is not None
    assert grid.passable(grid.world_to_cell(target))


def test_separation_steering_moves_overlapping_routes_apart():
    grid = GridMap(20, 20)
    first = PoliceCarAI(grid, Vec2(2.5, 2.5), [(2, 2), (10, 2)])
    second_position = Vec2(2.8, 2.5)
    first.path = [(3, 2), (4, 2)]
    first._steer(0.1, (second_position,))
    assert first.velocity.x < first.max_speed
````

### `tests/test_training_data.py`

````python
import torch

from make_datasets import make_role_dataset, make_trajectory_dataset
from marl import DecentralizedActor
from prediction import NeuralTrajectoryModel


def test_generated_datasets_have_expected_shapes(tmp_path, monkeypatch):
    import make_datasets

    monkeypatch.setattr(make_datasets, "DATA", tmp_path)
    role = make_role_dataset(40, 1)
    trajectory = make_trajectory_dataset(30, 2)
    assert len(role.read_text(encoding="utf-8").splitlines()) == 41
    assert len(trajectory.read_text(encoding="utf-8").splitlines()) == 31


def test_trainable_models_have_expected_outputs():
    assert DecentralizedActor(4)(torch.zeros(8, 11)).shape == (8, 5)
    assert NeuralTrajectoryModel()(torch.zeros(8, 6)).shape == (8, 10)
````

### `tests/test_simulation.py`

````python
from ai import GridMap, PoliceCarAI, PoliceState, Vec2
from simulation import PoliceSimulation, RadioReport


class CountingController:
    def __init__(self):
        self.calls = 0

    def choose_action(self, car, player_position, player_velocity, agent_index):
        self.calls += 1
        return agent_index


def make_simulation():
    grid = GridMap(20, 20)
    controller = CountingController()
    simulation = PoliceSimulation(
        grid,
        controller,
        None,
        [Vec2(2.5, 2.5), Vec2(17.5, 2.5), Vec2(2.5, 17.5), Vec2(17.5, 17.5)],
        [[(2, 2), (2, 8)], [(17, 2), (17, 8)], [(2, 17), (8, 17)], [(17, 17), (8, 17)]],
    )
    return simulation, controller


def test_neural_decisions_run_at_five_hz_not_physics_rate():
    simulation, controller = make_simulation()
    for car in simulation.police_cars:
        car.perception.sight_range = 100
    for _ in range(60):
        simulation.step(1 / 60)
    assert 16 <= controller.calls <= 24


def test_radio_uses_reported_position_not_hidden_live_position():
    grid = GridMap(20, 20)
    police = PoliceCarAI(grid, Vec2(1.5, 1.5), [(1, 1), (1, 5)])
    police.perception.sight_range = 1
    report = Vec2(5.5, 5.5)
    police.update(
        0.1,
        player_position=Vec2(18.5, 18.5),
        player_velocity=Vec2(0, 0),
        radio_position=report,
        radio_velocity=Vec2(1, 0),
    )
    assert police.last_seen == report


def test_headless_simulation_is_deterministic():
    first, _ = make_simulation()
    second, _ = make_simulation()
    for _ in range(180):
        first.step(1 / 60, Vec2(1, 0))
        second.step(1 / 60, Vec2(1, 0))
    assert first.player_position == second.player_position
    assert [car.position for car in first.police_cars] == [
        car.position for car in second.police_cars
    ]


def test_busted_state_freezes_until_reset():
    simulation, _ = make_simulation()
    simulation.police_cars[0].state = PoliceState.ARREST
    simulation.step(1 / 60)
    assert simulation.busted
    assert simulation.lives == 2
    frozen_position = simulation.player_position
    frozen_time = simulation.elapsed
    simulation.step(1.0, Vec2(1, 0))
    assert simulation.player_position == frozen_position
    assert simulation.elapsed == frozen_time
    simulation.retry()
    assert not simulation.busted
    assert simulation.lives == 2
    assert simulation.player_position == Vec2(5.5, 5.5)


def test_three_lives_then_new_game():
    simulation, _ = make_simulation()
    for expected_lives in (2, 1, 0):
        simulation.police_cars[0].state = PoliceState.ARREST
        simulation.step(1 / 60)
        assert simulation.lives == expected_lives
        assert simulation.busted
        if expected_lives:
            simulation.retry()
    assert simulation.game_over
    simulation.retry()
    assert simulation.lives == 3
    assert not simulation.busted


def test_radio_extrapolates_and_search_units_spread_out():
    simulation, _ = make_simulation()
    simulation.radio = RadioReport(Vec2(5.5, 5.5), Vec2(2, 0))
    simulation.radio.advance(1.0)
    position, velocity, available = simulation._observed_motion((False, False))
    assert available and velocity == Vec2(2, 0)
    assert position.x > 5.5

    for car in simulation.police_cars:
        car.last_seen = Vec2(10.5, 10.5)
        car.search_time = 2.0
    targets = {
        simulation._search_target(index, car)
        for index, car in enumerate(simulation.police_cars)
    }
    assert len(targets) == 4
````

### `pytest.ini`

````ini
[pytest]
testpaths = tests
pythonpath = .
````

### `README.md`

````markdown
# Police Car AI

A CPU-friendly cooperative police-pursuit simulation combining classical game AI, learned trajectory prediction, and multi-agent reinforcement learning with centralized training and decentralized execution (CTDE).

The project includes reproducible datasets, trained PyTorch checkpoints, held-out metrics, automated tests, a smooth Tkinter demonstration, and an exporter for embedding the trained networks in a dependency-free Go game.

## Highlights

- Four independently executing police agents controlled by one shared actor.
- Centralized critic and global cooperative reward used during training only.
- Learned direct pursuit, lead intercept, left flank, and right flank tactics.
- Supervised neural prediction at five future time horizons.
- Alpha-beta-gamma motion filtering and analytical/neural prediction blending.
- A* navigation, road projection, intercept utility, and separation steering.
- Patrol, chase, search, and sustained-proximity arrest states.
- Sight, hearing, obstacle occlusion, last-known-position memory, and radio dispatch.
- Fixed-step simulation, responsive rendering, wall sliding, and corner handling.
- Headless authoritative simulation separated from the Tk rendering adapter.
- Text-free playfield by default; optional diagnostics live outside the game area.
- CPU-only training and inference; no CUDA GPU is required.

## Architecture

```text
                          CENTRALIZED TRAINING
 team state ───────────────────> critic ─────────────> global value
     │                              │
     └─ local observation ─> shared actor ─> cooperative reward

                         DECENTRALIZED EXECUTION
 local sensor/radio state ─> shared actor ─> role ─> tactical offset
 player motion samples ─────> trajectory MLP ───────> future positions
                                                │
 filtered dynamics + A* route utility <─────────┘
                         │
              steering + collision safety
```

### Tactical actor

```text
11 inputs -> Linear(11,64) -> Tanh -> Linear(64,64) -> Tanh
          -> Linear(64,5) -> action logits
```

### Trajectory predictor

```text
6 inputs -> Linear(6,64) -> ReLU -> Linear(64,64) -> ReLU
         -> Linear(64,10) -> five X/Y future deltas
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `ai.py` | Grid, vector math, A*, perception, FSM, steering, and police controller |
| `marl.py` | Decentralized actor, centralized critic, CTDE training, and runtime controller |
| `prediction.py` | Motion filter, neural predictor, road projection, and intercept utility |
| `main.py` | Interactive Tkinter simulation |
| `simulation.py` | Headless authoritative state, honest radio reports, fixed-step physics, and 5 Hz model scheduling |
| `evaluate_gameplay.py` | Deterministic analytical-versus-trained gameplay benchmark |
| `make_datasets.py` | Deterministic role and trajectory dataset generation |
| `train_models.py` | Supervised training, validation selection, held-out evaluation, and CTDE fine-tuning |
| `export_models_to_go.py` | Safe float32 checkpoint export for native Go inference |
| `data/` | Included 50,000-row role and trajectory CSV datasets |
| `tests/` | Unit and integration tests |
| `marl_policy.pt` | Trained actor and training-only critic checkpoint |
| `trajectory_model.pt` | Trained trajectory checkpoint |
| `training_report.json` | Training configuration and held-out metrics |
| `DETAILED_DOCUMENTATION.md` | Full design, algorithms, schemas, integration guidance, and source reference |

## Requirements

- Python 3.10 or newer
- Tkinter, normally included with the Windows Python installer
- Dependencies from `requirements.txt`

Create an environment and install dependencies on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS activation uses `source .venv/bin/activate`. Linux users may need to install their distribution's Tk package separately.

## Run the included trained simulation

The repository already contains trained checkpoints:

```powershell
python main.py
```

Controls:

| Key | Action |
| --- | --- |
| `WASD` or arrows | Drive the blue player car |
| `R` or `Enter` | Retry/reset the simulation and predictors |
| `H` | Toggle routes and the external diagnostics strip |
| `Escape` | Exit |

Police cars patrol independently and broadcast legitimate timestamped detections. The normal playfield contains no instructional or status text. Press `H` only when you want to inspect roles, routes, and radio state; diagnostics appear in a separate strip below the playfield.

The player starts with **three lives**. The only busted rule is one police car remaining within `0.7` game units for `0.65` seconds; player speed and the number of nearby police cars do not affect capture. Each bust consumes one life, freezes physics and AI, and opens **Retry** and **Quit** below the playfield. After the third bust, Retry becomes **New Game** and restores all three lives.

### Police-car responsibilities

| Unit | Learned role | Primary responsibility | Search responsibility |
| --- | --- | --- | --- |
| Police 1 | Direct | Follow the best current/intercept position and maintain close pressure. | Search outward from the last sighting on the first sector. |
| Police 2 | Lead | Move ahead of the player's velocity to block the likely escape route. | Search the second sector, approximately 90 degrees from Police 1. |
| Police 3 | Left flank | Approach the predicted path from its left side and deny left turns. | Search the opposite sector, approximately 180 degrees from Police 1. |
| Police 4 | Right flank | Mirror Police 3 from the right and deny right turns. | Search the fourth sector, approximately 270 degrees from Police 1. |

On detection, police publish a three-second report containing observed position and velocity. If visual contact is lost, the report extrapolates motion for at most two seconds. After it expires, each unit searches a different rotating, expanding sector around the last known position instead of stacking at one point.

## Reproduce the data and training

```powershell
python make_datasets.py --rows 50000 --seed 2026
python train_models.py --role-epochs 60 --trajectory-epochs 90 --ctde-iterations 2500
```

The pipeline:

1. Generates deterministic synthetic role and trajectory samples.
2. Shuffles and splits them into 80% training, 10% validation, and 10% held-out test sets.
3. Supervised-pretrains the shared role actor.
4. Trains the trajectory MLP with AdamW, gradient clipping, early stopping, and best-validation restoration.
5. Fine-tunes the actor and centralized critic using a cooperative global reward.
6. Writes both checkpoints and `training_report.json`.

## Current trained results

| Metric | Result |
| --- | ---: |
| Role rows | 50,000 |
| Trajectory rows | 50,000 |
| Role validation accuracy | 1.0000 |
| Held-out role accuracy | 1.0000 |
| Trajectory validation MSE | 0.0000034386 |
| Held-out trajectory MSE | 0.0000036374 |
| CTDE iterations | 2,500 |
| Final sampled role accuracy | 1.0000 |

The perfect role score must be interpreted in context: the synthetic expert labels intentionally assign stable complementary roles by agent identity. It verifies that the model learned that curriculum; it is not evidence of general driving intelligence.

## Test

```powershell
python -m pytest -q
```

Tests cover pathfinding, perception, state transitions, steering, radio coordination, checkpoint loading, decentralized actions, prediction, road projection, intercept selection, dataset schemas, and model dimensions.

## Gameplay benchmark

Run deterministic pursuit scenarios with and without the trained trajectory model:

```powershell
python evaluate_gameplay.py --duration 45
```

The current report in `gameplay_report.json` uses 60 Hz physics and 5 Hz neural decisions. The trained predictor competes with analytical forecasts through route/timing utility instead of being forcibly averaged into every target.

| Mode | Captures | Mean capture time |
| --- | ---: | ---: |
| Analytical prediction only | 4 / 4 | 18.496 s |
| Trained trajectory candidates | 4 / 4 | 11.950 s |

Both modes now complete all deterministic scenarios under the distance-only busted rule. The trained candidate system cut mean capture time by 6.546 seconds and reduced the square-route scenario from 38.883 seconds to 12.650 seconds. These are deterministic synthetic gameplay scenarios, not a substitute for testing against human players.

## Export to a native Go game

The trained checkpoints can be converted into generated Go float32 arrays:

```powershell
python export_models_to_go.py --output path\to\game\police_models_generated.go
```

The exporter records checkpoint SHA-256 hashes, training metrics, and exact tensors. A native game can compile these arrays and implement the small dense networks directly, avoiding Python, PyTorch, unsafe checkpoint loading, or external model files at runtime. The Port Halden integration uses this deployment method.

## Production integration guidance

The engine-independent pieces in `ai.py`, `marl.py`, and `prediction.py` can be adapted to Unity, Godot, Unreal, or a custom engine. Replace `GridMap` and four-directional A* with the engine's road graph or navigation system, but keep collision and vehicle physics authoritative over model outputs.

For a production game:

1. Record trajectories and pursuit outcomes from the actual vehicle physics.
2. Retrain on those observations instead of relying only on synthetic samples.
3. Evaluate new policies against fixed seeded scenarios and the previous release.
4. Keep perception honest; do not give agents hidden player coordinates.
5. Version model schemas, feature normalization, weights, and checkpoint hashes together.
6. Run decisions at a lower fixed frequency than physics and profile CPU cost.
7. Log observations, chosen actions, routes, and state transitions for debugging.

## Limitations

- Training data is synthetic and does not model a complete commercial driving game.
- The grid planner is four-directional and does not model road curvature or turning radius.
- The current role curriculum makes agent identity highly predictive of the correct action.
- Checkpoints are trusted local PyTorch files; do not load untrusted pickle-based checkpoints.
- Runtime agents do not learn continuously. New weights require offline retraining and validation.

See [DETAILED_DOCUMENTATION.md](DETAILED_DOCUMENTATION.md) for the complete architecture, dataset schemas, algorithm descriptions, file-system guide, production checklist, and engine-specific integration suggestions.
````

### `training_report.json`

````json
{
  "role_rows": 50000,
  "trajectory_rows": 50000,
  "role_epochs": 60,
  "trajectory_epoch_limit": 90,
  "ctde_iterations": 2500,
  "role_validation_accuracy": 1.0,
  "role_test_accuracy": 1.0,
  "trajectory_validation_mse": 3.438608246142394e-06,
  "trajectory_test_mse": 3.637446752691176e-06,
  "ctde_final_role_accuracy": 1.0,
  "split": "80% train / 10% validation / 10% held-out test"
}
````

### `gameplay_report.json`

````json
{
  "physics_hz": 60,
  "decision_hz": 5,
  "duration_limit_seconds": 45.0,
  "analytical": {
    "mode": "analytical_only",
    "captures": 4,
    "scenarios": 4,
    "mean_capture_seconds": 18.496,
    "outcomes": [
      {
        "scenario": "east",
        "arrested": true,
        "seconds": 6.217
      },
      {
        "scenario": "zigzag",
        "arrested": true,
        "seconds": 26.467
      },
      {
        "scenario": "square",
        "arrested": true,
        "seconds": 38.883
      },
      {
        "scenario": "circle",
        "arrested": true,
        "seconds": 2.417
      }
    ]
  },
  "trained": {
    "mode": "trained_trajectory_ensemble",
    "captures": 4,
    "scenarios": 4,
    "mean_capture_seconds": 11.95,
    "outcomes": [
      {
        "scenario": "east",
        "arrested": true,
        "seconds": 6.217
      },
      {
        "scenario": "zigzag",
        "arrested": true,
        "seconds": 26.517
      },
      {
        "scenario": "square",
        "arrested": true,
        "seconds": 12.65
      },
      {
        "scenario": "circle",
        "arrested": true,
        "seconds": 2.417
      }
    ]
  }
}
````
