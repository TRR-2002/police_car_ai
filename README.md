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

When the player is captured, physics and AI freeze and a compact panel appears below the playfield with two choices: **Retry** and **Quit**. Retry can also be triggered with `R` or `Enter`; `Escape` quits. Restarting clears the radio report, motion predictors, tactical decisions, arrest state, velocities, and elapsed session time.

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
| Analytical prediction only | 2 / 4 | 19.692 s |
| Trained trajectory candidates | 3 / 4 | 19.494 s |

The trained system additionally captured the zigzag scenario and reduced circular-pursuit capture time from 33.633 seconds to 23.450 seconds. These are deterministic synthetic gameplay scenarios, not a substitute for testing against human players.

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
