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
