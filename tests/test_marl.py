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
