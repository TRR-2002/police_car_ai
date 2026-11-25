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
