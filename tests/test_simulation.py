from ai import GridMap, PoliceCarAI, PoliceState, Vec2
from simulation import PoliceSimulation


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
    frozen_position = simulation.player_position
    frozen_time = simulation.elapsed
    simulation.step(1.0, Vec2(1, 0))
    assert simulation.player_position == frozen_position
    assert simulation.elapsed == frozen_time
    simulation.reset()
    assert not simulation.busted
    assert simulation.player_position == Vec2(5.5, 5.5)

