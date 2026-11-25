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

