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
    police.update(0.1, Vec2(2.8, 2.5), Vec2(4.0, 0.0))
    assert police.state == PoliceState.ARREST


def test_world_edges_and_building_corners_are_solid_and_slide():
    grid = GridMap(5, 5, {(2, 2)})
    assert not grid.is_drivable_position(Vec2(-0.1, 1.0))
    assert not grid.is_drivable_position(Vec2(2.1, 2.1))
    position, blocked = grid.move_with_slide(Vec2(1.5, 2.5), Vec2(0.7, 0.4))
    assert position.x < 2.0
    assert position.y > 2.5
    assert blocked.x == 1.0 and blocked.y == 0.0
