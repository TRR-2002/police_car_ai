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
