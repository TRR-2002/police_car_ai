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
