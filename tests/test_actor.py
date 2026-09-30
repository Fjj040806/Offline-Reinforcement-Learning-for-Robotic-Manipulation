import torch
from offline_rl_pusht.models.actor import MLPActor


def test_actor_output_shape():
    model = MLPActor(input_dim=2, hidden_dims=[32, 32], output_dim=2)
    y = model(torch.randn(7, 2))
    assert y.shape == (7, 2)
