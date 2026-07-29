import torch
from torch import nn

from experiments.rebuttal.curvature_intervention import rectify_trajectory
from experiments.rebuttal.train_curvature_cd import (
    adjacent_pair_index,
    select_adjacent_training_pair,
    update_ema_target,
)


TIMESTEPS = [1000.0, 937.5, 833.3333129882812, 625.0, 0.0, None]


def make_trajectory() -> torch.Tensor:
    start = torch.tensor([[[[10.0]]], [[[20.0]]]])
    endpoint = torch.tensor([[[[0.0]]], [[[0.0]]]])
    clean = torch.tensor([[[[3.0]]], [[[4.0]]]])
    points = [
        start,
        torch.tensor([[[[8.0]]], [[[17.0]]]]),
        torch.tensor([[[[6.0]]], [[[13.0]]]]),
        torch.tensor([[[[4.0]]], [[[8.0]]]]),
        endpoint,
        clean,
    ]
    return torch.stack(points)


def test_low_to_high_pair_schedule_is_exactly_balanced():
    assert [adjacent_pair_index(step, 4) for step in range(1, 9)] == [
        3,
        2,
        1,
        0,
        3,
        2,
        1,
        0,
    ]


def test_adjacent_target_is_exposed_but_controlled_endpoints_are_fixed():
    curved = make_trajectory()
    rectified = rectify_trajectory(curved, TIMESTEPS)
    assert torch.equal(curved[0], rectified[0])
    assert torch.equal(curved[-2], rectified[-2])
    assert torch.equal(curved[-1], rectified[-1])
    assert not torch.equal(curved[1], rectified[1])

    curved_batch = curved.unsqueeze(0)
    rectified_batch = rectified.unsqueeze(0)
    curved_high, curved_low, curved_clean, high_t, low_t = (
        select_adjacent_training_pair(
            curved_batch, TIMESTEPS, pair_index=0, training_num_frames=1
        )
    )
    rectified_high, rectified_low, rectified_clean, _, _ = (
        select_adjacent_training_pair(
            rectified_batch, TIMESTEPS, pair_index=0, training_num_frames=1
        )
    )
    assert high_t == 1000.0 and low_t == 937.5
    assert torch.equal(curved_high, rectified_high)
    assert not torch.equal(curved_low, rectified_low)
    assert torch.equal(curved_clean, rectified_clean)


def test_boundary_pair_uses_zero_timestep_endpoint():
    trajectory = make_trajectory().unsqueeze(0)
    high, low, clean, high_t, low_t = select_adjacent_training_pair(
        trajectory, TIMESTEPS, pair_index=3, training_num_frames=2
    )
    assert high.shape[1] == low.shape[1] == clean.shape[1] == 2
    assert high_t == 625.0 and low_t == 0.0
    assert torch.equal(low, trajectory[:, 4, :2])


def test_ema_target_update_changes_only_target_toward_online():
    online = nn.Linear(2, 1, bias=True)
    target = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        online.weight.fill_(2.0)
        online.bias.fill_(4.0)
        target.weight.zero_()
        target.bias.zero_()
    update_ema_target(online, target, decay=0.75)
    assert torch.equal(target.weight, torch.full_like(target.weight, 0.5))
    assert torch.equal(target.bias, torch.full_like(target.bias, 1.0))
    assert torch.equal(online.weight, torch.full_like(online.weight, 2.0))
