"""Loss primitives for the run66 two-stage FNDA training."""

import torch
import torch.nn.functional as F


def profile_huber_loss(pred, target, profile_ids, delta=0.2):
    """Average Huber loss within each profile, then equally across profiles."""
    point_loss = F.huber_loss(pred, target, reduction='none', delta=delta).flatten()
    _, inverse = torch.unique(profile_ids.flatten(), sorted=False, return_inverse=True)
    sums = torch.zeros(inverse.max().item() + 1, device=pred.device, dtype=pred.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, point_loss)
    counts.scatter_add_(0, inverse, torch.ones_like(point_loss))
    return (sums / counts.clamp_min(1.0)).mean()


def second_difference_loss(values, beta=0.05):
    """Robust curvature penalty for [N, 3] samples ordered minus/center/plus."""
    second = values[:, 0] - 2.0 * values[:, 1] + values[:, 2]
    return F.smooth_l1_loss(second, torch.zeros_like(second), beta=beta)


if __name__ == '__main__':
    pred = torch.tensor([[0.0], [1.0], [3.0], [3.0]], requires_grad=True)
    target = torch.zeros_like(pred)
    pids = torch.tensor([1, 1, 2, 2])
    base = profile_huber_loss(pred, target, pids)
    duplicated = profile_huber_loss(
        torch.cat([pred[:2], pred[:2], pred[2:]]),
        torch.zeros(6, 1),
        torch.tensor([1, 1, 1, 1, 2, 2]),
    )
    assert torch.allclose(base, duplicated)
    linear = torch.tensor([[0.0, 1.0, 2.0]])
    spike = torch.tensor([[0.0, 2.0, 0.0]])
    assert second_difference_loss(linear) == 0
    assert second_difference_loss(spike) > 0
