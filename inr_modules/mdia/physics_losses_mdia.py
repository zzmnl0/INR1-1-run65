"""Loss primitives for the run66 two-stage FNDA training."""

import torch
import torch.nn.functional as F


def profile_huber_loss(pred, target, profile_ids, delta=0.2, valid_mask=None):
    """Average Huber loss within each profile, then equally across profiles."""
    point_loss = F.huber_loss(pred, target, reduction='none', delta=delta).flatten()
    flat_ids = profile_ids.flatten()
    if valid_mask is not None:
        valid = valid_mask.flatten().to(device=pred.device, dtype=torch.bool)
        if valid.shape != point_loss.shape:
            raise ValueError('valid_mask must have one value per prediction')
        if not valid.any():
            return pred.sum() * 0.0
        point_loss = point_loss[valid]
        flat_ids = flat_ids[valid]
    _, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
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
