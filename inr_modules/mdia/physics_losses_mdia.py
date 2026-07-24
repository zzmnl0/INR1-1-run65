"""Active physics losses for FSIA-INR training."""

import torch


_ALT_RANGE_HALF = 190.0


def profile_peak_alignment_loss(ne_at_peak, coords_peak):
    """Penalize a non-zero vertical derivative at the IRI F2 peak."""
    if not coords_peak.requires_grad:
        return torch.zeros((), device=ne_at_peak.device)

    with torch.amp.autocast('cuda', enabled=False):
        try:
            grad = torch.autograd.grad(
                ne_at_peak.float().sum(),
                coords_peak.float(),
                create_graph=True,
                retain_graph=True,
            )[0]
        except RuntimeError:
            return torch.zeros((), device=ne_at_peak.device)

    return (grad[:, 2] * _ALT_RANGE_HALF).square().mean()


def combined_mdia_physics_loss(
    pred_ne,
    ne_bkg,
    coords,
    w_bkg_low=0.25,
    w_bkg_high=0.02,
    w_bkg_transition=250.0,
    w_bkg_sharpness=25.0,
    trust_iri=None,
):
    """Height-adaptive IRI background anchoring used by the current model."""
    alt_km = coords[:, 2:3]
    high_alt = torch.sigmoid(
        (alt_km - w_bkg_transition) / w_bkg_sharpness)
    weight = w_bkg_high * high_alt + w_bkg_low * (1.0 - high_alt)
    if trust_iri is not None:
        weight = weight * trust_iri.detach().view(-1, 1)

    loss = (weight * (pred_ne - ne_bkg.detach()).square()).mean()
    return loss, {'bkg': loss.item(), 'physics_total': loss.item()}


if __name__ == '__main__':
    coords = torch.tensor(
        [[0.0, 0.0, 150.0, 0.0], [0.0, 0.0, 400.0, 0.0]])
    pred = torch.tensor([[11.0], [11.0]], requires_grad=True)
    bkg = torch.zeros_like(pred)
    loss, values = combined_mdia_physics_loss(pred, bkg, coords)
    assert loss.requires_grad
    assert values['bkg'] == values['physics_total']
