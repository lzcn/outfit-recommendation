import torch


def compute_weighted_losses(losses: dict, weights: dict, device=None):
    """
    Compute weighted total loss from selected losses.

    Only losses listed in `weights` (with nonzero, non-None weight)
    are included in the total loss.

    Args:
        losses (dict[str, torch.Tensor]): named scalar loss tensors.
        weights (dict[str, float]): mapping of loss name → weight.
        device (torch.device or None): optional, for initializing total loss tensor.

    Returns:
        total_loss (torch.Tensor): weighted sum (keeps gradients)
        metrics (dict[str, float]): detached individual losses (no overall term)
    """
    if not losses:
        raise ValueError("losses cannot be empty.")

    # pick device from any tensor if not provided
    device = device or next(iter(losses.values())).device
    total_loss = torch.zeros((), device=device)
    metrics = {}

    for name, weight in weights.items():
        if weight is None or weight == 0:
            continue
        term = losses.get(name)
        if term is None:
            continue

        # auto mean if not already scalar
        if term.ndim > 0:
            term = term.mean()

        total_loss = total_loss + weight * term
        metrics[name] = term.detach().item()

    return total_loss, metrics


def update_loss_weights(loss_weight=None, **default_weights):
    """
    Initialize loss weight dictionary.

    Priority: kwargs (defaults) → overridden by `loss_weight` if provided.

    Args:
        loss_weight (dict[str, float] | None): optional user-specified overrides.
        **default_weights: default key=value pairs for each loss.

    Returns:
        dict[str, float]: merged weights (user overrides take precedence)
    """
    weights = dict(default_weights)  # copy defaults first
    if loss_weight:
        weights.update(loss_weight)  # user-provided overrides win
    return weights
