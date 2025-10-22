from typing import Any, Mapping, Union

import torch


def to(data: Any, device: Union[str, torch.device] = "cuda") -> Any:
    """Recursively move data to the specified device.

    Args:
        data: Tensor, or nested structure of tensors (dict, list, tuple).
        device: Target device (e.g., "cuda", "cpu").

    Returns:
        Data placed on the specified device.
    """
    if isinstance(data, Mapping):
        return {k: to(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True) if device != "cpu" else data.cpu()
    elif isinstance(data, str):
        return data
    else:
        raise TypeError(f"Unsupported data type for device transfer: {type(data).__name__}")


def select_features(
    select_uidx: int | None = None,
    select_label: int | None = None,
    uidxs: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns a boolean mask for selecting features:
        features[selected] -> (n_selected, dim)

    """

    # Determine device and number of samples
    if uidxs is not None:
        device, n = uidxs.device, uidxs.size(0)
    elif labels is not None:
        device, n = labels.device, labels.size(0)
    else:
        raise ValueError("At least one of `uidxs` or `labels` must be provided.")

    selected = torch.ones(n, dtype=torch.bool, device=device)

    if uidxs is not None and select_uidx is not None:
        selected &= uidxs == select_uidx
    if labels is not None and select_label is not None:
        selected &= labels == select_label

    return selected


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
