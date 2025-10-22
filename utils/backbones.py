from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Callable, Dict, Iterable, Union

import torch.nn as nn
from torchvision import models

# Global registry for all backbone models
_BACKBONES: Dict[str, Callable[..., nn.Module]] = {}


def backbone(name: str, *, weights: Union[str, None] = "DEFAULT", **kwargs) -> nn.Module:
    """Retrieve a backbone model by name with classification head removed."""
    try:
        fn = _BACKBONES[name]
    except KeyError:
        raise ValueError(f"Unknown backbone '{name}'. Registered: {sorted(_BACKBONES.keys())}")
    return fn(weights=weights, **kwargs)


def _register_backbone(func: Callable = None, *, name: str = None) -> Callable:
    """Register a backbone constructor function."""
    if func is None:
        return partial(_register_backbone, name=name)

    model_name = name or func.__name__
    assert model_name not in _BACKBONES, f"{model_name} is already registered."
    _BACKBONES[model_name] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _strip_classification_head(model: nn.Module) -> nn.Module:
    """Replace the classification head with Identity layers."""
    # ResNet / RegNet style
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    # MobileNet / EfficientNet / VGG / ConvNeXt / etc.
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential) and len(clf) > 0:
            clf[-1] = nn.Identity()
        elif isinstance(clf, (nn.Linear, nn.Conv2d)):
            model.classifier = nn.Identity()
    # Vision Transformers
    if hasattr(model, "heads"):
        model.heads = nn.Identity()
    if hasattr(model, "head"):
        model.head = nn.Identity()
    return model


def _create_backbone(
    ctor: Callable,
    *,
    weights: Union[str, None] = "DEFAULT",
    force_kwargs: dict | None = None,
    **kwargs,
) -> nn.Module:
    """Instantiate a torchvision model and strip its head."""
    force_kwargs = force_kwargs or {}
    sig = inspect.signature(ctor)
    valid_kwargs = {k: v for k, v in {**kwargs, **force_kwargs}.items() if k in sig.parameters}
    if "weights" in sig.parameters:
        model = ctor(weights=weights, **valid_kwargs)
    else:
        model = ctor(**valid_kwargs)
    return _strip_classification_head(model)


# fmt: off
# Registration
_DEFAULT_ALLOW = [
    # ResNet family
    "resnet18","resnet34","resnet50","resnet101","resnet152",
    "wide_resnet50_2","wide_resnet101_2",
    "resnext50_32x4d","resnext101_32x8d",
    # DenseNet
    "densenet121","densenet169","densenet201","densenet161",
    # MobileNet / MNAS / ShuffleNet / SqueezeNet
    "mobilenet_v2","mobilenet_v3_small","mobilenet_v3_large",
    "mnasnet0_5","mnasnet0_75","mnasnet1_0","mnasnet1_3",
    "shufflenet_v2_x0_5","shufflenet_v2_x1_0","shufflenet_v2_x1_5","shufflenet_v2_x2_0",
    "squeezenet1_0","squeezenet1_1",
    # EfficientNet
    "efficientnet_b0","efficientnet_b1","efficientnet_b2","efficientnet_b3",
    "efficientnet_b4","efficientnet_b5","efficientnet_b6","efficientnet_b7",
    # VGG
    "vgg11","vgg11_bn","vgg13","vgg13_bn","vgg16","vgg16_bn","vgg19","vgg19_bn",
    # Inception / GoogLeNet
    "inception_v3","googlenet",
    # RegNet
    "regnet_y_400mf","regnet_y_800mf","regnet_y_1_6gf","regnet_y_3_2gf","regnet_y_8gf",
    "regnet_x_400mf","regnet_x_800mf","regnet_x_1_6gf","regnet_x_3_2gf","regnet_x_8gf",
    # ConvNeXt
    "convnext_tiny","convnext_small","convnext_base","convnext_large",
    # Vision Transformers / Swin / MaxViT
    "vit_b_16","vit_b_32","vit_l_16","vit_l_32",
    "swin_t","swin_s","swin_b",
    "maxvit_t","maxvit_s","maxvit_b",
]
# fmt: on
_FORCE_KW = {
    "inception_v3": {"aux_logits": False},
    "googlenet": {"aux_logits": False},
}


def _register_all_torchvision_backbones(
    *,
    allowlist: Iterable[str] | None = None,
    denylist: Iterable[str] | None = None,
) -> None:
    allow = set(allowlist) if allowlist is not None else set(_DEFAULT_ALLOW)
    deny = set(denylist or [])
    for name in sorted(allow - deny):
        if not hasattr(models, name):
            continue
        ctor = getattr(models, name)
        if not callable(ctor):
            continue
        force_kwargs = _FORCE_KW.get(name, {})
        _register_backbone(
            partial(_create_backbone, ctor, force_kwargs=force_kwargs),
            name=name,
        )


# Register on import
_register_all_torchvision_backbones()
