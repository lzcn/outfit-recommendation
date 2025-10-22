"""
Base classes for tasks (trainers and evaluators).

This module provides abstract base classes that define the common interface
and functionality for all trainers and evaluators.
"""

from abc import ABC
from typing import Any, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

import utils


class BaseTask(ABC):
    """Base class for all tasks (trainers and evaluators).

    This class provides common functionality for model building and
    model hook invocation.

    Args:
        cfg: The configuration object.
        model: Optional pre-built model. If None, will build from config.
    """

    def __init__(self, cfg: DictConfig, model: Optional[Any] = None):
        self.cfg = cfg
        self.model = model if model is not None else self.build_model()

    def build_model(self) -> Any:
        """Build and initialize the model from configuration.

        Returns:
            The initialized model on the configured device.
        """
        cfg = self.cfg
        model = instantiate(cfg.model).to(cfg.task.device)

        # Load pretrained weights if specified
        pretrained = cfg.task.get("load_pretrained") or cfg.get("load_pretrained")
        if pretrained:
            utils.load_pretrained(model, pretrained)

        return model

    def call_model_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Call a hook method on the model if it exists.

        This allows models to define optional setup/teardown methods
        that will be called automatically if present.

        Args:
            hook_name: The name of the hook method to call.
            *args: Positional arguments to pass to the hook.
            **kwargs: Keyword arguments to pass to the hook.

        Returns:
            The return value of the hook method, or None if not found.
        """
        hook = getattr(self.model, hook_name, None)
        if callable(hook):
            return hook(*args, **kwargs)
        return None
