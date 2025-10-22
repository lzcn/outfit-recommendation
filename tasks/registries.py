"""
Task registries for trainers and evaluators.

This module provides a simple registry pattern to register and retrieve
trainer and evaluator classes by name.
"""

from typing import Any, Dict, Type

from .base import BaseTask

TRAINER_REGISTRY: Dict[str, Type[BaseTask]] = {}
EVALUATOR_REGISTRY: Dict[str, Type[BaseTask]] = {}


def register_trainer(name: str):
    """Register a trainer class with the given name.

    Args:
        name: The name to register the trainer under.

    Returns:
        Decorator function that registers the class.

    Example:
        @register_trainer("compatibility")
        class CompatibilityTrainer(BaseTrainer):
            ...
    """

    def decorator(cls: Type[BaseTask]) -> Type[BaseTask]:
        if name in TRAINER_REGISTRY:
            raise ValueError(f"Trainer '{name}' is already registered")
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def register_evaluator(name: str):
    """Register an evaluator class with the given name.

    Args:
        name: The name to register the evaluator under.

    Returns:
        Decorator function that registers the class.

    Example:
        @register_evaluator("compatibility")
        class CompatibilityEvaluator(BaseEvaluator):
            ...
    """

    def decorator(cls: Type[BaseTask]) -> Type[BaseTask]:
        if name in EVALUATOR_REGISTRY:
            raise ValueError(f"Evaluator '{name}' is already registered")
        EVALUATOR_REGISTRY[name] = cls
        cls._name_ = name
        return cls

    return decorator


def get_trainer(name: str, *args: Any, **kwargs: Any) -> BaseTask:
    """Get an initialized trainer instance by name.

    Args:
        name: The name of the trainer to retrieve.
        *args: Positional arguments to pass to the trainer's constructor.
        **kwargs: Keyword arguments to pass to the trainer's constructor.

    Returns:
        An instance of the registered trainer class.

    Raises:
        KeyError: If the trainer name is not registered.
    """
    if name not in TRAINER_REGISTRY:
        available = ", ".join(TRAINER_REGISTRY.keys())
        raise KeyError(f"Trainer '{name}' not found. Available: {available}")
    return TRAINER_REGISTRY[name](*args, **kwargs)


def get_evaluator(name: str, *args: Any, **kwargs: Any) -> BaseTask:
    """Get an initialized evaluator instance by name.

    Args:
        name: The name of the evaluator to retrieve.
        *args: Positional arguments to pass to the evaluator's constructor.
        **kwargs: Keyword arguments to pass to the evaluator's constructor.

    Returns:
        An instance of the registered evaluator class.

    Raises:
        KeyError: If the evaluator name is not registered.
    """
    if name not in EVALUATOR_REGISTRY:
        available = ", ".join(EVALUATOR_REGISTRY.keys())
        raise KeyError(f"Evaluator '{name}' not found. Available: {available}")
    return EVALUATOR_REGISTRY[name](*args, **kwargs)
