"""
Tasks module for outfit recommendation.

This module provides a registry-based system for trainers and evaluators.
All trainer and evaluator modules are automatically discovered and registered.
"""

import importlib
import pkgutil
from pathlib import Path


def _auto_register_modules() -> None:
    """Automatically import all modules in trainers and evaluators.

    This triggers the @register_trainer and @register_evaluator decorators,
    making all tasks available in the registries.
    """
    current_dir = Path(__file__).parent

    for subdir in ["trainers", "evaluators"]:
        module_dir = current_dir / subdir
        if not module_dir.exists():
            continue

        for module_info in pkgutil.iter_modules([str(module_dir)]):
            if module_info.name.startswith("_"):
                continue

            module_path = f"tasks.{subdir}.{module_info.name}"
            try:
                importlib.import_module(module_path)
            except Exception as e:
                print(f"Warning: Failed to import {module_path}: {e}")


# Auto-register all trainers and evaluators
_auto_register_modules()

# Export getter functions for external use
from tasks.registries import get_evaluator, get_trainer

__all__ = [
    "get_trainer",
    "get_evaluator",
]
