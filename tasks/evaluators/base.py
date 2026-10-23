"""Model loading and repeated evaluation."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import datasets
import utils

logger = utils.get_logger(__name__)


class BaseEvaluator(ABC):
    """Base class for all evaluators.

    Handles common evaluation setup including:
    - Model initialization and loading
    - Test data loader creation
    - Model hook invocation

    Args:
        cfg: The configuration object.
        model: Optional pre-built model. If None, will build from config.
    """

    def __init__(self, cfg: DictConfig, model: Optional[Any] = None):
        self.cfg = cfg
        self.model = model

    def _build_model(self) -> Any:
        """Build and initialize the model from configuration."""
        model = instantiate(self.cfg.model).to(self.cfg.task.device)

        pretrained = self.cfg.task.get("load_pretrained")
        if pretrained:
            utils.load_pretrained(model, pretrained, strict=self.cfg.task.get("load_strict", True))
        elif self.cfg.task.stage == "test":
            raise ValueError("Evaluation requires task.load_pretrained=<checkpoint>; refusing random-weight results.")

        return model

    def call_model_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Call a hook method on the model if it exists."""
        hook = getattr(self.model, hook_name, None)
        if callable(hook):
            return hook(*args, **kwargs)
        return None

    @staticmethod
    @abstractmethod
    def compute(model, dataloader, device, run=0) -> dict:
        """Evaluate the model on the given data loader."""
        pass

    @torch.no_grad()
    def evaluate(self, data_cfg=None, model=None) -> dict:
        """Evaluate the model over the configured trials.

        Args:
            data_cfg: Data configuration; defaults to the test split.
            model: Model to evaluate; otherwise use the configured checkpoint.

        Returns:
            Metric names mapped to lists of per-trial values.
        """
        device = self.cfg.task.device
        num_runs = int(self.cfg.task.get("num_runs", 1))

        # get data loader
        data_cfg = data_cfg or self.cfg.data.test
        data_loader = datasets.dataloader.OutfitDataModule(**data_cfg)

        # get model
        model = model if model is not None else self.model
        model = model if model is not None else self._build_model()
        self.model = model
        model.eval()

        self.call_model_hook("setup_for_evaluation", data_cfg, device)

        results = defaultdict(list)
        for run in range(num_runs):
            utils.seed_everything(int(self.cfg.task.get("seed", 0)) + run)
            run_metrics = self.compute(model, data_loader, device, run)
            logger.info(f"Run[{run+1}] performance:")
            utils.log_metrics_table(f"Evaluation for {self.name()} Task", run_metrics, logger)
            for k, v in run_metrics.items():
                results[k].append(v)
        return results

    def name(self) -> str:
        """Get the name of the evaluator."""
        return getattr(self, "_name_", self.__class__.__name__)
