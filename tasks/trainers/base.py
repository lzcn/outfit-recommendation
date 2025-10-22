"""
Base trainer class for all training tasks.

This module provides the abstract base class for all trainers, handling
common setup like optimizer, scheduler, data loaders, and model checkpointing.
"""

from abc import ABC
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import get_evaluator

logger = utils.get_logger(__name__)


def config_clone(cfg):
    container = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = OmegaConf.create(container)
    return new_cfg


class BaseTrainer(ABC):
    """Base class for all trainers.

    Handles common training setup including:
    - Model initialization
    - Optimizer and scheduler setup
    - Data loader creation
    - TensorBoard logging
    - Model checkpointing

    Args:
        cfg: The configuration object.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._build_model()
        self._setup_training()
        self._setup_data_loaders()
        self._setup_logging()
        self.iterations = 0

    def _build_model(self) -> Any:
        """Build and initialize the model."""
        self.model = instantiate(self.cfg.model).to(self.cfg.task.device)

        pretrained = self.cfg.task.get("load_pretrained")
        if pretrained:
            utils.load_pretrained(self.model, pretrained)

    def _setup_training(self) -> None:
        """Setup optimizer and scheduler."""
        cfg = self.cfg
        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        self.scheduler = instantiate(cfg.scheduler, optimizer=self.optimizer) if cfg.get("scheduler") else None

    def _setup_data_loaders(self) -> None:
        """Setup train, validation and test data loaders."""
        cfg = self.cfg
        self.train_loader = datasets.dataloader.OutfitDataModule(**cfg.data.train)
        self.valid_loader = datasets.dataloader.OutfitDataModule(**cfg.data.valid)
        self.test_loader = datasets.dataloader.OutfitDataModule(**cfg.data.test)

    def _setup_logging(self) -> None:
        """Setup TensorBoard writer and model saver."""
        cfg = self.cfg
        output_dir = utils.get_output_dir()

        self.writer = SummaryWriter(log_dir=output_dir / "summary")

        score_name = cfg.task.get("monitor_metric", "auc")
        filename_prefix = cfg.task.get("name", "model")

        self.saver = utils.ModelSaver(
            dirname=output_dir / "checkpoints",
            filename_prefix=filename_prefix,
            score_name=score_name,
            save_best=True,
            save_latest=True,
            mode="max",
        )

    def call_model_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Call a hook method on the model if it exists."""
        hook = getattr(self.model, hook_name, None)
        if callable(hook):
            return hook(*args, **kwargs)
        return None

    def get_best_model(self) -> Any:
        """Load and return the best saved model.

        Returns:
            The model loaded from the best checkpoint.

        Raises:
            AssertionError: If no best checkpoint is found.
        """
        best_checkpoint = self.saver.best_checkpoint
        assert best_checkpoint is not None, "No best model checkpoint found!"

        model = instantiate(self.cfg.model).to(self.cfg.task.device)
        utils.load_pretrained(model, best_checkpoint)

        return model

    def train(self) -> None:
        cfg = self.cfg.task
        device = cfg.device

        num_epochs = int(cfg.num_epochs)
        grad_clip = float(cfg.get("grad_clip", 5.0))
        log_interval = cfg.get("display_interval", 100)
        tb_interval = cfg.get("summary_interval", 100)
        monitor_metric = cfg.get("monitor_metric", "auc")

        best_metric = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            self.train_loader.next()

            meter = utils.meter.BundleMeter(win_size=50)
            num_iterations = len(self.train_loader)

            for batch in self.train_loader:
                total_loss, intermediates = self.model.train_step(batch, device)

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                self.optimizer.step()

                meter.update(intermediates)

                if self.iterations % log_interval == 0:
                    current_iter = self.iterations % num_iterations
                    logger.info(
                        f"Epoch[{epoch}/{num_epochs}] "
                        f"Iter[{current_iter}/{num_iterations}] "
                        f"Total[{self.iterations}]"
                    )
                    meter.logging()

                if self.iterations % tb_interval == 0:
                    metrics_tb = dict(intermediates)
                    for key, value in metrics_tb.items():
                        self.writer.add_scalar(f"train/{key}", value, self.iterations)
                    self.writer.add_scalar("train/total_loss", total_loss.item(), self.iterations)

                self.iterations += 1

            val_metrics = self.validate_epoch()
            score = val_metrics[monitor_metric]
            is_best = score > best_metric
            best_metric = max(best_metric, score)

            mark = " (*)" if is_best else ""
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch[{epoch}] Val {monitor_metric}: {score:.4f}{mark} | " f"Best: {best_metric:.4f} | LR: {lr:.6f}"
            )

            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("train/lr", lr, epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            self.saver.save(self.model, score=score, epoch=epoch)

        logger.info("Training completed successfully.")
        # set the model to the best model after training
        self.model = self.get_best_model()

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Run validation and return computed metrics."""
        task = self.cfg.task
        eval_list = task.get("eval", {})
        # evaluate same task using single evaluator
        if len(eval_list) == 0:
            evaluator: BaseEvaluator = get_evaluator(task.name, cfg=self.cfg)
            metrics = evaluator.compute(self.model, self.data.valid, task.device)
            return metrics
        # evaluate multiple evaluators
        metrics = {}
        for name, value in eval_list.items():
            evaluator: BaseEvaluator = get_evaluator(name, cfg=self.cfg)
            data_cfg = config_clone(value.data)
            data_cfg.phase = "valid"
            data_loader = datasets.dataloader.OutfitDataModule(**data_cfg)
            eval_metrics = evaluator.compute(self.model, data_loader, task.device)
            metrics.update({k: v for k, v in eval_metrics.items()})
        return metrics
