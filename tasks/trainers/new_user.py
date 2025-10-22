from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import datasets
import utils
from tasks.registries import get_evaluator, register_trainer
from tasks.trainers.base import BaseTrainer

logger = utils.get_logger(__name__)


def config_clone(cfg):
    container = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = OmegaConf.create(container)
    return new_cfg


@register_trainer("new_user")
class NewUserTrainer(BaseTrainer):
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
        logger.info(self.model)

    def _setup_training(self) -> None:
        """Only setup optimizer and scheduler for finetuning."""
        if self.cfg.task.new_user == "finetune":
            self.model.new_user_finetune()
            super()._setup_training()

    def _setup_data_loaders(self) -> None:
        """Setup train, validation and test data loaders."""
        cfg = self.cfg
        self.train_loader = datasets.dataloader.OutfitDataModule(**cfg.data.train)
        self.valid_loader = datasets.dataloader.OutfitDataModule(**cfg.data.valid)
        self.test_loader = datasets.dataloader.OutfitDataModule(**cfg.data.test)

    def _setup_logging(self) -> None:
        """Setup TensorBoard writer and model saver."""
        if self.cfg.task.new_user == "finetune":
            super()._setup_logging()

    def get_best_model(self):
        if self.cfg.task.new_user == "finetune":
            return super().get_best_model()
        return self.model

    def train(self) -> None:
        cfg = self.cfg.task
        device = cfg.device
        num_outfits = cfg.get("n_outfits", 1)

        if cfg.new_user == "closed_form":
            self.model.build_new_user_closed_form(self.train_loader, device, num_outfits)
            return
        elif cfg.new_user == "adhoc":
            self.model.build_new_user_adhoc(self.train_loader, device, num_outfits)
            return
        elif cfg.new_user == "finetune":
            self.model.new_user_finetune()
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

        logger.info("Setting model to best checkpoint.")
        self.model = self.get_best_model()

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Run validation and return computed metrics."""
        task = self.cfg.task
        eval_list = task.get("eval")
        metrics = {}
        for name, value in eval_list.items():
            evaluator = get_evaluator(name, cfg=self.cfg, model=self.model)
            data_cfg = config_clone(value.data)
            data_cfg.phase = "valid"
            eval_metrics = evaluator.evaluate(data_cfg)
            metrics.update({k: np.mean(v) for k, v in eval_metrics.items()})
        return metrics
