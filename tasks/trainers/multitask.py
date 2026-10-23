"""Training and validation for the multi-task model."""

import torch
from omegaconf import OmegaConf

import datasets
import utils
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import get_evaluator, register_trainer
from tasks.trainers.base import BaseTrainer

logger = utils.get_logger(__name__)


def config_clone(cfg):
    container = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = OmegaConf.create(container)
    return new_cfg


@register_trainer("multitask")
class MultitaskTrainer(BaseTrainer):
    def train(self) -> None:
        """Run the full training loop."""
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
                total_loss, intermediates = self.model.train_multitask(batch, device)

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
            # score = val_metrics[monitor_metric]
            score = val_metrics["auc"] + val_metrics["accuracy_c"] + val_metrics["accuracy_d"]
            is_best = score > best_metric
            best_metric = max(best_metric, score)

            mark = " (*)" if is_best else ""
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch[{epoch}] Val {monitor_metric}: {score:.4f}{mark} | " f"Best: {best_metric:.4f} | LR: {lr:.6f}"
            )
            utils.log_metrics_table(title="Validation Metrics", metrics=val_metrics, logger=logger)

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

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Run validation and return computed metrics."""
        task = self.cfg.task
        self.model.eval()
        eval_list = task.get("eval", {})
        # evaluate same task using single evaluator
        if len(eval_list) == 0:
            evaluator: BaseEvaluator = get_evaluator(task.name, cfg=self.cfg)
            metrics = evaluator.compute(self.model, self.valid_loader, task.device)
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
