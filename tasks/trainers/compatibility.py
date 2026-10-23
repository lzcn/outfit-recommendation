"""Training and validation for outfit compatibility."""

import torch

import utils
from tasks.registries import register_trainer
from tasks.trainers.base import BaseTrainer

logger = utils.get_logger(__name__)


@register_trainer("compatibility")
class CompatibilityTrainer(BaseTrainer):
    """Train models implementing train_compatibility and eval_compatibility."""

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
                total_loss, intermediates = self.model.train_compatibility(batch, device)

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

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Run validation and return computed metrics."""
        device = self.cfg.task.device
        self.model.eval()

        self.call_model_hook("setup_for_evaluation", self.cfg.data.valid, device)

        metric = utils.metrics.UserMetricDict(
            {
                "loss": utils.metrics.BPRLoss(),
                "ndcg": utils.metrics.NDCG_Score(),
                "auc": utils.metrics.ROC_AUC_Score(),
            }
        )

        self.valid_loader.build()
        for batch in self.valid_loader:
            scores = self.model.eval_compatibility(batch, device).tolist()
            labels = batch["label"].tolist()
            user_ids = batch["uidx"].tolist()
            metric.update((scores, labels, user_ids))

        return metric.compute()
