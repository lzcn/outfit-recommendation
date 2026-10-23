"""Per-user AUC, NDCG, and ranking loss for outfit compatibility."""

from tqdm import tqdm

import utils
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import register_evaluator

logger = utils.get_logger(__name__)


@register_evaluator("compatibility")
class CompatibilityEvaluator(BaseEvaluator):
    """Compute AUC, NDCG, and BPR loss separately for each user."""

    @staticmethod
    def compute(model, dataloader, device, run=0) -> dict:
        dataloader.build()
        model.eval()
        metric = utils.metrics.UserMetricDict(
            {
                "loss": utils.metrics.BPRLoss(),
                "ndcg": utils.metrics.NDCG_Score(),
                "auc": utils.metrics.ROC_AUC_Score(),
            }
        )
        for batch in tqdm(dataloader, desc="Evaluating compatibility"):
            scores = model.eval_compatibility(batch, device)
            metric.update((scores.tolist(), batch["label"].tolist(), batch["uidx"].tolist()))
        return metric.compute()
