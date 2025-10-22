"""
Compatibility evaluator for outfit compatibility prediction.

This module implements the evaluator for the compatibility task,
which measures how well the model predicts outfit compatibility.
"""

from tqdm import tqdm

import utils
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import register_evaluator

logger = utils.get_logger(__name__)


@register_evaluator("compatibility")
class CompatibilityEvaluator(BaseEvaluator):
    """Evaluator for outfit compatibility prediction task.

    Computes metrics like AUC, NDCG, and loss for compatibility prediction.
    """

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
        result = metric.compute()
        return result
