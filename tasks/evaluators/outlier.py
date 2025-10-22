"""
Outlier detection evaluator for outfit compatibility.

This module implements the outlier detection task, where the model must
identify which items don't fit well in an outfit.
"""

import numpy as np
from tqdm import tqdm

import utils
from datasets import OutfitDataModule
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import register_evaluator

logger = utils.get_logger(__name__)


@register_evaluator("outlier")
class OutlierEvaluator(BaseEvaluator):
    """Outlier detection evaluator.

    Model must implement: eval_outlier(batch, device) -> Tensor[batch_size, num_items]
    Returns outlier scores where higher = more likely to be outlier.
    """

    @staticmethod
    def compute(model, dataloader: OutfitDataModule, device, run: int = 0) -> dict:
        """Compute outlier detection accuracy."""
        dataloader.build()
        model.eval()
        y_pred, y_true = [], []
        for batch in tqdm(dataloader, desc="Outlier detection", leave=False):
            scores = model.eval_outlier(batch, device)
            y_pred.extend(scores.argmax(dim=-1).tolist())
            y_true.extend(batch["y"].tolist())

        accuracy = float(np.mean(np.array(y_pred) == np.array(y_true)))
        return {"accuracy_d": accuracy}
