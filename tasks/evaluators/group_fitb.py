"""
Fill-In-The-Blank (FITB) evaluator for outfit compatibility.

This module implements the FITB evaluation task, where the model must
select the correct item from multiple choices to complete an outfit.
"""

import torch
from tqdm import tqdm

import utils
from datasets import OutfitDataModule
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import register_evaluator

logger = utils.get_logger(__name__)


@register_evaluator("group_fitb")
class GroupFITBEvaluator(BaseEvaluator):
    """Fill-In-The-Blank evaluator for outfit compatibility.

    Evaluates the model's ability to select the correct item to complete
    an outfit from multiple choices.
    """

    @staticmethod
    def compute(model, dataloader: OutfitDataModule, device, run=0):
        dataloader.build()
        model.eval()
        total, correct = 0, 0
        for batch in tqdm(dataloader, desc="Computing Group FITB scores"):
            logits = model.eval_group_fitb(batch, device)  # (B, N)
            preds = torch.argmax(logits, dim=1)  # (B,)
            correct += (preds == 0).sum().item()
            total += logits.size(0)
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy_c": accuracy}
