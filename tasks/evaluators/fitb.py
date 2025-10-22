"""
Fill-In-The-Blank (FITB) evaluator for outfit compatibility.

This module implements the FITB evaluation task, where the model must
select the correct item from multiple choices to complete an outfit.
"""

import numpy as np
from tqdm import tqdm

import utils
from datasets import OutfitDataModule
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import register_evaluator

logger = utils.get_logger(__name__)


@register_evaluator("fitb")
class FITBEvaluator(BaseEvaluator):
    """Fill-In-The-Blank evaluator for outfit compatibility.

    Evaluates the model's ability to select the correct item to complete
    an outfit from multiple choices.
    """

    @staticmethod
    def compute(model, dataloader: OutfitDataModule, device, run=0):
        dataloader.build()
        model.eval()

        scores = []
        num_choices = dataloader.num_answers
        for batch in tqdm(dataloader, desc="Computing FITB scores"):
            batch_scores = model.eval_compatibility(batch, device)
            scores.extend(batch_scores.cpu().numpy())

        # num_outfits x num_choices
        scores = np.array(scores).reshape(-1, num_choices)
        # assume the first choice is the correct one
        accuracy = (scores.argmax(axis=1) == 0).mean()
        return {"fitb": accuracy}
