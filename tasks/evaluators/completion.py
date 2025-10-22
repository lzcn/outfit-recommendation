import torch
from tqdm import tqdm

import utils
from datasets import OutfitDataModule
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import register_evaluator

logger = utils.get_logger(__name__)


@register_evaluator("completion")
class CompletionEvaluator(BaseEvaluator):

    @staticmethod
    def compute(model, dataloader: OutfitDataModule, device, run=0):
        dataloader.build()
        model.eval()
        total, correct = 0, 0

        for batch in tqdm(dataloader, desc="Evaluating completion task"):
            logits = model.eval_completion(batch, device)  # (B, L)
            preds = torch.argmax(logits, dim=1)  # (B,)
            correct += (preds == 0).sum().item()
            total += logits.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy_c": accuracy}
