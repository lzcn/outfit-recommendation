"""Plots of item-level outlier scores."""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

import datasets
import utils
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import register_evaluator

logger = utils.get_logger(__name__)


@register_evaluator("outlier_plotter")
class OutlierPlotter(BaseEvaluator):
    """Export outfit images and item-level outlier scores as PDF figures."""

    def __init__(self, cfg: DictConfig, model: Optional[Any] = None):
        self.cfg = cfg
        self.model = model
        reader = cfg.task.get("reader")
        self.image_reader = datasets.reader.getReader(**reader)

    def _build_model(self) -> Any:
        """Build and initialize the model from configuration."""
        model = instantiate(self.cfg.model).to(self.cfg.task.device)

        pretrained = self.cfg.task.get("load_pretrained")
        if pretrained:
            utils.load_pretrained(model, pretrained)

        return model

    def call_model_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Call a hook method on the model if it exists."""
        hook = getattr(self.model, hook_name, None)
        if callable(hook):
            return hook(*args, **kwargs)
        return None

    def compute():
        pass

    @torch.no_grad()
    def evaluate(self, data_cfg=None, model=None) -> dict:
        """Evaluate outlier detection and export per-outfit visualizations as PDFs."""

        device = self.cfg.task.device

        # 1. Build dataloader
        data_cfg = data_cfg or self.cfg.data.test
        data_loader = datasets.dataloader.OutfitDataModule(**data_cfg)

        # 2. Build or load model
        model = model or self._build_model()
        model.eval()

        # 3. Run inference
        outfits, lengths, scores, labels = [], [], [], []
        for batch in data_loader:
            outlier_scores = model.eval_outlier(batch, device)
            outfits.extend(batch["name"])
            lengths.extend(batch["size"].tolist())
            labels.extend(batch["y"].cpu().numpy().tolist())
            scores.append(outlier_scores.cpu().numpy())
        scores = -np.vstack(scores)

        out_dir = utils.get_output_dir() / "outlier_vis_pdf"
        out_dir.mkdir(parents=True, exist_ok=True)

        num_outfits = len(outfits)
        for i in tqdm(range(num_outfits), desc="Generating outlier plots"):
            # original outfit
            ori_outfit = data_loader.dataset.pos_data[i].tolist()
            _, _, *items = ori_outfit
            n = len(items) // 2
            pos_keys = data_loader.dataset.datum[0].get_key(items[:n], items[n:])

            # evaluated outfit
            outfit_keys = outfits[i].split(",")[: lengths[i]]
            outfit_scores = scores[i][: lengths[i]]
            gt_idx = int(labels[i])
            pred_idx = int(np.argmin(outfit_scores))

            pos_images = [np.array(self.image_reader(k)) for k in pos_keys]
            eval_images = [np.array(self.image_reader(k)) for k in outfit_keys]

            s = outfit_scores.astype(np.float32)
            s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
            num_items = len(eval_images)

            fig, axes = plt.subplots(2, num_items, figsize=(2.2 * num_items, 5.5), squeeze=False)

            # Row 0: original
            for idx, ax in enumerate(axes[0]):
                ax.imshow(pos_images[idx])
                ax.axis("off")
                # ax.set_title("orig", color="black", fontsize=8, pad=4)
                # ax.add_patch(
                #     plt.Rectangle(
                #         (0, 0),
                #         ax.get_xlim()[1],
                #         ax.get_ylim()[0],
                #         fill=False,
                #         linewidth=1.2,
                #         edgecolor=(0.3, 0.3, 0.3, 0.6),
                #     )
                # )

            # Row 1: evaluated outfit (only GT frame)
            for idx, ax in enumerate(axes[1]):
                ax.imshow(eval_images[idx])
                ax.axis("off")
                ax.set_title(f"{outfit_scores[idx]:.2f}", color="red", fontsize=8, pad=4)

                # Green dashed border = ground-truth outlier
                if idx == gt_idx:
                    ax.set_title(f"{outfit_scores[idx]:.2f} (GT)", color="green", fontsize=8, pad=4)
                    # ax.add_patch(
                    #     plt.Rectangle(
                    #         (0, 0),
                    #         ax.get_xlim()[1],
                    #         ax.get_ylim()[0],
                    #         fill=False,
                    #         linewidth=3,
                    #         edgecolor=(0, 1, 0, 0.9),
                    #         linestyle="--",
                    #     )
                    # )
                if idx == pred_idx:
                    ax.set_title(f"{outfit_scores[idx]:.2f} (PRED)", color="blue", fontsize=8, pad=4)
                    # ax.add_patch(
                    #     plt.Rectangle(
                    #         (0, 0),
                    #         ax.get_xlim()[1],
                    #         ax.get_ylim()[0],
                    #         fill=False,
                    #         linewidth=3,
                    #         edgecolor=(0, 0, 1, 0.9),
                    #         linestyle="--",
                    #     )
                    # )

            correct = pred_idx == gt_idx
            fig.suptitle(
                f"Outfit {i} | gt={gt_idx}, pred={pred_idx}, correct={correct}, "
                f"pred_score={outfit_scores[pred_idx]:.2f}",
                fontsize=10,
            )

            save_path = out_dir / f"outfit_{i:05d}.pdf"
            fig.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight", format="pdf", transparent=True)
            plt.close(fig)

        logger.info(f"Saved {num_outfits} outlier visualizations as PDFs to {out_dir}")
        return {"num_outfits": num_outfits}

    def name(self) -> str:
        """Get the name of the evaluator."""
        return getattr(self, "_name_", self.__class__.__name__)
