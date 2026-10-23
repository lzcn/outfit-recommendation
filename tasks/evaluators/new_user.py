"""New-user inference from positive support outfits."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import datasets
import utils
from tasks.evaluators.base import BaseEvaluator
from tasks.registries import register_evaluator

logger = utils.get_logger(__name__)


@register_evaluator("new_user")
class NewUserEvaluator(BaseEvaluator):
    @staticmethod
    def compute(model, dataloader, device, run=0):
        raise NotImplementedError("Use evaluate() to load both support and test outfits.")

    @staticmethod
    def _features(model, loader, device):
        loader.next()
        users, features, labels = [], [], []
        for batch in tqdm(loader, desc="Encoding outfits"):
            data = batch["data"].to(device)
            size, num_items = data.shape[:2]
            visual = model.visual_feature(data.flatten(0, 1)).reshape(size, num_items, -1)
            cate = batch["cate"].to(device).reshape(size, num_items, -1)
            features.append(model.visual_encoder.latent_z(visual, cate).flatten(1))
            users.append(batch["uidx"].to(device))
            labels.append(batch["label"].to(device))
        return torch.cat(users), F.normalize(torch.cat(features), dim=-1), torch.cat(labels)

    @torch.no_grad()
    def evaluate(self, data_cfg=None, model=None):
        cfg = self.cfg
        if cfg.task.new_user != "closed_form":
            raise ValueError("Use task.new_user=closed_form for support-based inference.")
        utils.seed_everything(int(cfg.task.seed))
        model = model if model is not None else self.model
        model = model if model is not None else self._build_model()
        model.eval()
        if model.num_seeds != 1 or not model.anchors.use_u_anchors:
            raise ValueError("New-user inference requires one outfit embedding and user anchors.")
        support_loader = datasets.OutfitDataModule(**cfg.data.train)
        test_loader = datasets.OutfitDataModule(**cfg.data.test)
        support_users, support_features, support_labels = self._features(model, support_loader, cfg.task.device)
        test_users, test_features, test_labels = self._features(model, test_loader, cfg.task.device)
        user_ids = sorted(test_users.unique().tolist())
        if user_ids != list(range(cfg.data.num_users)):
            raise ValueError("Test user IDs must cover 0..num_users-1.")

        source_anchors = F.normalize(model.anchors.u_anchors, dim=-1)
        anchor_pool = source_anchors.flatten(0, 1)
        source_scale = source_anchors.mean(1).norm(dim=-1).mean()
        user_weight = model.anchors._nonneg(model.anchors.lambda_u)
        global_score = torch.zeros(len(test_features), device=test_features.device)
        if model.anchors.use_g_anchors:
            global_anchors = F.normalize(model.anchors.g_anchors.flatten(0, 1), dim=-1)
            global_score = (test_features @ global_anchors.T).mean(-1)
            global_score *= model.anchors._nonneg(model.anchors.lambda_g)
        pools = {u: torch.where((support_users == u) & (support_labels == 1))[0].cpu().numpy()
                 for u in user_ids}
        test_indices = {u: torch.where(test_users == u)[0] for u in user_ids}
        labels = test_labels.cpu().numpy()
        results = {}
        for size in cfg.task.support_sizes:
            size = int(size)
            if size < 1 or any(len(pool) < size for pool in pools.values()):
                raise ValueError(f"Each user needs at least {size} positive support outfits.")
            values = []
            for run in range(int(cfg.task.num_runs)):
                rng = np.random.RandomState(int(cfg.task.seed) + run)
                user_scores = []
                for user in user_ids:
                    rows = rng.permutation(pools[user])[:size]
                    support = support_features[rows]
                    idx = test_indices[user]
                    query = test_features[idx]
                    if size == 1:
                        response = (anchor_pool @ support.T).mean(-1)
                        selected = anchor_pool[response > 0]
                        if len(selected) == 0:
                            selected = anchor_pool[response.argmax().reshape(1)]
                        user_score = (query @ F.normalize(selected, dim=-1).T).mean(-1)
                    else:
                        prototype = F.normalize(support.mean(0), dim=-1) * source_scale
                        user_score = query @ prototype
                    scores = user_weight * user_score + global_score[idx]
                    user_scores.append(roc_auc_score(labels[idx.cpu().numpy()], scores.cpu().numpy()))
                values.append(float(np.mean(user_scores)))
            results[f"auc@{size}"] = values
            logger.info("Support=%d: macro-user AUC %.6f +/- %.6f (%d trials, %d users)",
                        size, np.mean(values), np.std(values), len(values), len(user_ids))
        return results
