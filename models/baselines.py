import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import ops

logger = utils.get_logger(__name__)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def embed_layer(in_features, out_features, dropout=0.2):
    x = nn.Sequential(
        nn.Linear(in_features, out_features), nn.LayerNorm(out_features), nn.ReLU(inplace=True), nn.Dropout(dropout)
    )
    x.apply(weights_init)
    return x

class SiameseNet(nn.Module):
    """Siamese network baseline with item-level metric learning."""

    def __init__(
        self,
        embd_dim=128,
        feat_norm=True,
        last_fc=True,
        learned_metric=False,
        margin=0.2,
        dropout=0.2,
        backbone=None,
        use_nn_feature=True,
    ):
        super().__init__()
        self.margin = margin
        self.backbone = nn.Identity() if use_nn_feature else backbone
        num_features = 512 if use_nn_feature else None
        if not use_nn_feature:
            raise ValueError("SiameseNet baseline requires pre-extracted features")
        self.encoder = nn.Sequential(
            embed_layer(num_features, embd_dim, dropout),
            (
                nn.Linear(embd_dim, embd_dim)
                if last_fc
                else _DiagonalLayer(embd_dim)
            ),
        )
        self.encoder.apply(weights_init)
        self.feat_norm = feat_norm
        self.metric = _Metric(embd_dim, learned_metric)

    def encode_item(self, x):
        h = self.encoder(x)
        if self.feat_norm:
            h = F.normalize(h, p=2, dim=-1)
        return h

    def outfit_score(self, x, cate=None):
        batch, num, *_ = x.shape
        masks = torch.ones(batch, num, device=x.device)
        if cate is not None:
            masks = (cate != -1).float()
        feats = self.encode_item(x.flatten(0, 1)).view(batch, num, -1)
        indx, indy = np.triu_indices(num, k=1)
        mask_x, mask_y = masks[:, indx], masks[:, indy]
        valid = mask_x * mask_y
        dist = self.metric(feats[:, indx], feats[:, indy]).view(batch, -1)
        score = -(dist * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        return score.view(-1)

    def train_compatibility(self, batch, device):
        anc, pos, neg = [t.to(device) for t in batch["data"]]
        feat_anc, feat_pos, feat_neg = self.encode_item(anc), self.encode_item(pos), self.encode_item(neg)
        dist_pos = self.metric(feat_anc, feat_pos).view(-1)
        dist_neg = self.metric(feat_anc, feat_neg).view(-1)
        ones = torch.ones_like(dist_neg)
        triplet = F.margin_ranking_loss(dist_neg, dist_pos, ones, self.margin, reduction="none")
        l2reg = feat_anc.norm(2, dim=-1)
        loss = ops.compute_weighted_losses(
            {"triplet": triplet.view(-1, 1), "l2reg": l2reg.view(-1, 1)}, {"triplet": 1.0, "l2reg": 1.0e-3}
        )[0]
        accuracy = torch.gt(dist_neg - dist_pos, 0).float().mean().item()
        return loss, {"accuracy": accuracy}

    @torch.no_grad()
    def eval_compatibility(self, batch, device):
        data = batch["data"].to(device)
        cate = batch["cate"].to(device)
        return self.outfit_score(data, cate)

    def setup_for_evaluation(self, data_cfg, device):
        pass


class _DiagonalLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(1, dim))
        nn.init.normal_(self.weight, 0.9, 0.7)

    def forward(self, x):
        x = x.view(-1, self.dim)
        return F.relu(self.weight * x)


class _Metric(nn.Module):
    def __init__(self, dim, learned_metric=False):
        super().__init__()
        self.learned_metric = learned_metric
        if learned_metric:
            self.func = nn.Linear(dim, 1)
        else:
            self.func = None

    def forward(self, x, y):
        if self.learned_metric:
            return self.func(x * y)
        z = (x - y) ** 2
        return z.sum(dim=-1, keepdim=True).sqrt()


class ConditionalSimNet(nn.Module):
    def __init__(self, num_conditions, embedding_size, learnedmask=True, prein=False):
        super().__init__()
        self.num_conditions = num_conditions
        self.mask_func = torch.nn.ReLU() if learnedmask else torch.nn.Identity()
        if learnedmask:
            if prein:
                mask_array = 0.1 * np.ones([num_conditions, embedding_size])
                mask_len = int(embedding_size / num_conditions)
                for i in range(num_conditions):
                    mask_array[i, i * mask_len : (i + 1) * mask_len] = 1
                self.masks = torch.nn.Parameter(torch.Tensor(mask_array))
            else:
                self.masks = nn.Parameter(torch.zeros(num_conditions, embedding_size))
                self.masks.data.normal_(0.9, 0.7)
        else:
            mask_array = np.zeros([num_conditions, embedding_size])
            mask_len = int(embedding_size / num_conditions)
            for i in range(num_conditions):
                mask_array[i, i * mask_len : (i + 1) * mask_len] = 1
            self.masks = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, x):
        masked_embedding = x * self.mask_func(self.masks)
        masked_embedding = F.normalize(masked_embedding, dim=-1)
        return masked_embedding


class ConceptBranch(nn.Module):
    def __init__(self, num_concepts, embedding_dim):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 2, num_concepts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.branch(x)


class SCENet(nn.Module):
    """Self-Coded Ensemble (SCE) network baseline with conditional similarity."""

    def __init__(
        self,
        embd_dim=128,
        num_concepts=16,
        learnedmask=True,
        prein=False,
        margin=0.2,
        dropout=0.2,
        backbone=None,
        use_nn_feature=True,
    ):
        super().__init__()
        self.margin = margin
        self.backbone = nn.Identity() if use_nn_feature else backbone
        num_features = 512 if use_nn_feature else None
        self.embedding = embed_layer(num_features, embd_dim, dropout)
        self.csn = ConditionalSimNet(num_concepts, embd_dim, learnedmask, prein)
        self.concept_branch = ConceptBranch(num_concepts, embd_dim * 2)
        self.num_concepts = num_concepts

    def scoring(self, feat, masks):
        batch_size, num = feat.shape[0], feat.shape[1]
        norm_feat = F.normalize(feat.unsqueeze(-2), dim=-1, p=2)
        indx, indy = np.triu_indices(num, k=1)
        xid, yid = np.hstack((indx, indy)), np.hstack((indy, indx))
        feat_x, feat_y = norm_feat[:, xid], norm_feat[:, yid]
        valid = masks[:, xid] * masks[:, yid]
        cond_latent_x = self.csn(feat_x)
        cond_latent_y = self.csn(feat_y)
        branch_input = torch.cat((feat_x, feat_y), dim=-1)
        weights = self.concept_branch(branch_input)
        latent_x = weights.matmul(cond_latent_x)
        latent_y = weights.matmul(cond_latent_y)
        dist = ((latent_x - latent_y) ** 2).sum(dim=-1).sqrt().squeeze(-1)
        score = -(dist * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        return score.view(-1)

    def train_compatibility(self, batch, device):
        anc, pos, neg = [t.to(device) for t in batch["data"]]
        feat_anc = self.embedding(anc).unsqueeze(1)
        feat_pos = self.embedding(pos).unsqueeze(1)
        feat_neg = self.embedding(neg).unsqueeze(1)
        l2reg = feat_anc.norm(2, dim=-1)
        general_x = F.normalize(feat_anc, dim=-1, p=2)
        general_y = F.normalize(feat_pos, dim=-1, p=2)
        general_z = F.normalize(feat_neg, dim=-1, p=2)
        feat_xy = torch.cat((general_x, general_y), dim=-1)
        feat_xz = torch.cat((general_x, general_z), dim=-1)
        weights_xy = self.concept_branch(feat_xy)
        weights_xz = self.concept_branch(feat_xz)
        embedded_x = self.csn(feat_anc)
        embedded_y = self.csn(feat_pos)
        embedded_z = self.csn(feat_neg)
        embedded_xy = weights_xy.matmul(embedded_x).squeeze(1)
        embedded_yx = weights_xy.matmul(embedded_y).squeeze(1)
        embedded_xz = weights_xz.matmul(embedded_x).squeeze(1)
        embedded_zx = weights_xz.matmul(embedded_z).squeeze(1)
        dist_a = F.pairwise_distance(embedded_xy, embedded_yx, 2)
        dist_b = F.pairwise_distance(embedded_xz, embedded_zx, 2)
        ones = torch.ones_like(dist_a)
        triplet = F.margin_ranking_loss(dist_b, dist_a, ones, self.margin, reduction="none")
        l1reg = self.csn.masks.norm(1, dim=-1)
        loss = ops.compute_weighted_losses(
            {"triplet": triplet, "l1reg": l1reg, "l2reg": l2reg},
            {"triplet": 1.0, "l1reg": 1.0e-3, "l2reg": 1.0e-3},
        )[0]
        accuracy = torch.gt(dist_b - dist_a, 0).float().mean().item()
        return loss, {"accuracy": accuracy}

    @torch.no_grad()
    def eval_compatibility(self, batch, device):
        data = batch["data"].to(device)
        cate = batch["cate"].to(device)
        batch_size, num = cate.shape
        masks = (cate != -1).float()
        feats = self.embedding(data.flatten(0, 1)).view(batch_size, num, -1)
        return self.scoring(feats, masks)

    def setup_for_evaluation(self, data_cfg, device):
        pass
