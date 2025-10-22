import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import layers, losses

logger = utils.get_logger(__name__)


def embed_layer(in_features, out_features, dropout=0.2):
    layer = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )
    for m in layer:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return layer


class ContrastiveEmbeddingLoss(nn.Module):
    def __init__(self, i_dim, t_dim, c_dim, margin=0.2, word_embedding=False):
        """Compute the vse loss.

        Args:
            i_dim (int): dimension for image data
            t_dim (int): dimension for text data
            c_dim (int): dimension for embedding space
            margin (float, optional): margin for loss. Defaults to 0.2.
            word_embedding (bool, optional): whether the input is word embedding. Defaults to False.
        """
        super().__init__()
        self.i_dim = i_dim
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.margin = margin
        self.word_embedding = word_embedding
        self.Wi = nn.Linear(i_dim, c_dim, bias=False)
        self.Wt = nn.Linear(t_dim, c_dim, bias=False)

    def forward(self, i_data: torch.Tensor, t_data: torch.Tensor, types: torch.Tensor = None) -> torch.Tensor:
        # i_data: shape (b x n x d) or (b x d)
        # t_data: shape (b x n x d) or (b x d)
        # types: shape (b x n) or b
        if self.word_embedding:
            # normalize by number of words
            t_norm = t_data.sum(dim=-1, keepdim=True)
            t_data = t_data / t_norm.clamp_min(1.0e-12)
            if types is None:
                # has at least one word
                mask = (t_norm > 0.0).view(-1, 1)
            else:
                # has at least one word and valid type
                # although we can use non-valid type as long as there is one-to-one
                # mapping between image data and text data, it may introduce im-balance
                # for the final loss
                mask = (types.view(-1, 1) != -1) * (t_norm > 0.0).view(-1, 1)
        else:
            mask = None if types is None else (types.view(-1, 1) != -1) * 1.0
        i_feat = self.Wi(i_data).view(-1, self.c_dim)
        t_feat = self.Wt(t_data).view(-1, self.c_dim)
        loss = losses.contrastive_loss(i_feat, t_feat, self.margin, mask=mask, reduction="mean")
        return loss


class DiagonalLayer(nn.Module):
    def __init__(
        self,
        num_conditions,
        dim,
        learned_mask=True,
        preinit_mask=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_conditions = num_conditions
        if learned_mask:
            if preinit_mask:
                # initialize masks
                mask_array = 0.1 * np.ones([num_conditions, dim])
                # divide embedding to a n-by-n block matrix
                mask_len = int(dim / num_conditions)
                # set the diagonal blocks to ones
                for i in range(num_conditions):
                    mask_array[i, i * mask_len : (i + 1) * mask_len] = 1.0
                self.masks = torch.nn.Parameter(torch.Tensor(mask_array))
            else:
                # initialize weights
                self.masks = nn.Parameter(torch.zeros(num_conditions, dim))
                self.masks.data.normal_(0.9, 0.7)  # 0.1, 0.005
        else:
            # initialize masks
            mask_array = np.zeros([num_conditions, dim])
            mask_len = int(dim / num_conditions)
            for i in range(num_conditions):
                mask_array[i, i * mask_len : (i + 1) * mask_len] = 1.0
            # no gradients for the masks
            self.masks = nn.Parameter(torch.Tensor(mask_array), requires_grad=False)
        self.relu = nn.ReLU() if learned_mask else nn.Identity()

    @torch.no_grad()
    def reset(self, i):
        assert -1 < i < self.num_conditions
        self.masks[i].data.fill_(1.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        masks = self.masks.index_select(0, c)
        masks = self.relu(masks)
        out = masks * x
        return out, masks


class LinearLayer(nn.Module):
    def __init__(self, num_conditions, dim):
        super().__init__()
        self.num_conditions = num_conditions
        self.dim = dim
        self.masks = nn.Parameter(0.1 * torch.randn(num_conditions, dim, dim))
        self.bias = nn.Parameter(torch.zeros(num_conditions, dim, 1))

    @torch.no_grad()
    def reset(self, i):
        assert -1 < i < self.num_conditions
        mat = torch.eye(self.dim)
        self.masks[i].data.copy_(mat.data)
        self.bias[i].data.fill_(0.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # n x d x 1
        x = x.view(-1, self.dim, 1)
        # n x d x d
        masks = self.masks.index_select(0, c)
        # n x d x 1
        bias = self.bias.index_select(0, c)
        # n x d x 1
        out = F.relu(masks.matmul(x) + bias)
        return out.squeeze(-1), masks


class TypeNet(nn.Module):
    """Type-specific embedding wrapper.

    Args:
        embd_size (int): Feature dimension D.
        num_conditions (int): Number of condition IDs (K).
        l2_embed (bool): Apply L2 normalization to outputs.
        linear_mask (bool): Use full matrix masks (LinearLayer) if True,
            else diagonal/elementwise masks (DiagonalLayer).
        learned_mask (bool): Whether masks are learnable (passed to encoder).
        preinit_mask (bool): Whether to pre-initialize masks to block-diagonal (passed to encoder).
        norm_eps (float): Epsilon for numerical stability in normalization.
        norm_p (int): p for F.normalize (default L2).
        norm_dim (int): Dimension to normalize along (default last).
    """

    def __init__(
        self,
        embd_size: int,
        num_conditions: int,
        l2_embed: bool = True,
        linear_mask: bool = False,
        learned_mask: bool = True,
        preinit_mask: bool = True,
    ):
        super().__init__()

        self.normalize = l2_embed

        if linear_mask:
            self.encoder = LinearLayer(num_conditions=num_conditions, dim=embd_size)
        else:
            self.encoder = DiagonalLayer(
                num_conditions=num_conditions,
                dim=embd_size,
                learned_mask=learned_mask,
                preinit_mask=preinit_mask,
            )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        out, mask = self.encoder(x, c)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out, mask


class CSNet(nn.Module):
    def __init__(
        self,
        backbone: dict = None,
        embd_dim: int = 128,
        l2_embed: bool = False,
        learned_metric: bool = False,
        linear_mask: bool = False,
        learned_mask: bool = True,
        preinit_mask: bool = False,
        loss_weight: dict = None,
        margin: float = 0.2,
        dropout: float = 0.2,
        num_types: int = 3,
        use_nn_feature: bool = False,
        word_embedding: bool = False,
        semantic_dim: int = 2400,
        vse_dim: int = 128,
        use_vse: bool = False,
        vse_margin: float = 0.2,
    ):
        super().__init__()
        self.embd_dim = embd_dim
        self.learned_metric = learned_metric
        self.use_vse = use_vse
        self.margin = margin
        self.num_types = num_types
        num_conditions = num_types * num_types

        self.register_buffer("condition_index", torch.zeros(num_types, num_types, dtype=torch.int64))
        self.register_buffer("condition_masks", torch.zeros(num_types, num_types, dtype=torch.bool))

        for i in range(num_types):
            for j in range(i, num_types):
                c1 = i * num_types + j
                c2 = j * num_types + i
                self.condition_index[i, j] = min(c1, c2)

        # get backbone
        num_features = backbone.pop("dim")
        self.backbone = nn.Identity() if use_nn_feature else utils.backbone(**backbone)

        # embedding
        self.embeddingnet = embed_layer(num_features, embd_dim, dropout)

        self.encoder = TypeNet(
            embd_size=embd_dim,
            num_conditions=num_conditions,
            l2_embed=l2_embed,
            linear_mask=linear_mask,
            learned_mask=learned_mask,
            preinit_mask=preinit_mask,
        )
        # learn similarity with linear layer
        if learned_metric:
            self.metric = nn.Linear(embd_dim, 1)
        if use_vse:
            self.vse = layers.VisualSemanticEmbedding(
                i_dim=num_features,
                t_dim=semantic_dim,
                c_dim=vse_dim,
                margin=vse_margin,
                bow=word_embedding,
            )
        self.loss_weight = loss_weight or {}

        self.logger = logging.getLogger(__name__)
        # cache
        self.name_dict = dict()
        self.item_feat = dict()
        self.item_cache = dict()
        self.name_cache = set()
        self.feat_cache = dict()

    @torch.no_grad()
    def cache_batch_feature(self, item_data: torch.Tensor, item_name: List[str], item_size: torch.Tensor):
        batch, num, *shape = item_data.shape
        item_data = item_data.view(batch * num, *shape)
        item_data = self.backbone(item_data)
        item_data = self.embeddingnet(item_data)
        item_feat = item_data.view(batch, num, -1)
        for feat, name, size in zip(item_feat, item_name, item_size):
            name = name.split(",")
            for i in range(size):
                self.name_cache.add(name[i])
                self.item_cache[name[i]] = feat[i]

    def build_featture_cache(self):
        name_dict = dict()
        feat_list = []
        for i, key in enumerate(self.name_cache):
            name_dict[key] = i
            feat_list.append(self.item_cache[key])
        self.item_feat = torch.stack(feat_list, dim=0)
        self.item_cache = dict()
        self.name_dict = name_dict
        self.logger.info("Item shape: {}".format(self.item_feat.shape))

    @torch.no_grad()
    def compatibility(self, item_names: List[str], item_types: torch.Tensor):
        batch, num = item_types.shape
        item_masks = (item_types != -1) * 1.0
        index = [[self.name_dict[key] for key in names.split(",")] for names in item_names]
        index = torch.LongTensor(index).view(-1)
        item_feats = self.item_feat[index].view(batch, num, -1)
        # batch x pairs
        indx, indy = np.triu_indices(num, k=1)
        xid = np.hstack((indx, indy))
        yid = np.hstack((indy, indx))
        items_x = item_feats[:, xid]
        items_y = item_feats[:, yid]
        masks_x = item_masks[:, xid]
        masks_y = item_masks[:, yid]
        types_x = item_types[:, xid]
        types_y = item_types[:, yid]
        item_masks = masks_x * masks_y
        item_masks = item_masks.view(batch, -1)
        conditions = self.condition_index[types_x, types_y].view(-1)
        feat_x, _ = self.encoder(items_x.view(-1, self.embd_dim), conditions)
        feat_y, _ = self.encoder(items_y.view(-1, self.embd_dim), conditions)
        if self.learned_metric:
            dist = self.metric(feat_x * feat_y)
        else:
            dist = F.pairwise_distance(feat_x, feat_y, 2)
        dist = dist.view(batch, -1)
        score = -(dist * item_masks).sum(dim=-1) / torch.clip(item_masks.sum(dim=-1), min=1.0)
        return score.view(-1, 1)

    def eval(self):
        super().eval()
        condition_set = set(self.condition_index[self.condition_masks].tolist())
        num_used_conditions = len(condition_set)
        logger.info("Number of conditions used: %d", num_used_conditions)
        for i in range(self.num_types):
            for j in range(self.num_types):
                if not self.condition_masks[i, j]:
                    self.encoder.encoder.reset(self.condition_index[i, j])

    @torch.no_grad()
    def eval_suggestion(self, items: torch.Tensor, types: torch.Tensor):
        masks = (types != -1) * 1.0
        batch, num, *shape = items.shape
        items = items.view(batch * num, *shape)
        items = self.backbone(items)
        items = self.embeddingnet(items)
        items = items.view(batch, num, -1)
        xid = []
        yid = []
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                xid.append(i)
                yid.append(j)
        # n x (n - 1)
        items_x = items[:, xid]
        items_y = items[:, yid]
        masks_x = masks[:, xid]
        masks_y = masks[:, yid]
        types_x = types[:, xid]
        types_y = types[:, yid]
        conditions_masks = self.condition_masks[types_x, types_y]
        masks = masks_x * masks_y * conditions_masks
        masks = masks.view(batch, -1)
        conditions = self.condition_index[types_x, types_y].view(-1)
        feat_x, _ = self.encoder(items_x.view(-1, self.embd_dim), conditions)
        feat_y, _ = self.encoder(items_y.view(-1, self.embd_dim), conditions)
        if self.learned_metric:
            dists = self.metric(feat_x * feat_y)
        else:
            dists = F.pairwise_distance(feat_x, feat_y, 2)
        dists = dists.view(batch, num, num - 1)
        masks = masks.view(batch, num, num - 1)
        # batch x num
        sim = -(dists * masks).sum(dim=-1) / torch.clip(masks.sum(dim=-1), min=1.0)
        sim = sim.cpu()
        index = torch.where(torch.argmin(sim, dim=-1) == 0, torch.ones(len(sim)), torch.zeros(len(sim)))
        return index.view(-1, 1)

    def eval_batch(self, items: torch.Tensor, types: torch.Tensor):
        masks = (types != -1) * 1.0
        batch, num, *shape = items.shape
        items = items.view(batch * num, *shape)
        items = self.backbone(items)
        items = self.embeddingnet(items)
        items = items.view(batch, num, -1)
        indx, indy = np.triu_indices(num, k=1)
        xid = np.hstack((indx, indy))
        yid = np.hstack((indy, indx))
        items_x = items[:, xid]
        items_y = items[:, yid]
        masks_x = masks[:, xid]
        masks_y = masks[:, yid]
        types_x = types[:, xid]
        types_y = types[:, yid]
        masks = masks_x * masks_y  # * (types_x != types_y)
        masks = masks.view(batch, -1)
        conditions = self.condition_index[types_x, types_y].view(-1)
        feat_x, _ = self.encoder(items_x.view(-1, self.embd_dim), conditions)
        feat_y, _ = self.encoder(items_y.view(-1, self.embd_dim), conditions)
        if self.learned_metric:
            dist = self.metric(feat_x * feat_y)
        else:
            dist = F.pairwise_distance(feat_x, feat_y, 2)
        dist = dist.view(batch, -1)
        score = -(dist * masks).sum(dim=-1) / torch.clip(masks.sum(dim=-1), min=1.0)
        return score.view(-1, 1)

    def eval_item_score(self, items: torch.Tensor, types: torch.Tensor):
        masks = (types != -1) * 1.0
        batch, num, *shape = items.shape
        items = items.view(batch * num, *shape)
        items = self.backbone(items)
        items = self.embeddingnet(items)
        items = items.view(batch, num, -1)
        xid = []
        yid = []
        for i in range(num):
            for j in range(num):
                if i != j:
                    xid.append(i)
                    yid.append(j)
        xid = np.array(xid)
        yid = np.array(yid)
        items_x = items[:, xid]
        items_y = items[:, yid]
        masks_x = masks[:, xid]
        masks_y = masks[:, yid]
        types_x = types[:, xid]
        types_y = types[:, yid]
        valid = masks_x * masks_y * (types_x != types_y)
        conditions = self.condition_index[types_x, types_y].view(-1)
        feat_x, _ = self.encoder(items_x.view(-1, self.embd_dim), conditions)
        feat_y, _ = self.encoder(items_y.view(-1, self.embd_dim), conditions)
        if self.learned_metric:
            dist = self.metric(feat_x * feat_y)
        else:
            dist = F.pairwise_distance(feat_x, feat_y, 2)
        score = dist.view(batch, num, num - 1)
        valid = valid.view(batch, num, num - 1)
        # b x n
        score = (score * valid).sum(dim=-1) / torch.clip(valid.sum(dim=-1), min=1.0)
        return score

    def train_batch(self, triplet: torch.Tensor, types: torch.Tensor):
        for i, j in types:
            self.condition_masks[i, j] = True
        condition = self.condition_index[types[:, 0], types[:, 1]].view(-1)
        # extract features from backbone
        if self.use_vse:
            anc, pos, neg = triplet[0]
            anc = self.backbone(anc)
            pos = self.backbone(pos)
            neg = self.backbone(neg)
            t_data = torch.stack(triplet[1], dim=1)
            i_data = torch.stack([anc, pos, neg], dim=1)
            loss = {"vse": self.vse(i_data, t_data).view(-1, 1)}
        else:
            anc, pos, neg = triplet
            anc = self.backbone(anc)
            pos = self.backbone(pos)
            neg = self.backbone(neg)
            loss = {}
        # general embedding
        data_anc = self.embeddingnet(anc)
        data_pos = self.embeddingnet(pos)
        data_neg = self.embeddingnet(neg)
        # l2-reg on general embedding
        l2reg = (data_anc.norm(2, dim=-1) + data_pos.norm(2, dim=-1) + data_neg.norm(2, dim=-1)) / 3.0
        # type-aware embedding
        feat_anc, mask_anc = self.encoder(data_anc, condition)
        feat_pos, mask_pos = self.encoder(data_pos, condition)
        feat_neg, mask_neg = self.encoder(data_neg, condition)
        # l1-reg on mask
        l1reg = (mask_anc.norm(1, dim=-1) + mask_pos.norm(1, dim=-1) + mask_neg.norm(1, dim=-1)) / 3.0
        # calculate the similarity loss on the general embedding
        dist_pos_neg = F.pairwise_distance(data_pos, data_neg, p=2)
        dist_pos_anc = F.pairwise_distance(data_pos, data_anc, p=2)
        dist_neg_anc = F.pairwise_distance(data_neg, data_anc, p=2)
        # items from same category (pos and neg) is more similar
        ones = torch.ones_like(dist_pos_neg)
        # d(pos, anc) > d(pos, neg)
        loss_pos_anc = F.margin_ranking_loss(dist_pos_anc, dist_pos_neg, ones, self.margin, reduction="none")
        # d(neg, anc) > d(pos, neg)
        loss_neg_anc = F.margin_ranking_loss(dist_neg_anc, dist_pos_neg, ones, self.margin, reduction="none")
        loss_sim = (loss_pos_anc + loss_neg_anc) / 2.0
        # calculate the tripelt loss
        if self.learned_metric:
            dist_anc_pos = self.metric(feat_anc * feat_pos)
            dist_anc_neg = self.metric(feat_anc * feat_neg)
        else:
            dist_anc_pos = F.pairwise_distance(feat_anc, feat_pos, 2)
            dist_anc_neg = F.pairwise_distance(feat_anc, feat_neg, 2)
        # d(anc, neg) > d(anc, pos)
        loss_triplet = F.margin_ranking_loss(dist_anc_neg, dist_anc_pos, ones, self.margin, reduction="none")
        loss.update(triplet=loss_triplet, l1reg=l1reg, l2reg=l2reg, sim=loss_sim)
        accuracy = dict(accuracy=torch.gt(dist_anc_neg - dist_anc_pos, 0))
        return loss, accuracy

    def forward(self, *inputs):
        if self.training:
            return self.train_batch(*inputs)
        return self.eval_batch(*inputs)

    def train_step(self, batch: dict, device):
        """Train step for compatibility task."""
        data = utils.to(batch["data"], device)
        types = utils.to(batch["types"], device)
        loss, accuracy = self.train_batch(data, types)
        total_loss = sum(self.loss_weight.get(k, 1.0) * v.mean() for k, v in loss.items())
        loss = {k: v.mean().item() for k, v in loss.items()}
        accuracy = {k: v.float().mean().item() for k, v in accuracy.items()}
        metrics = {**loss, **accuracy}
        return total_loss, metrics

    def eval_compatibility(self, batch: dict, device):
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        score = self.eval_batch(data, cate)
        return score

    def eval_outlier(self, batch: dict, device):
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        score = self.eval_item_score(data, cate)
        return score
