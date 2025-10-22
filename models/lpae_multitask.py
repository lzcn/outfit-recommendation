import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import utils
from models import blocks


class SetTransformer(nn.Module):
    """
    Encode a set of elements into a fixed-length representation:
        x -> h -> [SAB] -> d -> [Pool] -> z

    Supports multi-stage feature transformations for multitask learning:
        1) embedding_x : applies `dense_h` to transform h (item embeddings)
        2) embedding_f : applies `dense_d` to transform d (post-attention features)
        3) embedding_z : applies `dense_o` to transform z (set-level embedding)

    Notation:
        x : raw input features
        h : item embeddings (after dense_i and optional type embedding)
        d : contextualized item features after stacked SAB blocks
        z : aggregated set embedding after PMA (and optional SAB)
    """

    def __init__(
        self,
        in_features,
        embd_dim=128,
        num_types=None,
        type_encoding="none",  # "none", "learnable", "fixed"
        # set transformer parameters
        num_points=0,
        num_sab=2,
        num_seeds=1,
        num_heads=4,
        use_dense_h=True,
        use_dense_d=True,
        use_dense_o=True,
    ):
        super().__init__()
        # TODO: use better names
        self.embed_dim = embd_dim
        self.num_types = num_types
        # multi-task
        self.use_dense_h = use_dense_h
        self.use_dense_d = use_dense_d
        self.use_dense_o = use_dense_o

        # item embedding layer for all downstream tasks
        self.dense_i = blocks.embed_layer(in_features, embd_dim)

        # category embedding mode
        self.type_encoding = type_encoding
        if type_encoding == "learnable":
            self.dense_c = nn.Embedding(num_types, embd_dim)

        # extra dense layer for multi-task
        self.dense_h = blocks.dense_layer(embd_dim) if use_dense_h else None
        # extra dense layer for multi-task
        self.dense_d = blocks.dense_layer(embd_dim) if use_dense_d else None
        # extra dense layer for multi-task
        self.dense_o = blocks.dense_layer(embd_dim) if use_dense_o else None

        # x = iSAB(iSAB(x)) or SAB(SAB(x)) according to the number of points
        self.encoder = blocks.stack_sab(dim=embd_dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
        self.encoder_f = blocks.stack_sab(dim=embd_dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
        self.pma = blocks.AttentionPool(embd_dim, num_heads, num_seeds, ln=True)
        self.sab = blocks.SelfAttentionBlock(embd_dim, num_heads, ln=True) if num_seeds > 1 else nn.Identity()

    def embed_items(self, x: torch.Tensor, cate: torch.Tensor = None) -> torch.Tensor:
        """Compute item-level embeddings. Flow: x -> h."""
        x = self.dense_i(x)
        if cate is None or self.type_encoding == "none":
            return x

        b, n = cate.shape[:2]
        if self.type_encoding == "learnable":
            cate = torch.where(cate == -1, torch.zeros_like(cate), cate)
            c = self.dense_c(cate.view(b, n))
        elif self.type_encoding == "fixed":
            c = blocks.position_encoding(cate.view(b, n, 1), self.embed_dim, self.num_types)
        else:
            return x

        return x + c

    def encode_set(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Flow: h -> [SAB] -> f"""
        # Encode items with stacked self-attention blocks
        for encoder in self.encoder:
            h = encoder(h, mask)
        return h

    def pool_set(self, f: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Flow: f -> [PMA + SAB] -> z"""
        return self.sab(self.pma(f, mask))

    def embedding_x(self, item: torch.Tensor, cate: torch.Tensor = None, use_dense=False) -> torch.Tensor:
        h = self.embed_items(item, cate)
        return self.dense_h(h) if use_dense else h

    def embedding_f(self, item: torch.Tensor, cate: torch.Tensor = None, use_dense=False) -> torch.Tensor:
        h = self.embed_items(item, cate)
        mask = (cate != -1).float()
        f = self.encode_set(h, mask)
        for encoder in self.encoder_f:
            f = encoder(f, mask)
        return f
        return self.dense_d(f) if use_dense else f

    def embedding_z(self, item: torch.Tensor, cate: torch.Tensor = None, use_dense=False) -> torch.Tensor:
        """Outfit embeddings for downstream tasks.
        Args:
           item (torch.Tensor): item, shape of b x n x d.
           cate (torch.Tensor): cate. shape of b x n x 1.

        Return:
            torch.Tensor: outfit embedding shape of b x 1 x d
        """
        mask = (cate != -1).float() if cate is not None else None
        # general embedding. b x n x d
        h = self.embed_items(item, cate)
        f = self.encode_set(h, mask)
        z = self.pool_set(f, mask)
        return self.dense_o(z) if use_dense else z


class Predictor(nn.Module):
    def __init__(
        self,
        num_users=1,
        embd_dim=128,
        num_seeds=1,
        num_anchors=16,  # number of anchors per type
        use_g_anchors=False,
        use_u_anchors=True,
        use_g_lambda=False,
        use_u_lambda=False,
        use_z_score=False,
        kappa_i=10.0,
        kappa_u=1.0,
        logdet=True,
        num_u_neg=8,
        cold_start=False,
    ):
        super().__init__()
        self.embd_dim = embd_dim
        self.num_u_neg = num_u_neg
        self.logdet = logdet
        self.kappa_i = kappa_i
        self.kappa_u = kappa_u
        self.num_users = num_users
        self.num_seeds = num_seeds
        self.num_anchors = num_anchors
        self.use_g_anchors = use_g_anchors
        self.use_u_anchors = use_u_anchors
        self.use_g_lambda = use_g_lambda
        self.use_u_lambda = use_u_lambda
        self.use_z_score = use_z_score
        # build anchors
        assert use_g_anchors or use_u_anchors, "At least one of user or global anchors must be enabled."
        scale = math.sqrt(3.0 / embd_dim)

        if use_g_anchors:
            self.g_anchors = nn.Parameter(torch.empty(1, num_anchors, embd_dim))
            nn.init.uniform_(self.g_anchors, -scale, scale)
            self.num_g_anchors = num_anchors
        else:
            self.g_anchors = None
            self.num_g_anchors = 0

        if use_u_anchors:
            self.u_anchors = nn.Parameter(torch.empty(num_users, num_anchors, embd_dim))
            nn.init.uniform_(self.u_anchors, -scale, scale)
            self.num_u_anchors = num_anchors
        else:
            self.u_anchors = None
            self.num_u_anchors = 0

        # lambdas (learnable or fixed)
        if use_u_lambda:
            self.lambda_u = nn.Parameter(torch.ones(1), requires_grad=not cold_start)
        if use_g_lambda:
            self.lambda_g = nn.Parameter(torch.ones(1), requires_grad=not cold_start)

    def reg(self):
        """Compute regularization for selected users."""
        if self.num_anchors <= 1:
            return None
        gram = self.u_anchors @ self.u_anchors.transpose(1, 2)
        if self.logdet:
            gram = gram + 1e-6 * torch.eye(gram.size(-1), device=gram.device)
            sign, logdet = torch.linalg.slogdet(gram)
            logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
            reg = (gram.diagonal(dim1=1, dim2=2).sum() - logdet.sum()) / self.num_users
        else:
            eye = torch.eye(self.num_anchors, device=gram.device)
            reg = ((gram * (1 - eye)) ** 2).sum() / self.num_users
        return reg

    def score_r(self, uidx: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)  # (B, S, D)

        if self.use_u_anchors:
            u = self.u_anchors.index_select(0, uidx)  # (B, M, D)
            u = F.normalize(u, dim=-1)  # (B, M, D)
            u_score = z.matmul(u.transpose(1, 2)).mean(dim=-1)
        else:
            u_score = 0.0

        if self.use_g_anchors:
            g = self.g_anchors
            g = F.normalize(g, dim=-1)
            g_score = z.matmul(g.transpose(1, 2)).mean(dim=-1)
        else:
            g_score = 0.0

        u_lambda = self.lambda_u.abs() if self.use_u_lambda else 1.0
        g_lambda = self.lambda_g.abs() if self.use_g_lambda else 1.0

        score = self.kappa_i * (u_lambda * u_score + g_lambda * g_score)
        return score

    def score_c(self, uidx: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        z: outfit with shape (B, 1, D)
        h: items with shape (B, M, D), where M is the number of items
        """
        # Normalize to compute cosine similarity
        z = F.normalize(z, dim=-1)  # (B, 1, D)
        h = F.normalize(h, dim=-1)  # (B, M, D)

        if self.use_u_anchors:
            u = self.u_anchors.index_select(0, uidx)  # (B, S, D)
            u = F.normalize(u, dim=-1)  # (B, S, D)
            u_term = h.matmul(u.transpose(1, 2)).mean(dim=-1)  # (B, M)
        else:
            u_term = 0.0

        u_lambda = self.lambda_u.abs() if self.use_u_lambda else 1.0

        z_term = h.matmul(z.transpose(1, 2)).mean(dim=-1)  # (B, M)
        score = self.kappa_i * (z_term + u_lambda * u_term)

        return score

    def score_d(self, uidx: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h = F.normalize(h, dim=-1)
        if self.use_u_anchors:
            u = F.normalize(self.u_anchors.index_select(0, uidx), dim=-1)
            u_score = h.matmul(u.transpose(1, 2)).mean(dim=-1)
        else:
            u_score = 0.0
        if self.use_g_anchors:
            g = F.normalize(self.g_anchors, dim=-1)
            g_score = h.matmul(g.transpose(1, 2)).mean(dim=-1)
        else:
            g_score = 0.0
        u_lambda = self.lambda_u.abs() if self.use_u_lambda else 1.0
        g_lambda = self.lambda_g.abs() if self.use_g_lambda else 1.0
        return self.kappa_i * (u_lambda * u_score + g_lambda * g_score)


class LPAENet(nn.Module):
    """LPAE-Net."""

    def __init__(
        self,
        # core counts
        num_users: int = 1,
        num_types: int = 1,
        # vision backbone config
        backbone: Dict = None,
        embd_dim: int = 128,
        # anchors
        multi_anchors: bool = False,
        num_anchors: int = 16,
        use_g_anchors: bool = False,
        use_u_anchors: bool = True,
        use_g_lambda: bool = False,
        use_u_lambda: bool = False,
        # set transformer
        num_points: int = 0,
        num_sab: int = 2,
        num_seeds: int = 1,
        num_heads: int = 4,
        type_encoding: str = "none",  # "none", "learnable"
        # scoring
        kappa_i: float = 10.0,
        logdet: bool = True,
        # training
        loss_type: str = "bpr",
        loss_weight: Dict = None,
        use_nn_feature: bool = False,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_types = num_types
        self.embd_dim = embd_dim
        self.num_seeds = num_seeds
        self.loss_type = loss_type.lower()
        # update default loss weight
        self.loss_weight = utils.update_loss_weights(loss_weight, nll_c=1.0, nll_d=1.0, cl_o=1.0)

        # type_encoding = "learnable" if learnable_type_embed else "fixed" if use_type_embed else "none"
        in_features = backbone.pop("dim")
        self.visual_feature = nn.Identity() if use_nn_feature else utils.backbone(**backbone)
        self.visual_encoder = SetTransformer(
            in_features=in_features,
            embd_dim=embd_dim,
            num_types=num_types,
            type_encoding=type_encoding,
            num_points=num_points,
            num_sab=num_sab,
            num_seeds=num_seeds,
            num_heads=num_heads,
        )
        kwargs = dict(
            num_users=num_users,
            embd_dim=embd_dim,
            num_seeds=num_seeds,
            num_anchors=num_anchors,
            use_g_anchors=use_g_anchors,
            use_u_anchors=use_u_anchors,
            use_g_lambda=use_g_lambda,
            use_u_lambda=use_u_lambda,
            kappa_i=kappa_i,
            logdet=logdet,
        )
        self.anchors = Predictor(**kwargs)
        self.multi_anchors = multi_anchors
        if multi_anchors:
            self.anchors_c = Predictor(**kwargs)
            self.anchors_d = Predictor(**kwargs)
            self.anchors_r = Predictor(**kwargs)
        else:
            self.anchors = Predictor(**kwargs)

    def outfit_completion(
        self,
        uidx: torch.Tensor,
        data: torch.Tensor,
        cate: torch.Tensor,
        pool_items: torch.Tensor,
        pool_cates: torch.Tensor,
    ):
        """Outfit completion task.

        Args:
            uidx (torch.Tensor): uidx
            data (torch.Tensor): data
            cate (torch.Tensor): cate
            pool_items (torch.Tensor): pool_items
            pool_cates (torch.Tensor): pool_cates
            only_u_score (bool): only_u_score
        """
        predictor = self.anchors_c if self.multi_anchors else self.anchors
        batch, num_items, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch, num_items, -1)
        cate = cate.view(batch, num_items, -1)
        z = self.visual_encoder.embedding_z(feat, cate, use_dense=True)
        q = self.visual_encoder.embedding_x(pool_items, pool_cates, use_dense=True)
        scores = predictor.score_c(uidx, z, q)
        if not self.training:
            return scores
        acc = (torch.argmax(scores, dim=-1) == 0).float().mean()
        logprob = -F.log_softmax(scores, dim=-1)
        nll = logprob[:, 0].mean()
        loss = dict(nll_c=nll)
        return loss, acc

    def outlier_detection(
        self,
        uidx: torch.Tensor,
        data: torch.Tensor,
        cate: torch.Tensor,
        yidx: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        predictor = self.anchors_d if self.multi_anchors else self.anchors
        batch, num_items, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch, num_items, -1)
        cate = cate.view(batch, num_items, 1)
        mask = cate.view(batch, num_items) == -1
        g = self.visual_encoder.embedding_f(feat, cate, use_dense=True)
        score = -predictor.score_d(uidx, g)
        value = torch.finfo(score.dtype).min
        score.masked_fill_(mask, value)
        if not self.training:
            return score
        logprob = F.log_softmax(score, dim=-1)
        nll = F.nll_loss(logprob, yidx)
        correct = torch.argmax(score, dim=-1) == yidx
        acc = correct.float().mean()
        loss = dict(nll_d=nll)
        return loss, acc

    def outfit_compatibility(self, uidx, data, cate):
        predictor = self.anchors_r if self.multi_anchors else self.anchors
        if not self.training:
            batch, num, *shape = data.shape
            data = data.view(-1, *shape)
            cate = cate.view(batch, num, -1)
            feat = self.visual_feature(data)
            feat = feat.view(batch, num, -1)
            z = self.visual_encoder.embedding_z(feat, cate, use_dense=False)
            scores = predictor.score_r(uidx, z)
            return scores

        batch, num_outfits, num_items, *shape = data.shape
        flat = data.view(-1, *shape)
        feat = self.visual_feature(flat)
        feat = feat.view(batch * num_outfits, num_items, -1)
        cate = cate.view(batch * num_outfits, num_items, 1)
        z = self.visual_encoder.embedding_z(feat, cate, use_dense=False)
        z = z.reshape(batch, num_outfits * self.num_seeds, self.embd_dim)
        scores = predictor.score_r(uidx, z)

        if self.loss_type == "n-pair":
            neg_log_prob = -F.log_softmax(scores, dim=-1)
            cl_o = neg_log_prob[:, 0].mean()
            accuracy = (torch.argmax(scores, dim=-1) == 0).float().mean()
        elif self.loss_type == "bpr":
            pos_score, neg_score = scores[:, 0], scores[:, 1]
            diff_o = (pos_score.view(-1, 1) - neg_score.view(1, -1)).view(-1)
            cl_o = F.soft_margin_loss(diff_o, torch.ones_like(diff_o), reduction="mean")
            accuracy = (diff_o > 0.0).float().mean()
        elif self.loss_type == "flat-nce":
            pos_score, neg_score = scores[:, 0], scores[:, 1]
            diff_o = -(pos_score.view(-1, 1) - neg_score.view(1, -1))
            clogits_o = torch.logsumexp(diff_o, dim=-1)
            cl_o = torch.exp(clogits_o - clogits_o.detach()).mean()
            accuracy = (diff_o < 0.0).float().mean()
        elif self.loss_type == "info-nce":
            pos_score, neg_score = scores[:, 0], scores[:, 1]
            pos_score = pos_score.view(-1, 1)
            neg_score = neg_score.view(-1, 1)
            scores = torch.cat((pos_score, neg_score.repeat(1, batch)), dim=-1)
            neg_log_prob = -F.log_softmax(scores, dim=-1)
            cl_o = neg_log_prob[:, 0].mean()
            accuracy = ((pos_score - neg_score) > 0.0).float().mean()
        else:
            raise KeyError(f"Unknown loss type: {self.loss_type}")

        loss = {"cl_o": cl_o}
        if self.loss_weight.get("reg") is not None and self.anchors_r.num_u_anchors > 1:
            loss["reg"] = self.anchors_r.reg()
        return loss, accuracy

    def train_multitask(self, batch, device):
        uidx = batch["uidx"].to(device)
        data_d = batch["data_d"].to(device)
        data_c = batch["data_c"].to(device)
        data_p = batch["data_p"].to(device)
        data_r = batch["data_r"].to(device)
        cate_d = batch["cate_d"].to(device)
        cate_c = batch["cate_c"].to(device)
        cate_p = batch["cate_p"].to(device)
        cate_r = batch["cate_r"].to(device)
        y_d = batch["y_d"].to(device)

        loss_d, acc_d = self.outlier_detection(uidx, data_d, cate_d, y_d)
        loss_c, acc_c = self.outfit_completion(uidx, data_c, cate_c, data_p, cate_p)
        loss_r, acc_r = self.outfit_compatibility(uidx, data_r, cate_r)

        loss = {}
        loss.update(loss_c)
        loss.update(loss_r)
        loss.update(loss_d)

        # weighted sum
        overall_loss = 0.0
        for key, value in loss.items():
            weight = self.loss_weight.get(key, 1.0)
            overall_loss = overall_loss + weight * value.mean()

        metrics = {k: v.float().mean().item() for k, v in loss.items()}
        metrics.update({"accuracy_d": acc_d.item(), "accuracy_c": acc_c.item(), "accuracy_r": acc_r.item()})
        return overall_loss, metrics

    def train_compatibility(self, batch, device):
        # batch_size
        uidx = utils.to(batch["uidx"], device)
        # batch_size x num_outfits x num_items
        data = utils.to(batch["data"], device)
        # batch_size x num_outfits x num_items
        cate = utils.to(batch["cate"], device)
        loss, accuracy = self.outfit_compatibility(uidx, data, cate)
        overall_loss, metrics = utils.compute_weighted_losses(loss, self.loss_weight)
        metrics["accuracy"] = accuracy.item()
        return overall_loss, metrics

    @torch.no_grad()
    def eval_compatibility(self, batch, device):
        data = utils.to(batch["data"], device)
        uidx = utils.to(batch["uidx"], device)
        cate = utils.to(batch["cate"], device)
        return self.outfit_compatibility(uidx, data, cate)

    def train_completion(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        pool_data = utils.to(batch["pool_data"], device)
        pool_cate = utils.to(batch["pool_cate"], device)
        loss, accuracy = self.outfit_completion(uidx, data, cate, pool_data, pool_cate)
        overall_loss, metrics = utils.compute_weighted_losses(loss, self.loss_weight)
        metrics["accuracy"] = accuracy.item()
        return overall_loss, metrics

    @torch.no_grad()
    def eval_completion(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        pool_data = utils.to(batch["pool_data"], device)
        pool_cate = utils.to(batch["pool_cate"], device)
        return self.outfit_completion(uidx, data, cate, pool_data, pool_cate)

    def train_outlier(self, batch, device):
        data = utils.to(batch["data"], device)
        uidx = utils.to(batch["uidx"], device)
        cate = utils.to(batch["cate"], device)
        y = utils.to(batch["y"], device)
        loss, accuracy = self.outlier_detection(uidx, data, cate, y)
        overall_loss, metrics = utils.compute_weighted_losses(loss, self.loss_weight)
        metrics["accuracy"] = accuracy.item()
        return overall_loss, metrics

    @torch.no_grad()
    def eval_outlier(self, batch, device):
        data = utils.to(batch["data"], device)
        uidx = utils.to(batch["uidx"], device)
        cate = utils.to(batch["cate"], device)
        score = self.outlier_detection(uidx, data, cate)
        return score
