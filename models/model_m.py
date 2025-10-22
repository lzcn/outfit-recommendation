import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

import utils
from models import blocks, ops


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
        if type_encoding == "learnable" and num_types is not None:
            self.dense_c = nn.Embedding(num_types, embd_dim)

        # extra dense layer for multi-task
        self.dense_h = (
            blocks.stack_sab(dim=embd_dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
            if use_dense_h
            else None
        )
        # extra dense layer for multi-task
        self.dense_d = (
            blocks.stack_sab(dim=embd_dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
            if use_dense_d
            else None
        )
        # extra dense layer for multi-task for back-compatibility, use single layer
        self.dense_o = nn.Sequential(nn.Linear(embd_dim, embd_dim)) if use_dense_o else None
        # self.dense_o = dense_layer(embd_dim) if use_dense_o else None

        # x = iSAB(iSAB(x)) or SAB(SAB(x)) according to the number of points
        self.encoder = blocks.stack_sab(dim=embd_dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
        # z = SAB(PMA(seed, rFF(f))) to get outfit representation
        self.pma = blocks.AttentionPool(embd_dim, num_heads, num_seeds, ln=True)
        self.sab = blocks.SelfAttentionBlock(embd_dim, num_heads, ln=True) if num_seeds > 1 else nn.Identity()

    def _encode_item(self, x: torch.Tensor, cate: torch.Tensor = None) -> torch.Tensor:
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

    def _encode_context(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Flow: h -> SAB(h) -> f"""
        # Encode items with stacked self-attention blocks
        for encoder in self.encoder:
            h = encoder(h, mask)
        return h

    def _pool_context(self, f: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Flow: f -> SAB(PMA(f)) -> z"""
        return self.sab(self.pma(f, mask))

    def latent_h(self, item: torch.Tensor, cate: torch.Tensor | None = None) -> torch.Tensor:
        """Item-level encoding: x -> h -> [dense_h].
        Returns h or h' with shape (B, N, D).
        """
        h = self._encode_item(item, cate)
        mask = (cate != -1).float()
        f = self._encode_context(h, mask)
        if not self.use_dense_h:
            return f
        for encoder in self.dense_h:
            f = encoder(f, mask)
        return f

    def latent_f(self, item: torch.Tensor, cate: torch.Tensor = None) -> torch.Tensor:
        """Context-level encoding: x -> h -> SAB() -> f -> [dense_d].
        Returns f or f' with shape (B, N, D).
        """
        h = self._encode_item(item, cate)
        mask = (cate != -1).float()
        f = self._encode_context(h, mask)
        if not self.use_dense_d:
            return f
        for encoder in self.dense_d:
            f = encoder(f, mask)
        return f

    def latent_z(self, item: torch.Tensor, cate: torch.Tensor = None) -> torch.Tensor:
        """Outfit embeddings for downstream tasks.

        x -> h -> SAB(h) -> d -> Pool(d) -> z -> [dense_o] -> z'

        Returns z or z' with shape (B, S, D), where S = num_seeds.
        """
        mask = (cate != -1).float() if cate is not None else None
        # general embedding. b x n x d
        h = self._encode_item(item, cate)
        f = self._encode_context(h, mask)
        z = self._pool_context(f, mask)
        return self.dense_o(z) if self.use_dense_o else z


class Anchors(nn.Module):
    def __init__(
        self,
        num_users: int = 1,
        embd_dim: int = 128,
        num_anchors: int = 16,  # anchors per type
        use_g_anchors: bool = False,
        use_u_anchors: bool = True,
        use_g_lambda: bool = False,
        use_u_lambda: bool = False,
        kappa: float = 10.0,
    ):
        super().__init__()
        self.embd_dim = embd_dim
        self.num_users = num_users
        self.num_anchors = num_anchors
        self.use_g_anchors = use_g_anchors
        self.use_u_anchors = use_u_anchors
        self.use_g_lambda = use_g_lambda
        self.use_u_lambda = use_u_lambda

        self.kappa = kappa

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

        # lambdas
        self.lambda_u = nn.Parameter(torch.ones(1)) if self.use_u_lambda else None
        self.lambda_g = nn.Parameter(torch.ones(1)) if self.use_g_lambda else None

    def reg(self, use_logdet=True) -> torch.Tensor | None:
        """Compute regularization for users."""
        if self.num_anchors <= 1:  # no regularization needed when only one anchor
            return 0.0
        gram = self.u_anchors @ self.u_anchors.transpose(1, 2)
        if use_logdet:
            gram = gram + 1e-6 * torch.eye(gram.size(-1), device=gram.device)
            sign, logdet = torch.linalg.slogdet(gram)
            logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
            reg = (gram.diagonal(dim1=1, dim2=2).sum() - logdet.sum()) / self.num_users
        else:
            eye = torch.eye(self.num_anchors, device=gram.device)
            reg = ((gram * (1 - eye)) ** 2).sum() / self.num_users
        return reg

    @staticmethod
    def _nonneg(param: torch.Tensor) -> float | torch.Tensor:
        return param.abs() if param is not None else 1.0

    def item_score(
        self,
        h: torch.Tensor,
        z: torch.Tensor = None,
        uidx: torch.Tensor = None,
        use_z_term: bool = True,
        use_u_term: bool = True,
        use_g_term: bool = True,
    ) -> torch.Tensor:
        """
        Compute item-level scores, i.e., p(x|o,u).

        p(x|o,u) is proportional to exp(kappa * (z_term + u_lambda * u_term + g_lambda * g_term))

        Args:
            h (torch.Tensor): Item features, shape (B, N_i, D).
            z (torch.Tensor, optional): Outfit features, shape (B, S, D). Defaults to None.
            uidx (torch.Tensor, optional): User indices. Defaults to None.
            use_z_term (bool): Flag to use the outfit term.
            use_u_term (bool): Flag to use the user-specific anchor term.
            use_g_term (bool): Flag to use the global anchor term.

        Returns:
            torch.Tensor: Item-level scores, shape (B, N_i).
        """
        h = F.normalize(h, dim=-1)

        score = 0.0
        if use_z_term and z is not None:
            z_norm = F.normalize(z, dim=-1)  # (B, S, D)
            score += h.matmul(z_norm.transpose(1, 2)).mean(dim=-1)  # (B, N_i)

        if use_u_term and self.use_u_anchors and uidx is not None:
            u_lambda = self._nonneg(self.lambda_u)
            u = self.u_anchors.index_select(0, uidx)  # (B, A, D)
            u_norm = F.normalize(u, dim=-1)
            score += u_lambda * h.matmul(u_norm.transpose(1, 2)).mean(dim=-1)

        if use_g_term and self.use_g_anchors:
            g_lambda = self._nonneg(self.lambda_g)
            g_norm = F.normalize(self.g_anchors, dim=-1)
            score += g_lambda * h.matmul(g_norm.transpose(1, 2)).mean(dim=-1)

        return self.kappa * score

    def outfit_score(
        self,
        z: torch.Tensor,
        uidx: torch.Tensor = None,
        use_u_term: bool = True,
        use_g_term: bool = True,
    ) -> torch.Tensor:
        """
        Compute outfit-level scores, i.e., p(o|u).

        p(o|u) is proportional to exp(kappa * (u_lambda * u_term + g_lambda * g_term))

        Args:
            z (torch.Tensor): Outfit features, shape (B, S, D).
            uidx (torch.Tensor, optional): User indices. Defaults to None.
            use_u_term (bool): Flag to use the user-specific anchor term.
            use_g_term (bool): Flag to use the global anchor term.

        Returns:
            torch.Tensor: Outfit-level scores, shape (B, S).
        """
        z = F.normalize(z, dim=-1)  # (B, S, D)
        score = 0.0

        if use_u_term and self.use_u_anchors and uidx is not None:
            u_lambda = self._nonneg(self.lambda_u)
            u = self.u_anchors.index_select(0, uidx)  # (B, A, D)
            u_norm = F.normalize(u, dim=-1)
            score += u_lambda * z.matmul(u_norm.transpose(1, 2)).mean(dim=-1)

        if use_g_term and self.use_g_anchors:
            g_lambda = self._nonneg(self.lambda_g)
            g_norm = F.normalize(self.g_anchors, dim=-1)  # (1, A, D)
            score += g_lambda * z.matmul(g_norm.transpose(1, 2)).mean(dim=-1)

        return self.kappa * score

    def forward(
        self,
        uidx: torch.Tensor,
        z: torch.Tensor = None,
        h: torch.Tensor = None,
        score_type: str = "item",
        **kwargs,
    ) -> torch.Tensor:
        if score_type == "item":
            return self.item_score(h, z, uidx, **kwargs)
        if score_type == "outfit":
            return self.outfit_score(z, uidx, **kwargs)
        raise ValueError(f"Unknown score type: {score_type}")


class MultiTaskModel(nn.Module):
    """LPAE-Net."""

    def __init__(
        self,
        # core counts
        embd_dim: int = 128,
        num_users: int = 1,
        num_types: int = 1,
        # vision encoder config
        backbone: Dict = None,
        type_encoding: str = "none",  # "none", "learnable"
        use_nn_feature: bool = False,
        use_dense_h=False,
        use_dense_d=False,
        use_dense_o=False,
        # anchors
        num_anchors: int = 1,
        use_g_anchors: bool = True,
        use_u_anchors: bool = True,
        use_g_lambda: bool = False,
        use_u_lambda: bool = False,
        kappa: float = 10.0,
        # set transformer
        num_points: int = 0,
        num_sab: int = 2,
        num_seeds: int = 1,
        num_heads: int = 4,
        # loss
        logdet: bool = True,
        loss_type: str = "bpr",
        loss_weight: Dict = None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_types = num_types
        self.embd_dim = embd_dim
        self.num_seeds = num_seeds
        self.logdet = logdet

        self.loss_type = loss_type.lower()

        # update default loss weight
        self.loss_weight = ops.update_loss_weights(loss_weight, nll_c=1.0, nll_d=1.0, cl_o=1.0)

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
            use_dense_h=use_dense_h,
            use_dense_d=use_dense_d,
            use_dense_o=use_dense_o,
        )

        self.anchors = Anchors(
            num_users=num_users,
            embd_dim=embd_dim,
            num_anchors=num_anchors,
            use_g_anchors=use_g_anchors,
            use_u_anchors=use_u_anchors,
            use_g_lambda=use_g_lambda,
            use_u_lambda=use_u_lambda,
            kappa=kappa,
        )

    def group_fitb(self, uidx, data, cate, index):
        batch, num_outfits, num_items, *shape = data.shape
        # get visual features
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # reshape
        feat = feat.view(batch * num_outfits, num_items, -1)
        cate = cate.view(batch * num_outfits, num_items, -1)

        z = self.visual_encoder.latent_h(feat, cate)
        z = z.view(batch, num_outfits, num_items, -1)

        # select z[b, :, index[b], :] for each b
        idx = index[:, None, None, None].expand(-1, num_outfits, -1, z.shape[-1])
        z_selected = torch.gather(z, 2, idx).squeeze(2)  # (batch, num_answers, dim)

        scores = self.anchors.outfit_score(z_selected, uidx)
        if not self.training:
            return scores
        logprob = -F.log_softmax(scores, dim=-1)
        nll_c = logprob[:, 0].mean()
        loss = dict(nll_c=nll_c)
        accuracy = (torch.argmax(scores, dim=-1) == 0).float().mean()
        return loss, accuracy

    def outlier_detection(
        self,
        uidx: torch.Tensor,
        data: torch.Tensor,
        cate: torch.Tensor,
        yidx: torch.Tensor = None,
    ):
        batch, num_items, *shape = data.shape
        # get visual features
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # encoder
        feat = feat.view(batch, num_items, -1)
        cate = cate.view(batch, num_items, 1)
        mask = cate.view(batch, num_items) == -1
        q = self.visual_encoder.latent_f(feat, cate)
        # z = self.visual_encoder.latent_z(feat, cate)
        score = -self.anchors.item_score(q, uidx=uidx)
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
        if not self.training:
            batch, num, *shape = data.shape
            data = data.view(-1, *shape)
            cate = cate.view(batch, num, -1)
            feat = self.visual_feature(data)
            feat = feat.view(batch, num, -1)
            z = self.visual_encoder.latent_z(feat, cate)
            scores = self.anchors.outfit_score(z, uidx)
            return scores

        batch, num_outfits, num_items, *shape = data.shape
        flat = data.view(-1, *shape)
        feat = self.visual_feature(flat)
        feat = feat.view(batch * num_outfits, num_items, -1)
        cate = cate.view(batch * num_outfits, num_items, 1)
        z = self.visual_encoder.latent_z(feat, cate)
        z = z.reshape(batch, num_outfits * self.num_seeds, self.embd_dim)
        scores = self.anchors.outfit_score(z, uidx)

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
        # data_p = batch["data_p"].to(device)
        data_r = batch["data_r"].to(device)
        cate_r = batch["cate_r"].to(device)
        data_d = batch["data_d"].to(device)
        cate_d = batch["cate_d"].to(device)
        y_d = batch["y_d"].to(device)
        data_c = batch["data_g"].to(device)
        cate_c = batch["cate_g"].to(device)
        index = batch["index_g"].to(device)

        loss_r, acc_r = self.outfit_compatibility(uidx, data_r, cate_r)

        loss_d, acc_d = self.outlier_detection(uidx, data_d, cate_d, y_d)

        loss_c, acc_c = self.group_fitb(uidx, data_c, cate_c, index)

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

    @torch.no_grad()
    def eval_compatibility(self, batch, device):
        data = utils.to(batch["data"], device)
        uidx = utils.to(batch["uidx"], device)
        cate = utils.to(batch["cate"], device)
        return self.outfit_compatibility(uidx, data, cate)

    @torch.no_grad()
    def eval_completion(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        pool_data = utils.to(batch["pool_data"], device)
        pool_cate = utils.to(batch["pool_cate"], device)
        return self.group_fitb(uidx, data, cate, pool_data, pool_cate)

    @torch.no_grad()
    def eval_outlier(self, batch, device):
        data = utils.to(batch["data"], device)
        uidx = utils.to(batch["uidx"], device)
        cate = utils.to(batch["cate"], device)
        score = self.outlier_detection(uidx, data, cate)
        return score

    def eval_group_fitb(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        index = utils.to(batch["index"], device)  # b x num_questions
        scores = self.group_fitb(uidx, data, cate, index)
        return scores
