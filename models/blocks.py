import math
from functools import partial
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

import utils
from models import ops


def kaiming_normal_weight(a=0, mode="fan_in", nonlinearity="relu"):
    """apply kaiming normal weight initialization."""

    def _init_func(module, a, mode, nonlinearity):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    return partial(_init_func, a=a, mode=mode, nonlinearity=nonlinearity)


def dense_layer(dim):
    layer = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    layer.apply(kaiming_normal_weight(nonlinearity="relu"))
    return layer


def embed_layer(in_features, out_features, dropout=0.2) -> nn.Module:
    """General embedding layer.

    It consists of one non-liear layer with LayerNorm and Dropout.

    """
    x = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )
    x.apply(kaiming_normal_weight(nonlinearity="relu"))
    return x


def position_encoding(cate, d_model, max_length=3):
    # pos: b x n x 1
    # index: 1 x 1 x d
    index = torch.arange(d_model, dtype=torch.int64, device=cate.device).reshape(1, 1, -1)
    k = 2.0 * torch.div(index, 2, rounding_mode="floor")
    # exp(log(1 / max_length ** (i / d_model)))
    div = torch.exp(-k * math.log(max_length) / d_model)
    # b x n x d
    phase = cate * div
    return torch.where(index % 2 == 0, torch.sin(phase), torch.cos(phase))


class CategoryEncoding(nn.Module):
    def __init__(self, num_types, input_dim, embd_dim, dropout=0.2):
        super().__init__()
        self.num_types = num_types
        self.weights = nn.Parameter(torch.zeros(num_types + 1, input_dim, embd_dim))
        self.bias = nn.Parameter(torch.zeros(num_types + 1, embd_dim))
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embd_dim)

        bound = 1 / math.sqrt(input_dim + embd_dim)
        nn.init.uniform_(self.weights, -bound, bound)

    def forward(self, x, cate):
        # b x n x d_in
        b, n, _ = x.shape
        cate = cate.reshape(b, n) + 1
        # b x n x d_in x d_out
        weights = self.weights[cate]
        # b x n x d_out
        bias = self.bias[cate]
        y = torch.matmul(x.unsqueeze(2), weights).squeeze(2) + bias
        y = F.relu(self.ln(y))
        return self.dropout(y)


class MAB(nn.Module):
    """The implementation of Multihead attetion from Set Transformer."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        # TODO: use weight initialization to improve the performance

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, mask_q: torch.Tensor = None, mask_k: torch.Tensor = None
    ) -> torch.Tensor:
        """Multihead attention.

        Args:
            Q (torch.Tensor): query. shape of b x n x d
            K (torch.Tensor): key. shape of b x m x d
            mask_q (torch.Tensor, optional): shape of b x n x 1. Defaults to None.
            mask_k (torch.Tensor, optional): shape of b x m x 1. Defaults to None.

        Returns:
            torch.Tensor: attentived outputs
        """
        # Get query, key and value
        b, n, _ = Q.shape
        b, m, _ = K.shape
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        if mask_q is None:
            mask_q = Q.new_ones(b, n, 1)
        if mask_k is None:
            mask_k = K.new_ones(b, m, 1)
        # (h b) x n x 1
        mask_q = mask_q.repeat(self.num_heads, 1, 1)
        # (h b) x m x 1
        mask_k = mask_k.repeat(self.num_heads, 1, 1)
        # (h b) x n x d
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        # (h b) x m x d
        K_ = torch.cat(K.split(dim_split, 2), 0)
        # (h b) x m x d
        V_ = torch.cat(V.split(dim_split, 2), 0)
        # (n b) x n x m
        dots = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
        # (n b) x n x m
        mask = torch.bmm(mask_q, mask_k.transpose(1, 2)) == 0.0
        # set the coressponding value to -inf
        value = torch.finfo(dots.dtype).min
        dots.masked_fill_(mask, value)
        A = torch.softmax(dots, dim=2)
        # H = LayerNorm(Q + Multihead(Q, K, V))
        H = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        H = H if getattr(self, "ln0", None) is None else self.ln0(H)
        H = H + F.relu(self.fc_o(H))
        H = H if getattr(self, "ln1", None) is None else self.ln1(H)
        return H


class MultiheadAttention(nn.Module):
    """Multihead attention block."""

    def __init__(self, dim_q, dim_k=None, dim_v=None, num_heads=4):
        super(MultiheadAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k if dim_k is not None else dim_q
        self.dim_v = dim_v if dim_v is not None else dim_q
        self.num_heads = num_heads
        self.fc_q = nn.Linear(self.dim_q, self.dim_v)
        self.fc_k = nn.Linear(self.dim_k, self.dim_v)
        self.fc_v = nn.Linear(self.dim_k, self.dim_v)
        self.fc_o = nn.Linear(self.dim_v, self.dim_v)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, mask_q: torch.Tensor = None, mask_k: torch.Tensor = None
    ) -> torch.Tensor:
        """Multihead attention.

        Args:
            Q (torch.Tensor): query. shape of b x n x d_q
            K (torch.Tensor): key. shape of b x m x d_k
            mask_q (torch.Tensor, optional): shape of b x n x 1. Defaults to None.
            mask_k (torch.Tensor, optional): shape of b x m x 1. Defaults to None.

        Returns:
            torch.Tensor: attentived outputs. shape of b x n x d_v
        """
        # Get query, key and value
        b, n, _ = Q.shape
        b, m, _ = K.shape
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_v // self.num_heads
        if mask_q is None:
            mask_q = Q.new_ones(b, n, 1)
        if mask_k is None:
            mask_k = K.new_ones(b, m, 1)
        # (h b) x n x 1
        mask_q = mask_q.repeat(self.num_heads, 1, 1)
        # (h b) x m x 1
        mask_k = mask_k.repeat(self.num_heads, 1, 1)
        # (h b) x n x d_v
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        # (h b) x m x d_v
        K_ = torch.cat(K.split(dim_split, 2), 0)
        # (h b) x m x d_v
        V_ = torch.cat(V.split(dim_split, 2), 0)
        # (n b) x n x m
        dots = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
        # (n b) x n x m
        mask = torch.bmm(mask_q, mask_k.transpose(1, 2)) == 0.0
        # set the coressponding value to -inf
        value = torch.finfo(dots.dtype).min
        dots.masked_fill_(mask, value)
        A = torch.softmax(dots, dim=2)
        # b x n x d_v
        O_ = self.fc_o(torch.cat(A.bmm(V_).split(Q.size(0), 0), 2))
        return O_


class SelfAttentionBlock(nn.Module):
    """Self attention block.

    Args:
        dim_in (int): input dimension
        dim_out (int): output dimension
        num_heads (int): number of heads
    """

    def __init__(self, dim, num_heads, ln=True):
        super(SelfAttentionBlock, self).__init__()
        self.mab = MultiheadAttention(dim, num_heads=num_heads)
        if ln:
            self.ln0 = nn.LayerNorm(dim)
            self.ln1 = nn.LayerNorm(dim)
        else:
            self.ln0 = None
            self.ln1 = None
        self.rff = nn.Linear(dim, dim)

    def forward_qk(self, Q, K, mask_q=None, mask_k=None):
        H = Q + self.mab(Q, K, mask_q, mask_k)
        H = H if self.ln0 is None else self.ln0(H)
        H = H + F.relu(self.rff(H))
        H = H if self.ln1 is None else self.ln1(H)
        return H

    def forward(self, X, mask=None):
        # H = LayerNorm(X + MultiheadAttention(X, X, X))
        H = X + self.mab(X, X, mask, mask)
        H = H if self.ln0 is None else self.ln0(H)
        # SelfAttn(X) = LayerNorm(H + rff(H))
        H = H + F.relu(self.rff(H))
        H = H if self.ln1 is None else self.ln1(H)
        return H


class AttentionPool(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(AttentionPool, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        self.register_buffer("mask", torch.ones(1, num_seeds, 1))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiheadAttention(dim, num_heads=num_heads)
        if ln:
            self.ln0 = nn.LayerNorm(dim)
            self.ln1 = nn.LayerNorm(dim)
        else:
            self.ln0 = None
            self.ln1 = None
        self.rff = nn.Linear(dim, dim)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b = X.size(0)
        h = self.mab(self.S.repeat(b, 1, 1), X, self.mask.repeat(b, 1, 1), mask)
        h = h if self.ln0 is None else self.ln0(h)
        # SelfAttn(X) = LayerNorm(H + rff(H))
        h = h + F.relu(self.rff(h))
        h = h if self.ln1 is None else self.ln1(h)
        return h


class SAB(nn.Module):
    """Self attention block.

    Args:
        dim_in (int): input dimension
        dim_out (int): output dimension
        num_heads (int): number of heads
    """

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask, mask)


class InducedSelfAttentionBlock(nn.Module):
    """Induced self attention block."""

    def __init__(self, dim, num_heads, num_points, ln=False):
        super(InducedSelfAttentionBlock, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_points, dim))  # noqa: E741
        nn.init.xavier_uniform_(self.I)
        self.mab0 = SelfAttentionBlock(dim, num_heads, ln=ln)
        self.mab1 = SelfAttentionBlock(dim, num_heads, ln=ln)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        H = self.mab0.forward_qk(self.I.repeat(X.size(0), 1, 1), X, None, mask)
        return self.mab1.forward_qk(X, H, mask, None)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        self.register_buffer("mask", torch.ones(1, num_seeds, 1))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b = X.size(0)
        return self.mab(self.S.repeat(b, 1, 1), X, self.mask.repeat(b, 1, 1), mask)


def stack_sab(dim, num_heads, num_sab, num_points=0, v1=False):
    """Stack of (induced) self attention layers.

    Assume that all dimension are the same.

    If `num_points > 0`, then the induced self-attention-block is used.
    """

    def _isab():
        return InducedSelfAttentionBlock(dim, num_heads=num_heads, num_points=num_points, ln=True)

    def _sab():
        return SelfAttentionBlock(dim, num_heads=num_heads, ln=True)

    if num_points and num_points > 0:
        sab = _isab
    else:
        sab = _sab
    sabs = [sab() for _ in range(num_sab)]
    return nn.ModuleList(sabs)


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
        self.dense_i = embed_layer(in_features, embd_dim)

        # category embedding mode
        self.type_encoding = type_encoding
        if type_encoding == "learnable" and num_types is not None:
            self.dense_c = nn.Embedding(num_types, embd_dim)

        # extra dense layer for multi-task
        self.dense_h = dense_layer(embd_dim) if use_dense_h else None
        # extra dense layer for multi-task
        self.dense_d = dense_layer(embd_dim) if use_dense_d else None
        # extra dense layer for multi-task for back-compatibility, use single layer
        self.dense_o = nn.Sequential(nn.Linear(embd_dim, embd_dim)) if use_dense_o else None
        # self.dense_o = dense_layer(embd_dim) if use_dense_o else None

        # x = iSAB(iSAB(x)) or SAB(SAB(x)) according to the number of points
        self.encoder = stack_sab(dim=embd_dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
        # z = SAB(PMA(seed, rFF(f))) to get outfit representation
        self.pma = AttentionPool(embd_dim, num_heads, num_seeds, ln=True)
        self.sab = SelfAttentionBlock(embd_dim, num_heads, ln=True) if num_seeds > 1 else nn.Identity()

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
            c = position_encoding(cate.view(b, n, 1), self.embed_dim, self.num_types)
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

    def latent_x(self, item: torch.Tensor, cate: torch.Tensor | None = None) -> torch.Tensor:
        """Item-level encoding: x -> h -> [dense_h].
        Returns h or h' with shape (B, N, D).
        """
        h = self._encode_item(item, cate)
        return self.dense_h(h) if self.use_dense_h else h

    def latent_f(self, item: torch.Tensor, cate: torch.Tensor = None) -> torch.Tensor:
        """Context-level encoding: x -> h -> SAB() -> f -> [dense_d].
        Returns f or f' with shape (B, N, D).
        """
        h = self._encode_item(item, cate)
        mask = (cate != -1).float()
        f = self._encode_context(h, mask)
        return self.dense_d(f) if self.use_dense_d else f

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


class BaseModel(nn.Module):
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
