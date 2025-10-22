import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import utils


def select_features(
    select_uidx: int | None = None,
    select_label: int | None = None,
    uidxs: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns a boolean mask for selecting features:
        features[selected] -> (n_selected, dim)

    """

    # Determine device and number of samples
    if uidxs is not None:
        device, n = uidxs.device, uidxs.size(0)
    elif labels is not None:
        device, n = labels.device, labels.size(0)
    else:
        raise ValueError("At least one of `uidxs` or `labels` must be provided.")

    selected = torch.ones(n, dtype=torch.bool, device=device)

    if uidxs is not None and select_uidx is not None:
        selected &= uidxs == select_uidx
    if labels is not None and select_label is not None:
        selected &= labels == select_label

    return selected


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


class UserEmbedding(nn.Module):
    def __init__(self, num_users, dim):
        super().__init__()
        self.num_users = num_users
        self.encoder = nn.Linear(num_users, dim, bias=False)
        nn.init.normal_(self.encoder.weight, std=0.01)

    def forward(self, x):
        x = F.one_hot(x, self.num_users).float()
        h = self.encoder(x)
        return h


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
        # additonal mlp for downstream tasks
        use_dense_h=False,
        use_dense_d=False,
        use_dense_o=False,
    ):
        super().__init__()
        # TODO: use better names
        self.dim = embd_dim
        self.num_types = num_types
        self.use_dense_h = use_dense_h
        self.use_dense_d = use_dense_d
        self.use_dense_o = use_dense_o

        dim = embd_dim
        # item embedding layer for all downstream tasks
        self.dense_i = embed_layer(in_features, dim)

        # category embedding mode
        self.type_encoding = type_encoding
        if type_encoding == "learnable":
            # self.dense_c = nn.Embedding(num_types + 1, dim, padding_idx=0)
            self.dense_c = nn.Embedding(num_types, dim)

        if self.use_dense_h:
            self.dense_h = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.dense_h.apply(kaiming_normal_weight(nonlinearity="relu"))
        if self.use_dense_d:
            self.dense_d = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.dense_d.apply(kaiming_normal_weight(nonlinearity="relu"))
        if self.use_dense_o:
            self.dense_o = nn.Sequential(nn.Linear(dim, dim))
            # self.dense_o = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.dense_o.apply(kaiming_normal_weight(nonlinearity="relu"))

        # x = iSAB(iSAB(x)) or SAB(SAB(x)) according to the number of points
        self.encoder = stack_sab(dim=dim, num_heads=num_heads, num_sab=num_sab, num_points=num_points)
        self.pma = AttentionPool(dim, num_heads, num_seeds, ln=True)
        self.sab = SelfAttentionBlock(dim, num_heads, ln=True) if num_seeds > 1 else nn.Identity()

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
            c = position_encoding(cate.view(b, n, 1), self.embd_dim, self.num_types)
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

    def embedding_x(self, item: torch.Tensor, cate: torch.Tensor = None) -> torch.Tensor:
        h = self.embed_items(item, cate)
        if self.use_dense_h:
            h = self.dense_h(h)
        return h

    def embedding_f(self, item: torch.Tensor, cate: torch.Tensor = None) -> torch.Tensor:
        h = self.embed_items(item, cate)
        mask = (cate != -1).float()
        f = self.encode_set(h, mask)
        if self.use_dense_d:
            f = self.dense_d(f)
        return f

    def embedding_z(self, item: torch.Tensor, cate: torch.Tensor = None) -> torch.Tensor:
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
        if self.use_dense_o:
            z = self.dense_o(z)
        return z

    def forward(self, item: torch.Tensor, cate: torch.Tensor = None, return_h: bool = False) -> torch.Tensor:
        """
        Args:
           item (torch.Tensor): item features. shape of b x n x d.
           cate (torch.Tensor): item categories. shape of b x n x 1.

        Return:
            torch.Tensor: outfit embedding shape of b x 1 x d
        """
        mask = (cate != -1).float() if cate is not None else None
        # b x n x d
        h = self.embed_items(item, cate)
        # b x s x d
        f = self.encode_set(h, mask)
        z = self.pool_set(f, mask)
        if self.use_dense_o:
            z = self.dense_o(z)
        if self.use_dense_d:
            h = self.dense_d(h)
        if return_h:
            return z, h
        return z


class Predictor(nn.Module):
    r"""Von Mises-Fisher (vMF) embedding layer for user and global anchors.

    This module computes compatibility scores between outfit embeddings and learnable
    anchor embeddings, combining user-specific preferences with global style patterns.

    Components:
    -----------
    - U: User anchor matrix, shape (num_users, num_anchors, dim)
         Captures personalized style preferences for each user

    - G: Global anchor matrix, shape (1, num_anchors, dim)
         Captures universal style patterns across all users

    - z: Outfit embedding vector, shape (batch, 1, dim)
         Represents the overall style of an outfit

    Score Calculation:
    ------------------
    The compatibility score is computed as:

        score = kappa * (lambda_u * mean(U @ z) + lambda_g * mean(G @ z))

    where:
        - kappa: Concentration parameter (sharpness of the distribution)
        - lambda_u: Weight for user-specific preferences (learnable)
        - lambda_g: Weight for general preferences (learnable)
        - mean(): Average similarity across all anchors
        - @: Matrix multiplication (cosine similarity after normalization)

    Default Settings (from paper):
    ------------------------------
        kappa = 10.0        # concentration parameter
        lambda_u = 1.0      # user preference weight
        lambda_g = 1.0      # general preference weight
        num_u_anchors = 1
        num_g_anchors = 1
    """

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

    def ldd(self):
        """Log-determinant regularization for all user anchors."""
        u_anchors = self.u_anchors
        n = u_anchors.size(0)
        gram = u_anchors @ u_anchors.transpose(1, 2)
        gram = gram + 1e-6 * torch.eye(gram.size(-1), device=gram.device)
        sign, logdet = torch.linalg.slogdet(gram)
        logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
        reg = (gram.diagonal(dim1=1, dim2=2).sum() - logdet.sum()) / n
        return reg

    def orth(self):
        """Orthogonality regularization (used when logdet is unstable)."""
        u_anchors = self.u_anchors
        s = u_anchors.size(1)
        n = u_anchors.size(0)
        gram = u_anchors @ u_anchors.transpose(1, 2)
        eye = torch.eye(s, device=gram.device)
        reg = ((gram * (1 - eye)) ** 2).sum() / n
        return reg

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

    @torch.no_grad()
    def attn_map(self, uidx: torch.Tensor, x: torch.Tensor):
        n = x.size(0)
        z = F.normalize(x, dim=-1)
        # n x s x d
        if self.use_g_anchors and self.use_u_anchors:
            u_anchors = self.u_anchors.index_select(0, uidx)
            u_anchors = F.normalize(u_anchors, dim=-1)
            g_anchors = self.g_anchors.repeat(n, 1, 1)
            g_anchors = F.normalize(g_anchors, dim=-1)
            anchors = torch.cat((u_anchors, g_anchors), dim=1)
        elif self.use_g_anchors:
            anchors = F.normalize(self.g_anchors, dim=-1)
        elif self.use_u_anchors:
            anchors = self.u_anchors.index_select(0, uidx)
            anchors = F.normalize(anchors, dim=-1)
        else:
            raise ValueError("No anchors are used")
        attn_map = z.matmul(anchors.transpose(1, 2))
        return attn_map

    def forward(self, uidx: torch.Tensor, z: torch.Tensor, attn_map=False, return_z=False):
        """Compute the compatibility.

        Args:
            uidx (torch.Tensor): uidx. shape of n
            x (torch.Tensor): x. shape of n x s x d.
                where n is batch size, s is the number of outfits.
            attn_map: Defaults to False
        """
        n = z.size(0)
        # cosine similarity, b x n_outfits x d
        z = F.normalize(z, dim=-1)

        score_u, score_g = 0.0, 0.0
        s = self.num_seeds

        if self.use_g_anchors and self.use_u_anchors:
            # n x n_anchors x d
            u_anchors = self.u_anchors.index_select(0, uidx)
            u_anchors = F.normalize(u_anchors, dim=-1)
            g_anchors = self.g_anchors.repeat(n, 1, 1)
            g_anchors = F.normalize(g_anchors, dim=-1)
            score_u = z.matmul(u_anchors.transpose(1, 2)).mean(dim=-1)
            score_g = z.matmul(g_anchors.transpose(1, 2)).mean(dim=-1)
            if self.num_seeds > 1:
                score_u = score_u.view(n, -1, s).mean(dim=-1)
                score_g = score_g.view(n, -1, s).mean(dim=-1)
            anchors = torch.cat((u_anchors, g_anchors), dim=1)
        elif self.use_g_anchors:
            anchors = F.normalize(self.g_anchors, dim=-1)
            score_g = z.matmul(anchors.transpose(1, 2)).mean(dim=-1)
            if self.num_seeds > 1:
                score_g = score_g.view(n, -1, s).mean(dim=-1)
        elif self.use_u_anchors:
            anchors = self.u_anchors.index_select(0, uidx)
            anchors = F.normalize(anchors, dim=-1)
            score_u = z.matmul(anchors.transpose(1, 2)).mean(dim=-1)
            if self.num_seeds > 1:
                score_u = score_u.view(n, -1, s).mean(dim=-1)
        else:
            raise ValueError("No anchors are used")
        if self.use_g_lambda:
            score_g = self.lambda_g.abs() * score_g
        if self.use_u_lambda:
            score_u = self.lambda_u.abs() * score_u
        score = self.kappa_i * (score_u + score_g)
        # n x m x s
        with torch.no_grad():
            similarity = z.matmul(anchors.transpose(1, 2))
        # n x m
        if attn_map and return_z:
            return score, similarity, z
        elif attn_map:
            return score, similarity
        elif return_z:
            return score, z
        else:
            return score

    def outfit_item_score(
        self,
        uidx: torch.Tensor,
        z: torch.Tensor,
        h: torch.Tensor,
        only_u_score=False,
        return_z_score=False,
    ) -> torch.Tensor:
        """Compute the outfit item score.

        if use_u_anchors:
            user_score = lambda_u * mean(U @ z)
        if use_g_anchors:
            general_score = lambda_g * mean(G @ z)
        if use_z_score:
            z_score = z @ h
        total_score = kappa * (user_score + general_score + z_score)

        For personalized outfit recommendation:
            kappa * (user_score + general_score) is used by default.

        z_score is used for outfit completion and outlier detection.

        Args:
            uidx (torch.Tensor): one-hot user encoding
            z (torch.Tensor): outfit embedding, shape of b x 1 X d
            h (torch.Tensor): item embeddings, shape of b x m x d, where m is the number of items.

        Return:
            torch.Tensor: b x m
        """
        b, m, _ = h.shape
        # batch x 1 x d
        z = F.normalize(z, dim=-1)
        # batch x m x d
        h = F.normalize(h, dim=-1)
        u_score = 0.0
        g_score = 0.0
        z_score = 0.0
        if self.use_z_score:
            z_score = h.matmul(z.transpose(1, 2))
            # return z_score.reshape(b, m)
        if self.use_u_anchors:
            # batch x s x d
            u = self.u_anchors.index_select(0, uidx)
            # batch x s x d
            u = F.normalize(u, dim=-1)
            # batch x m x 1
            u_score = h.matmul(u.transpose(1, 2)).mean(dim=-1, keepdim=True)
        if self.use_g_anchors:
            # 1 x s x d
            g = self.g_anchors
            # 1 x s x d
            g = F.normalize(g, dim=-1)
            # batch x m x 1
            g_score = h.matmul(g.transpose(1, 2)).mean(dim=-1, keepdim=True)
        if self.use_u_lambda:
            u_score = self.lambda_u.abs() * u_score
        if self.use_g_lambda:
            g_score = self.lambda_g.abs() * g_score
        # no general bias
        if only_u_score:
            # batch x s x d
            u = self.u_anchors.index_select(0, uidx)
            # batch x s x d
            u = F.normalize(u, dim=-1)
            # batch x m x 1
            u_score = h.matmul(u.transpose(1, 2)).mean(dim=-1, keepdim=True)
            score = self.kappa_i * (z_score + u_score)
            return score.reshape(b, m)
        score = self.kappa_i * (z_score + u_score + g_score)
        if return_z_score:
            return score.reshape(b, m), self.kappa_i * z_score.reshape(b, m)
        return score.reshape(b, m)

    def model_r(self, uidx: torch.Tensor, z: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:

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

    def model_c(self, uidx: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
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

        if self.use_g_anchors:
            g = self.g_anchors  # (1, S, D)
            g = F.normalize(self.g_anchors, dim=-1)  # (1, S, D)
            g_term = h.matmul(g.transpose(1, 2)).mean(dim=-1)  # (B, M)
        else:
            g_term = 0.0

        u_lambda = self.lambda_u.abs() if self.use_u_lambda else 1.0
        g_lambda = self.lambda_g.abs() if self.use_g_lambda else 1.0

        z_term = h.matmul(z.transpose(1, 2)).mean(dim=-1)  # (B, M)
        score = self.kappa_i * (z_term + u_lambda * u_term + g_lambda * g_term)

        return score

    def model_d(self, uidx: torch.Tensor, z: torch.Tensor = None, h: torch.Tensor = None) -> torch.Tensor:
        h = F.normalize(h, dim=-1)  # (B, N, D)
        if z is not None:
            z = F.normalize(z, dim=-1)  # (B, 1, D)
            z_score = h.matmul(z.transpose(1, 2)).mean(dim=-1)  # (B, N)
        else:
            z_score = 0.0
        if self.use_u_anchors:
            u = self.u_anchors.index_select(0, uidx)  # (B, S, D)
            u = F.normalize(u, dim=-1)  # (B, S, D)
            u_score = h.matmul(u.transpose(1, 2)).mean(dim=-1)  # (B, N)
        else:
            u_score = 0.0

        if self.use_g_anchors:
            g = self.g_anchors  # (B, S, D)
            g = F.normalize(g, dim=-1)  # (B, S, D)
            g_score = h.matmul(g.transpose(1, 2)).mean(dim=-1)  # (B, N)
        else:
            g_score = 0.0

        u_lambda = self.lambda_u.abs() if self.use_u_lambda else 1.0
        g_lambda = self.lambda_g.abs() if self.use_g_lambda else 1.0
        score = self.kappa_i * (z_score + u_lambda * u_score + g_lambda * g_score)
        return score


class LPAENet(nn.Module):
    """LPAE-Net."""

    def __init__(
        self,
        num_users=1,
        num_types=1,
        backbone=None,
        embd_dim=128,
        num_anchors=16,
        use_g_anchors=False,
        use_u_anchors=True,
        use_g_lambda=False,
        use_u_lambda=False,
        use_z_score=True,
        num_points=0,
        num_sab=2,
        num_seeds=1,
        num_heads=4,
        type_encoding="none",  # "none", "learnable", "fixed"
        kappa_i=10.0,
        kappa_u=1.0,
        logdet=True,
        num_u_neg=8,
        num_o_neg=8,
        use_cl_u=False,
        use_dense_h=False,
        use_dense_d=False,
        use_dense_o=False,
        outlier_use_sab=True,  # select which features for outlier detection
        new_user_closed_form=True,
        new_user_num_outfits=10,
        freeze_z=False,
        use_nn_feature=False,
        use_semantic=False,
        use_visual=True,
        cold_start=False,
        loss_type="bpr",
        v1=False,
        loss_weight=None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_types = num_types
        self.embd_dim = embd_dim
        self.freeze_z = freeze_z
        self.new_user_closed_form = new_user_closed_form
        self.new_user_num_outfits = new_user_num_outfits
        self.use_cl_u = use_cl_u
        self.num_o_neg = num_o_neg
        self.num_seeds = num_seeds
        self.use_z_score = use_z_score
        self.use_u_anchors = use_u_anchors
        self.use_g_anchors = use_g_anchors
        self.use_u_lambda = use_u_lambda
        self.use_g_lambda = use_g_lambda
        self.outlier_use_sab = outlier_use_sab
        self.logdet = logdet
        self.kappa_i = kappa_i
        self.kappa_u = kappa_u

        # update default loss weight
        self.loss_weight = utils.update_loss_weights(loss_weight, nll_c=1.0, nll_d=1.0, cl_o=1.0)

        # type_encoding = "learnable" if learnable_type_embed else "fixed" if use_type_embed else "none"
        if use_visual:
            in_features = backbone.pop("dim")
            backbone = nn.Identity() if use_nn_feature else utils.backbone(**backbone)
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
        # TODO: add vse loss, do not use another transformer for textual feature
        if use_semantic:
            in_features = 2400
            raise NotImplementedError
        self.anchors = Predictor(
            num_users=num_users,
            embd_dim=embd_dim,
            num_seeds=num_seeds,
            num_anchors=num_anchors,
            use_g_anchors=use_g_anchors,
            use_u_anchors=use_u_anchors,
            use_g_lambda=use_g_lambda,
            use_u_lambda=use_u_lambda,
            use_z_score=use_z_score,
            kappa_i=kappa_i,
            kappa_u=kappa_u,
            logdet=logdet,
            num_u_neg=num_u_neg,
            cold_start=cold_start,
        )
        assert loss_type.lower() in ["bpr", "n-pair", "flat-nce"]
        self.loss_type = loss_type.lower()

    def embedding_x(self, data: torch.Tensor, cate: torch.Tensor) -> torch.Tensor:
        """Get item embeddings before attention."""
        batch, num_items, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # reshape
        feat = feat.view(batch, num_items, -1)
        cate = cate.view(batch, num_items, -1)
        return self.visual_encoder.embedding_x(feat, cate)

    def embedding_f(self, data: torch.Tensor, cate: torch.Tensor) -> torch.Tensor:
        """Get item embeddings after attention."""
        batch, num_items, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # reshape
        feat = feat.view(batch, num_items, -1)
        cate = cate.view(batch, num_items, -1)
        return self.visual_encoder.embedding_f(feat, cate)

    def embedding_z(self, data: torch.Tensor, cate: torch.Tensor) -> torch.Tensor:
        """Compute the outfit embedding.

        Args:
            data (torch.Tensor): item data
            cate (torch.Tensor): item cate

        Returns:
            torch.Tensor: outfit embedding, shape of b x n x d, where n is the number of outfits
        """

        if self.training:
            batch, num_outfits, num_items, *shape = data.shape
        else:
            batch, num_items, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # reshape
        if self.training:
            feat = feat.view(batch * num_outfits, num_items, -1)
            cate = cate.view(batch * num_outfits, num_items, 1)
        else:
            feat = feat.view(batch, num_items, -1)
            cate = cate.view(batch, num_items, -1)
        return self.visual_encoder.embedding_z(feat, cate)

    def outfit_completion(
        self,
        uidx: torch.Tensor,
        data: torch.Tensor,
        cate: torch.Tensor,
        pool_items: torch.Tensor,
        pool_cates: torch.Tensor,
        only_u_score: bool = False,
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
        batch, num_items, *shape = data.shape
        # get visual features
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # reshape
        feat = feat.view(batch, num_items, -1)
        cate = cate.view(batch, num_items, -1)
        # b x 1 x d
        z = self.visual_encoder.embedding_z(feat, cate)
        # b x n x d
        q = self.visual_encoder.embedding_x(pool_items, pool_cates)
        # b x n x 1
        scores = self.anchors.model_c(uidx, z, q)
        # scores = self.anchors.outfit_item_score(uidx, z, q, only_u_score=only_u_score)
        if not self.training:
            return scores
        acc = (torch.argmax(scores, dim=-1) == 0).float().mean()
        logprob = -F.log_softmax(scores, dim=-1)
        nll = logprob[:, 0].mean()
        loss = dict(nll_c=nll)
        accuracy = dict(accuracy_c=acc)
        return loss, acc

    def outlier_detection(
        self,
        uidx: torch.Tensor,
        data: torch.Tensor,
        cate: torch.Tensor,
        yidx: torch.Tensor = None,
        only_u_score: bool = False,
    ):
        batch, num_items, *shape = data.shape
        # get visual features
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # encoder
        feat = feat.view(batch, num_items, -1)
        cate = cate.view(batch, num_items, 1)
        mask = cate.view(batch, num_items) == -1
        # features
        if self.outlier_use_sab:
            # use item embedding with self-attention
            q = self.visual_encoder.embedding_f(feat, cate)
        else:
            # use item embedding without self-attention
            q = self.visual_encoder.embedding_x(feat, cate)
        z = self.visual_encoder.embedding_z(feat, cate)
        # score = -self.anchors.model_d(uidx, z, q)
        score = -self.anchors.outfit_item_score(uidx, z, q, only_u_score=only_u_score)
        value = torch.finfo(score.dtype).min
        score.masked_fill_(mask, value)
        if not self.training:
            return score
        logprob = F.log_softmax(score, dim=-1)
        nll = F.nll_loss(logprob, yidx)
        correct = torch.argmax(score, dim=-1) == yidx
        acc = correct.float().mean()
        loss = dict(nll_d=nll)
        accuracy = dict(accuracy_d=acc)
        return loss, acc

    def outfit_compatibility(self, uidx, data, cate):
        if not self.training:
            batch, num, *shape = data.shape
            data = data.view(-1, *shape)
            cate = cate.view(batch, num, -1)
            feat = self.visual_feature(data)
            feat = feat.view(batch, num, -1)
            feat = self.visual_encoder(feat, cate).reshape(batch, -1, self.embd_dim)
            return self.anchors(uidx, feat)

        batch, num_outfits, num_items, *shape = data.shape

        # get visual features
        flat = data.view(-1, *shape)
        feat = self.visual_feature(flat)
        # (batch * num_outfits) x num_items x d
        feat = feat.view(batch * num_outfits, num_items, -1)

        # (batch * num_outfits) x num_items x 1
        cate = cate.view(batch * num_outfits, num_items, 1)
        feat = self.visual_encoder(feat, cate)
        feat = feat.reshape(batch, num_outfits * self.num_seeds, self.embd_dim)
        scores = self.anchors(uidx, feat)

        # batch x num_outfits
        if self.loss_type == "n-pair":
            neg_log_prob = -F.log_softmax(scores, dim=-1)
            cl_o = neg_log_prob[:, 0].mean()
            accuracy = (torch.argmax(scores, dim=-1) == 0).float().mean()

        elif self.loss_type == "bpr":
            # BPR loss
            pos_score, neg_score = scores[:, 0], scores[:, 1]
            diff_o = (pos_score.view(-1, 1) - neg_score.view(1, -1)).view(-1)
            cl_o = F.soft_margin_loss(diff_o, torch.ones_like(diff_o), reduction="mean")
            accuracy = (diff_o > 0.0).float().mean()

        elif self.loss_type == "flat-nce":
            # FlatNCE
            pos_score, neg_score = scores[:, 0], scores[:, 1]
            # batch x batch
            diff_o = -(pos_score.view(-1, 1) - neg_score.view(1, -1))
            clogits_o = torch.logsumexp(diff_o, dim=-1)
            cl_o = torch.exp(clogits_o - clogits_o.detach()).mean()
            accuracy = (diff_o < 0.0).float().mean()
        elif self.loss_type == "info-nce":
            # InfoNCE
            pos_score, neg_score = scores[:, 0], scores[:, 1]
            pos_score = pos_score.view(-1, 1)
            neg_score = neg_score.view(-1, 1)
            scores = torch.cat((pos_score, neg_score.repeat(1, batch)), dim=-1)
            neg_log_prob = -F.log_softmax(scores, dim=-1)
            cl_o = neg_log_prob[:, 0].mean()
            accuracy = ((pos_score - neg_score) > 0.0).float().mean()
        else:
            raise KeyError(f"Unknown loss type: {self.loss_type}")

        # get loss
        loss = {"cl_o": cl_o}
        # user must has multiple style vectors
        if self.loss_weight["reg"] is not None and self.anchors.num_u_anchors > 1:
            loss["reg"] = self.anchors.reg()
        return loss, accuracy

    def forward(self, *inputs):
        if self.training:
            return self.train_compatibility(*inputs)
        return self._eval_compatibility(*inputs)

    def train_step(self, batch, device):
        # batch_size
        uidx = utils.to(batch["uidx"], device)
        # batch_size x num_outfits x num_items
        data = utils.to(batch["data"], device)
        # batch_size x num_outfits x num_items
        cate = utils.to(batch["cate"], device)
        loss, accuracy = self.train_compatibility(data, uidx, cate)
        overall_loss = 0.0
        for key, value in loss.items():
            weight = self.loss_weight.get(key, 1.0)
            overall_loss = overall_loss + weight * value.mean()
        metrics = {k: v.float().mean().item() for k, v in loss.items()}
        metrics.update({k: v.float().mean().item() for k, v in accuracy.items()})
        return overall_loss, metrics

    def train_multitask(self, batch, device):
        # uidx, data_d, cate_d, y_d, data_c, cate_c, data_p, cate_p, data_r, cate_r
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

        metrics = {}
        loss = {}
        accuracy = {}
        loss_d, accuracy_d = self.outlier_detection(uidx, data_d, cate_d, y_d, only_u_score=True)
        loss_c, accuracy_c = self.outfit_completion(uidx, data_c, cate_c, data_p, cate_p, only_u_score=True)
        loss_r, accuracy_r = self.outfit_compatibility(uidx, data_r, cate_r)
        loss.update(**loss_c)
        loss.update(**loss_r)
        loss.update(**loss_d)
        accuracy.update(**accuracy_r)
        accuracy.update(**accuracy_c)
        accuracy.update(**accuracy_d)
        overall_loss = 0.0
        for key, value in loss.items():
            weight = self.loss_weight.get(key, 1.0)
            overall_loss = overall_loss + weight * value.mean()
        metrics = {k: v.float().mean().item() for k, v in loss.items()}
        metrics.update({k: v.float().mean().item() for k, v in accuracy.items()})

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

    def train_group_fitb(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        index = utils.to(batch["index"], device)  # b
        batch, num_outfits, num_items, *shape = data.shape
        # get visual features
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # reshape
        feat = feat.view(batch * num_outfits, num_items, -1)
        cate = cate.view(batch * num_outfits, num_items, -1)
        # b x n x d
        z = self.visual_encoder.embedding_f(feat, cate)
        z = z.view(batch, num_outfits, num_items, -1)
        # select z[b, :, index[b], :] for each b
        idx = index[:, None, None, None].expand(-1, num_outfits, -1, z.shape[-1])
        z_selected = torch.gather(z, 2, idx).squeeze(2)  # (batch, num_answers, dim)
        scores = self.anchors.model_r(uidx, z_selected)
        acc = (torch.argmax(scores, dim=-1) == 0).float().mean()
        logprob = -F.log_softmax(scores, dim=-1)
        nll = logprob[:, 0].mean()
        loss = dict(nll_c=nll)
        overall_loss, metrics = utils.compute_weighted_losses(loss, self.loss_weight)
        metrics["accuracy"] = acc.item()
        return overall_loss, metrics

    def eval_group_fitb(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        index = utils.to(batch["index"], device)  # b x num_questions
        batch, num_outfits, num_items, *shape = data.shape
        # get visual features
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # reshape
        feat = feat.view(batch * num_outfits, num_items, -1)
        cate = cate.view(batch * num_outfits, num_items, -1)
        # b x n x d
        z = self.visual_encoder.embedding_f(feat, cate)
        z = z.view(batch, num_outfits, num_items, -1)
        # select z[b, :, index[b], :] for each b
        idx = index[:, None, None, None].expand(-1, num_outfits, -1, z.shape[-1])
        z_selected = torch.gather(z, 2, idx).squeeze(2)  # (batch, num_answers, dim)
        scores = self.anchors.model_r(uidx, z_selected)
        return scores

    @torch.no_grad()
    def build_features(self, dataloader, device):
        dataloader.build()
        # get test features
        cates = []
        uidxs = []
        embeddings = []
        labels = []
        self.eval()
        dataloader.next()
        for batch in tqdm(dataloader, desc="Building features", ncols=88):
            data = utils.to(batch["data"], device)
            uidx = utils.to(batch["uidx"], device)
            # reshape
            cate = utils.to(batch["cate"], device)
            n = data.size(0)
            # for outfit completion task
            batch, num_items, *shape = data.shape
            # get visual features
            data = data.view(-1, *shape)
            feat = self.visual_feature(data)
            feat = feat.view(batch, num_items, -1)
            cate = cate.view(batch, num_items, -1)
            x = self.visual_encoder.embedding_f(feat, cate)
            for i in range(n):
                embeddings.append(x[i])
                uidxs.append(uidx[i])
                cates.append(cate[i])

        self._uidxs = torch.stack(uidxs)
        self._embeddings = F.normalize(torch.stack(embeddings), dim=-1)
        self._cates = torch.stack(cates)
        self._num_new_users = self._uidxs.max().item() + 1

    @torch.no_grad()
    def build_new_user_closed_form(self, dataloader, device, n_outfits=10):
        self.build_features(dataloader, device)
        dim = self.embd_dim
        anchors = []
        for uidx in range(self._num_new_users):
            selected = select_features(select_uidx=uidx, uidxs=self._uidxs)
            # support outfit from training dataset
            supp_feats = self._embeddings[selected]  # (num_outfits, num_items, dim)
            supp_cates = self._cates[selected]  # (num_outfits, num_items, 1)
            supp_masks = supp_cates.ne(-1)  # (num_outfits, num_items, 1)
            index = torch.randperm(len(supp_feats), device=supp_feats.device)[:n_outfits]
            # num_items x dim
            supp_feats = supp_feats[index].view(n_outfits, -1, dim)
            supp_masks = supp_masks[index].view(n_outfits, -1, 1)
            # num_outfits x 1 x dim
            supp_feats = (supp_feats * supp_masks).sum(dim=1, keepdim=True) / supp_masks.sum(dim=1, keepdim=True)
            # supp_feats = F.normalize(supp_feats, dim=-1)
            supp_feats = supp_feats.mean(dim=0)
            u = F.normalize(supp_feats, dim=-1)
            anchors.append(u)
        anchors = torch.cat(anchors, dim=0)
        anchors = anchors.view(-1, 1, dim)
        self.anchors.u_anchors.data = anchors.clone()

    def new_user_adhoc(self):
        pass

    def new_user_finetune(self):
        # disable all parameters except user anchors
        for name, param in self.named_parameters():
            if "u_anchors" not in name:
                param.requires_grad = False


class LatentMatch(nn.Module):
    def __init__(self, embd_dim, com_score=False):
        super().__init__()
        self.d = d = embd_dim
        self.com_score = com_score
        self.dense_g = nn.Parameter(torch.zeros(d, d))
        self.out_g = nn.Parameter(torch.zeros(d, 1))
        self.bias_g = nn.Parameter(torch.zeros(1, d))
        if com_score:
            self.dense_c = nn.Parameter(torch.zeros(d, d))
            self.out_c = nn.Parameter(torch.zeros(d, 1))
            self.bias_c = nn.Parameter(torch.zeros(1, d))
            self.scale = nn.Parameter(torch.ones(1, 1))
        self.init()

    def init(self):
        nn.init.normal_(self.out_g, std=math.sqrt(2 / self.d))
        nn.init.normal_(self.dense_g, std=math.sqrt(2 / self.d))
        if self.com_score:
            nn.init.normal_(self.out_c, std=math.sqrt(2 / self.d))
            nn.init.normal_(self.dense_c, std=math.sqrt(2 / self.d))

    def forward(self, u, z):
        u = F.normalize(u, dim=-1)
        z = F.normalize(z, dim=-1)
        if self.com_score:
            score_u = F.relu((u * z).matmul(self.dense_g) + self.bias_g).matmul(self.out_g)
            score_i = F.relu(z.matmul(self.dense_c) + self.bias_c).matmul(self.out_c)
            score = score_u * self.scale + score_i
        else:
            score = F.relu((u * z).matmul(self.dense_g) + self.bias_g).matmul(self.out_g)
        return score
