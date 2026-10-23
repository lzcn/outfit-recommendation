import time
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import utils
from models import blocks, ops

logger = utils.get_logger(__name__)


class CompatibilityModel(nn.Module):
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
        self.visual_encoder = blocks.SetTransformer(
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

        self.anchors = blocks.Anchors(
            num_users=num_users,
            embd_dim=embd_dim,
            num_anchors=num_anchors,
            use_g_anchors=use_g_anchors,
            use_u_anchors=use_u_anchors,
            use_g_lambda=use_g_lambda,
            use_u_lambda=use_u_lambda,
            kappa=kappa,
        )

    def forward(self, uidx, data, cate):
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
        if self.loss_weight.get("reg") is not None and self.anchors.num_u_anchors > 1:
            loss["reg"] = self.anchors.reg(use_logdet=self.logdet)
        return loss, accuracy

    def train_compatibility(self, batch, device):
        # batch_size
        uidx = ops.to(batch["uidx"], device)
        # batch_size x num_outfits x num_items
        data = ops.to(batch["data"], device)
        # batch_size x num_outfits x num_items
        cate = ops.to(batch["cate"], device)
        loss, accuracy = self(uidx, data, cate)
        overall_loss, metrics = ops.compute_weighted_losses(loss, self.loss_weight)
        metrics["accuracy"] = accuracy.item()
        return overall_loss, metrics

    @torch.no_grad()
    def eval_compatibility(self, batch, device):
        data = ops.to(batch["data"], device)
        uidx = ops.to(batch["uidx"], device)
        cate = ops.to(batch["cate"], device)
        return self(uidx, data, cate)

    @torch.no_grad()
    def build_new_user_features(self, dataloader, device):
        # building features
        self.eval()
        dataloader.build()
        features, uidxs, labels = [], [], []
        for batch in tqdm(dataloader, desc="Building features", ncols=88):
            data = ops.to(batch["data"], device)
            uidx = ops.to(batch["uidx"], device)
            cate = ops.to(batch["cate"], device)
            label = ops.to(batch["label"], device)
            n = data.size(0)
            # for outfit completion task
            batch, num, *shape = data.shape
            # get visual features
            data = data.view(-1, *shape)
            feat = self.visual_feature(data)
            feat = feat.view(batch, num, -1)
            cate = cate.view(batch, num, -1)
            z = self.visual_encoder.latent_z(feat, cate)  # b x 1 x d
            for i in range(n):
                features.append(z[i])
                uidxs.append(uidx[i])
                labels.append(label[i])
        features = torch.stack(features, dim=0)  # total_outfits x 1 x d
        uidxs = torch.stack(uidxs, dim=0)  # total_outfits
        labels = torch.stack(labels, dim=0)  # total_outfits
        logger.info(f"Features built: {features.shape}, UIDs: {uidxs.shape}, Labels: {labels.shape}")
        return uidxs, features, labels

    @torch.no_grad()
    def build_new_user_closed_form(self, dataloader, device, n_outfits=1):
        uidxs, features, labels = self.build_new_user_features(dataloader, device)
        # solve for new user anchors
        num_new_users = uidxs.max().item() + 1
        new_anchors = []
        start_time = time.time()

        for uidx in range(num_new_users):
            selected = ops.select_features(select_uidx=uidx, select_label=1, uidxs=uidxs, labels=labels)
            supp_feats = features[selected]
            index = torch.randperm(len(supp_feats), device=supp_feats.device)[:n_outfits]
            supp_feats = supp_feats[index]  # n_outfits x 1 x d
            supp_feats = F.normalize(supp_feats, dim=-1)
            u_proto = supp_feats.mean(dim=0)  # 1 x d
            u_proto = F.normalize(u_proto, dim=-1)
            new_anchors.append(u_proto)
        new_anchors = torch.stack(new_anchors, dim=0)  # num_new_users x 1 x d
        # reset user anchors
        logger.info(f"New user anchors built: {new_anchors.shape} from {n_outfits} outfits each.")
        self.anchors.u_anchors = nn.Parameter(new_anchors)
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / self.num_users
        logger.info(f"Closed-form new user modeling completed in {elapsed_time:.2f}s (avg {avg_time:.4f}s/user).")

    @torch.no_grad()
    def build_new_user_omega_only(self, dataloader, device, n_outfits):
        start_time = time.time()

        omega = self.anchors.g_anchors.detach().to(device)
        if omega.dim() == 3:  # (1,1,d)
            omega = omega.squeeze(0)  # -> (1,d)
        omega = omega / (omega.norm(dim=-1, keepdim=True) + 1e-12)  # L2

        new_anchors = omega.unsqueeze(0).repeat(self.num_users, 1, 1)  # (U,1,d)
        self.anchors.u_anchors = nn.Parameter(new_anchors)

        elapsed = time.time() - start_time
        logger.info(f"[omega_only] New user anchors built: {new_anchors.shape}. Time {elapsed:.2f}s.")

    @torch.no_grad()
    def build_new_user_shrink(self, dataloader, device, n_outfits=1, alpha=5.0, deterministic=False):
        uidxs, features, labels = self.build_new_user_features(dataloader, device)
        num_new_users = uidxs.max().item() + 1
        start_time = time.time()

        omega = self.anchors.g_anchors.detach().to(device)
        if omega.dim() == 3:
            omega = omega.squeeze(0)  # (1,d)
        omega = omega / (omega.norm(dim=-1, keepdim=True) + 1e-12)

        new_anchors = []
        for u in range(num_new_users):
            sel = ops.select_features(select_uidx=u, select_label=1, uidxs=uidxs, labels=labels)
            feats = features[sel]  # (T,1,d)
            if feats is not None and feats.numel() > 0:
                if not deterministic and feats.shape[0] > n_outfits:
                    idx = torch.randperm(feats.shape[0], device=feats.device)[:n_outfits]
                    feats = feats[idx]
                feats = F.normalize(feats, dim=-1)  # (k,1,d)
                k = feats.shape[0]
                mpos = feats.mean(dim=0)  # (1,d)
                mu_hat = (k * mpos + alpha * omega) / (k + alpha)
                mu = mu_hat / (mu_hat.norm(dim=-1, keepdim=True) + 1e-12)
            else:
                mu = omega.clone()
            new_anchors.append(mu)

        new_anchors = torch.stack(new_anchors, dim=0)  # (U,1,d)
        self.anchors.u_anchors = nn.Parameter(new_anchors)

        elapsed = time.time() - start_time
        logger.info(
            f"[shrink] New user anchors built: {new_anchors.shape} (alpha={alpha}, n_outfits={n_outfits}). "
            f"Time {elapsed:.2f}s."
        )

    @torch.no_grad()
    def build_new_user_rocchio(self, dataloader, device, n_outfits=1, alpha_p=1.0, alpha_n=0.5, deterministic=False):
        uidxs, features, labels = self.build_new_user_features(dataloader, device)
        num_new_users = uidxs.max().item() + 1
        start_time = time.time()

        omega = self.anchors.g_anchors.detach().to(device)
        if omega.dim() == 3:
            omega = omega.squeeze(0)  # (1,d)
        omega = omega / (omega.norm(dim=-1, keepdim=True) + 1e-12)

        new_anchors = []
        for u in range(num_new_users):
            pos_sel = ops.select_features(select_uidx=u, select_label=1, uidxs=uidxs, labels=labels)
            neg_sel = ops.select_features(select_uidx=u, select_label=0, uidxs=uidxs, labels=labels)

            pos = features[pos_sel] if pos_sel is not None else None
            neg = features[neg_sel] if neg_sel is not None else None

            if pos is not None and pos.numel() > 0 and (not deterministic) and pos.shape[0] > n_outfits:
                idx = torch.randperm(pos.shape[0], device=pos.device)[:n_outfits]
                pos = pos[idx]
            if neg is not None and neg.numel() > 0 and (not deterministic) and neg.shape[0] > n_outfits:
                idx = torch.randperm(neg.shape[0], device=neg.device)[:n_outfits]
                neg = neg[idx]

            have_pos = pos is not None and pos.numel() > 0
            have_neg = neg is not None and neg.numel() > 0

            if not have_pos and not have_neg:
                mu = omega.clone()
            else:
                mpos = F.normalize(pos, dim=-1).mean(dim=0) if have_pos else torch.zeros_like(omega)  # (1,d)
                mneg = F.normalize(neg, dim=-1).mean(dim=0) if have_neg else torch.zeros_like(omega)  # (1,d)
                mu_raw = alpha_p * mpos - alpha_n * mneg
                norm = mu_raw.norm(dim=-1, keepdim=True)
                if torch.all(norm <= 1e-12):
                    mu = omega.clone()
                else:
                    mu = mu_raw / (norm + 1e-12)

            new_anchors.append(mu)

        new_anchors = torch.stack(new_anchors, dim=0)  # (U,1,d)
        self.anchors.u_anchors = nn.Parameter(new_anchors)

        elapsed = time.time() - start_time
        logger.info(
            f"[rocchio] New user anchors built: {new_anchors.shape} (alpha_p={alpha_p}, alpha_n={alpha_n}, "
            f"n_outfits={n_outfits}). Time {elapsed:.2f}s."
        )

    @torch.no_grad()
    def build_new_user_ridge(
        self,
        dataloader,
        device,
        n_outfits=1,
        ridge_lambda=5.0,
        y_pos=1.0,
        y_neg=-1.0,
        lambda_global=1.0,
        deterministic=False,
    ):
        uidxs, features, labels = self.build_new_user_features(dataloader, device)
        num_new_users = uidxs.max().item() + 1
        start_time = time.time()

        omega = self.anchors.g_anchors.detach().to(device)
        if omega.dim() == 3:
            omega = omega.squeeze(0)  # (1,d)
        omega = omega / (omega.norm(dim=-1, keepdim=True) + 1e-12)
        omega_vec = omega.squeeze(0)  # (d,)

        new_anchors = []
        for u in range(num_new_users):
            pos_sel = ops.select_features(select_uidx=u, select_label=1, uidxs=uidxs, labels=labels)
            neg_sel = ops.select_features(select_uidx=u, select_label=0, uidxs=uidxs, labels=labels)

            pos = features[pos_sel] if pos_sel is not None else None
            neg = features[neg_sel] if neg_sel is not None else None

            if pos is not None and pos.numel() > 0 and (not deterministic) and pos.shape[0] > n_outfits:
                idx = torch.randperm(pos.shape[0], device=pos.device)[:n_outfits]
                pos = pos[idx]
            if neg is not None and neg.numel() > 0 and (not deterministic) and neg.shape[0] > n_outfits:
                idx = torch.randperm(neg.shape[0], device=neg.device)[:n_outfits]
                neg = neg[idx]

            Z_rows, y_rows = [], []
            if pos is not None and pos.numel() > 0:
                pos = F.normalize(pos, dim=-1).squeeze(1)  # (k_pos,d)
                Z_rows.append(pos)
                y_rows.append(torch.full((pos.shape[0],), y_pos, device=device))
            if neg is not None and neg.numel() > 0:
                neg = F.normalize(neg, dim=-1).squeeze(1)  # (k_neg,d)
                Z_rows.append(neg)
                y_rows.append(torch.full((neg.shape[0],), y_neg, device=device))

            if len(Z_rows) == 0:
                mu = omega.clone()  # fallback
            else:
                Z = torch.cat(Z_rows, dim=0)  # (k,d)
                y = torch.cat(y_rows, dim=0)  # (k,)
                r = y - (Z @ (lambda_global * omega_vec))  # residuals (k,)
                d = Z.shape[1]
                A = Z.T @ Z + ridge_lambda * torch.eye(d, device=device)  # (d,d)
                b = Z.T @ r  # (d,)
                mu_vec = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)  # (d,)
                mu = mu_vec.unsqueeze(0)  # (1,d)
                mu = mu / (mu.norm(dim=-1, keepdim=True) + 1e-12)

            new_anchors.append(mu)

        new_anchors = torch.stack(new_anchors, dim=0)  # (U,1,d)
        self.anchors.u_anchors = nn.Parameter(new_anchors)

        elapsed = time.time() - start_time
        logger.info(
            f"[ridge] New user anchors built: {new_anchors.shape} "
            f"(lambda_global={lambda_global}, ridge_lambda={ridge_lambda}, "
            f"n_outfits={n_outfits}). Time {elapsed:.2f}s."
        )

    @torch.no_grad()
    def build_new_user_adhoc(self, tdataloader, device, n_outfits=1):
        uidxs, features, labels = self.build_new_user_features(tdataloader, device)
        num_new_users = uidxs.max().item() + 1
        start_time = time.time()
        num_old_users = self.anchors.num_users
        num_u_anchors = self.anchors.num_u_anchors
        anchor_pool = self.anchors.u_anchors.data  # num_users x num_anchors x d
        anchor_pool = F.normalize(anchor_pool, dim=-1).view(-1, self.embd_dim)
        new_anchors = []
        for uidx in range(num_new_users):
            selected = ops.select_features(select_uidx=uidx, uidxs=uidxs)
            supp_feats = features[selected]
            index = torch.randperm(len(supp_feats), device=supp_feats.device)[:n_outfits]
            supp_feats = supp_feats[index].squeeze(1)  # n_outfits x d
            supp_feats = F.normalize(supp_feats, dim=-1)
            sims = anchor_pool @ supp_feats.t()  # (num_users * num_anchors) x n_outfits
            sims = sims.mean(dim=-1)  # num_users * num_anchors
            if n_outfits == 1:
                # # use all positive anchors for stability
                # topk = sims >= 0.0
                # selected_anchors = anchor_pool[topk]  # num_selected x d
                # if len(selected_anchors) == 0:
                #     # fall back to selecting the best anchor
                #     _, topk = torch.topk(sims, k=1, largest=True)
                #     selected_anchors = anchor_pool[topk]
                # selected_anchors = F.normalize(selected_anchors, dim=-1)
                # selected_anchors = selected_anchors.sum(dim=0, keepdim=True)  # 1 x d

                # simply select top 16 anchors
                _, topk = torch.topk(sims, k=16, largest=True)
                selected_anchors = anchor_pool[topk]
                selected_anchors = F.normalize(selected_anchors, dim=-1)
            else:  # select num_u_anchors anchors
                _, topk = torch.topk(sims, k=num_u_anchors, largest=True)
                selected_anchors = anchor_pool[topk]  # num_u_anchors x d
            new_anchors.append(F.normalize(selected_anchors, dim=-1))
        new_anchors = torch.stack(new_anchors, dim=0)  # num_new_users x num_u_anchors x d
        elapsed = time.time() - start_time
        logger.info(
            f"[adhoc] New user anchors built: {new_anchors.shape} from {n_outfits} outfits each."
            f" Time {elapsed:.2f}s. Each user avg {(elapsed / num_new_users):.4f}s."
        )

        # reset user anchors
        self.anchors.u_anchors = nn.Parameter(new_anchors)

    def setup_new_user_finetune(self):
        # disable all parameters except user anchors
        for name, param in self.named_parameters():
            if "u_anchors" not in name:
                param.requires_grad = False
