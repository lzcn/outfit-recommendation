from typing import Dict

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from tqdm import tqdm

import utils
from models import blocks, ops

logger = utils.get_logger(__name__)


class OutlierModel(nn.Module):
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
        z = self.visual_encoder.latent_z(feat, cate)
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

    @torch.no_grad()
    def build_new_user_features(self, dataloader, device):
        # building features
        self.eval()
        dataloader.build()
        features, uidxs, cates = [], [], []
        for batch in tqdm(dataloader, desc="Building features", ncols=88):
            data = ops.to(batch["data"], device)
            uidx = ops.to(batch["uidx"], device)
            cate = ops.to(batch["cate"], device)
            n = data.size(0)
            # for outfit completion task
            batch, num, *shape = data.shape
            # get visual features
            data = data.view(-1, *shape)
            feat = self.visual_feature(data)
            feat = feat.view(batch, num, -1)
            cate = cate.view(batch, num, -1)
            z = self.visual_encoder.latent_f(feat, cate)  # b x n_items x d
            for i in range(n):
                features.append(z[i])
                uidxs.append(uidx[i])
                cates.append(cate[i])
        features = torch.stack(features, dim=0)  # total_outfits x n_items x d
        features = F.normalize(features, dim=-1)
        uidxs = torch.stack(uidxs, dim=0)  # total_outfits
        cates = torch.stack(cates, dim=0)  # total_outfits x n_items x 1
        logger.info(f"Features built: {features.shape}, Uidxs built: {uidxs.shape}, Cates built: {cates.shape}")
        return uidxs, features, cates

    @torch.no_grad()
    def build_new_user_closed_form(self, dataloader, device, n_outfits=1):
        uidxs, features, cates = self.build_new_user_features(dataloader, device)
        # solve for new user anchors
        num_new_users = uidxs.max().item() + 1
        new_anchors = []
        for uidx in range(num_new_users):
            selected = ops.select_features(select_uidx=uidx, uidxs=uidxs)
            supp_masks = cates[selected] != -1  # (n_selected, n_items, 1)
            supp_feats = features[selected]  # (n_selected, n_items, d)
            # randomly select n_outfits
            index = torch.randperm(len(supp_feats), device=supp_feats.device)[:n_outfits]
            supp_feats = supp_feats[index]  # n_outfits x n_items x d
            supp_masks = supp_masks[index]  # n_outfits x n_items x 1
            supp_feats = rearrange(supp_feats, "b n d -> (b n) d")  # (n_outfits * n_items) x d
            supp_masks = rearrange(supp_masks, "b n 1 -> (b n) 1")  # (n_outfits * n_items) x 1
            supp_feats = (supp_feats * supp_masks).sum(dim=0, keepdim=True) / supp_masks.sum(dim=0, keepdim=True)
            u_proto = F.normalize(supp_feats, dim=-1)
            new_anchors.append(u_proto)
        new_anchors = torch.stack(new_anchors, dim=0)  # num_new_users x 1 x d
        # reset user anchors
        logger.info(f"New user anchors built: {new_anchors.shape} from {n_outfits} outfits each.")
        self.anchors.u_anchors = nn.Parameter(new_anchors)

    @torch.no_grad()
    def build_new_user_adhoc(self, dataloader, device, n_outfits=1):
        uidxs, features, cates = self.build_new_user_features(dataloader, device)
        num_new_users = uidxs.max().item() + 1
        num_old_users = self.anchors.num_users
        num_u_anchors = self.anchors.num_u_anchors
        anchor_pool = self.anchors.u_anchors.data  # num_users x num_anchors x d
        anchor_pool = F.normalize(anchor_pool, dim=-1).view(-1, self.embd_dim)
        new_anchors = []
        for uidx in range(num_new_users):
            selected = ops.select_features(select_uidx=uidx, uidxs=uidxs)
            supp_feats = features[selected]  # (n_selected, n_items, d)
            supp_masks = cates[selected] != -1  # (n_selected, n_items, 1)
            # randomly select n_outfits
            index = torch.randperm(len(supp_feats), device=supp_feats.device)[:n_outfits]
            supp_feats = supp_feats[index]  # n_outfits x items x d
            supp_masks = supp_masks[index]  # n_outfits x items x 1
            supp_feats = rearrange(supp_feats, "b n d -> (b n) d")  # (n_outfits * n_items) x d
            supp_masks = rearrange(supp_masks, "b n 1 -> (b n) 1")  # (n_outfits * n_items) x 1
            supp_feats = (supp_feats * supp_masks).sum(dim=0, keepdim=True) / supp_masks.sum(dim=0, keepdim=True)
            supp_feats = F.normalize(supp_feats, dim=-1)  # 1 x d
            # compute similarity with anchor pool
            sims = anchor_pool @ supp_feats.t()  # (num_users * num_anchors) x 1
            sims = sims.view(-1)  # (num_users * num_anchors)
            if n_outfits == 1:
                # # use all positive anchors for stability
                # topk = sims >= 0.0
                # selected_anchors = anchor_pool[topk]  # num_selected x d
                # if len(selected_anchors) == 0:
                #     # fall back to selecting the best anchor
                #     _, topk = torch.topk(sims, k=1, largest=True)
                # selected_anchors = selected_anchors.mean(dim=0, keepdim=True)  # 1 x d
                # simply select top 16 anchors
                _, topk = torch.topk(sims, k=16, largest=True)
                selected_anchors = anchor_pool[topk]
                selected_anchors = F.normalize(selected_anchors, dim=-1)
            else:  # select num_u_anchors anchors
                _, topk = torch.topk(sims, k=num_u_anchors, largest=True)
                selected_anchors = anchor_pool[topk]  # num_u_anchors x d
            new_anchors.append(F.normalize(selected_anchors, dim=-1))
        new_anchors = torch.stack(new_anchors, dim=0)  # num_new_users x num_u_anchors x d
        logger.info(f"New user anchors built: {new_anchors.shape} from {n_outfits} outfits each.")
        # reset user anchors
        self.anchors.u_anchors = nn.Parameter(new_anchors)

    def setup_new_user_finetune(self):
        # disable all parameters except user anchors
        for name, param in self.named_parameters():
            if "u_anchors" not in name:
                param.requires_grad = False
