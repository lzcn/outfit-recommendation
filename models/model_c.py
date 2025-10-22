import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from tqdm import tqdm

import utils
from models import blocks, ops

logger = utils.get_logger(__name__)


class CompletionModel(blocks.BaseModel):
    """Compute f(q|o,u) = (o + u)^t q for outfit completion task."""

    def score_function(
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
        batch, num_items, *shape = data.shape
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        feat = feat.view(batch, num_items, -1)
        cate = cate.view(batch, num_items, -1)
        z = self.visual_encoder.latent_z(feat, cate)
        q = self.visual_encoder.latent_x(pool_items, pool_cates)
        scores = self.anchors.item_score(q, z, uidx, use_g_term=False)
        if not self.training:
            return scores
        acc = (torch.argmax(scores, dim=-1) == 0).float().mean()
        logprob = -F.log_softmax(scores, dim=-1)
        nll = logprob[:, 0].mean()
        loss = dict(nll_c=nll)
        return loss, acc

    def train_completion(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        pool_data = utils.to(batch["pool_data"], device)
        pool_cate = utils.to(batch["pool_cate"], device)
        loss, accuracy = self.score_function(uidx, data, cate, pool_data, pool_cate)
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
        return self.score_function(uidx, data, cate, pool_data, pool_cate)

    @torch.no_grad()
    def build_new_user_features(self, dataloader, device):
        self.eval()
        dataloader.next()
        features, uidxs, cates = [], [], []
        for batch in tqdm(dataloader, desc="Building features", ncols=88):
            data = utils.to(batch["data"], device)
            uidx = utils.to(batch["uidx"], device)
            cate = utils.to(batch["cate"], device)
            n = data.size(0)
            # for outfit completion task
            batch, num, *shape = data.shape
            # get visual features
            data = data.view(-1, *shape)
            feat = self.visual_feature(data)
            feat = feat.view(batch, num, -1)
            cate = cate.view(batch, num, -1)
            x = self.visual_encoder.latent_x(feat, cate)  # b x n_items x d
            for i in range(n):
                features.append(x[i])
                uidxs.append(uidx[i])
                cates.append(cate[i])

        features = torch.stack(features, dim=0)  # total_outfits x n_items x d
        uidxs = torch.stack(uidxs, dim=0)  # total_outfits
        cates = torch.stack(cates, dim=0)  # total_outfits x n_items x 1
        logger.info(f"Features built: {features.shape}, Uidxs built: {uidxs.shape}, Cates built: {cates.shape}")
        return uidxs, features, cates

    @torch.no_grad()
    def build_new_user_closed_form(self, dataloader, device, n_outfits=1):
        uidxs, features, cates = self.build_new_user_features(dataloader, device)
        num_new_users = uidxs.max().item() + 1
        new_anchors = []
        for uidx in range(num_new_users):
            selected = ops.select_features(select_uidx=uidx, uidxs=uidxs)
            # support outfit from training dataset
            supp_masks = cates[selected] != -1  # (n_selected, n_items, 1)
            supp_feats = features[selected]  # (num_outfits, num_items, dim)
            # randomly select n_outfits
            index = torch.randperm(len(supp_feats), device=supp_feats.device)[:n_outfits]
            supp_feats = supp_feats[index]  # n_outfits x n_items x d
            supp_masks = supp_masks[index]  # n_outfits x n_items x 1
            supp_feats = (supp_feats * supp_masks).sum(dim=1, keepdim=True) / supp_masks.sum(dim=1, keepdim=True)
            supp_feats = supp_feats.mean(dim=0)
            u_proto = supp_feats.mean(dim=0)  # 1 x d
            u_proto = F.normalize(u_proto, dim=-1)
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


class GroupFITBModel(blocks.BaseModel):
    """Compute f(q|o,u) = (omega + u)^t g for outfit completion task."""

    @torch.no_grad()
    def build_new_user_features(self, dataloader, device):
        self.eval()
        dataloader.next()
        features, uidxs, cates = [], [], []
        for batch in tqdm(dataloader, desc="Building features", ncols=88):
            data = utils.to(batch["data"], device)
            uidx = utils.to(batch["uidx"], device)
            cate = utils.to(batch["cate"], device)
            n = data.size(0)
            # for outfit completion task
            batch, num, *shape = data.shape
            # get visual features
            data = data.view(-1, *shape)
            feat = self.visual_feature(data)
            feat = feat.view(batch, num, -1)
            cate = cate.view(batch, num, -1)
            x = self.visual_encoder.latent_f(feat, cate)  # b x n_items x d
            for i in range(n):
                features.append(x[i])
                uidxs.append(uidx[i])
                cates.append(cate[i])

        features = torch.stack(features, dim=0)  # total_outfits x n_items x d
        uidxs = torch.stack(uidxs, dim=0)  # total_outfits
        cates = torch.stack(cates, dim=0)  # total_outfits x n_items x 1
        logger.info(f"Features built: {features.shape}, Uidxs built: {uidxs.shape}, Cates built: {cates.shape}")
        return uidxs, features, cates

    @torch.no_grad()
    def build_new_user_closed_form(self, dataloader, device, n_outfits=1):
        uidxs, features, cates = self.build_new_user_features(dataloader, device)
        num_new_users = uidxs.max().item() + 1
        new_anchors = []
        for uidx in range(num_new_users):
            selected = ops.select_features(select_uidx=uidx, uidxs=uidxs)
            # support outfit from training dataset
            supp_masks = cates[selected] != -1  # (n_selected, n_items, 1)
            supp_feats = features[selected]  # (n_selected, num_items, dim)
            # randomly select n_outfits
            index = torch.randperm(len(supp_feats), device=supp_feats.device)[:n_outfits]
            supp_feats = supp_feats[index]  # n_outfits x n_items x d
            supp_masks = supp_masks[index]  # n_outfits x n_items x 1
            supp_feats = rearrange(supp_feats, "b n d -> (b n) d")  # (n_outfits * n_items) x d
            supp_masks = rearrange(supp_masks, "b n 1 -> (b n) 1")  # (n_outfits * n_items) x 1
            supp_feats = F.normalize(supp_feats, dim=-1)
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

    def score_function(self, uidx, data, cate, index):
        batch, num_outfits, num_items, *shape = data.shape
        # get visual features
        data = data.view(-1, *shape)
        feat = self.visual_feature(data)
        # reshape
        feat = feat.view(batch * num_outfits, num_items, -1)
        cate = cate.view(batch * num_outfits, num_items, -1)

        z = self.visual_encoder.latent_f(feat, cate)
        z = z.view(batch, num_outfits, num_items, -1)

        # select z[b, :, index[b], :] for each b
        idx = index[:, None, None, None].expand(-1, num_outfits, -1, z.shape[-1])
        z_selected = torch.gather(z, 2, idx).squeeze(2)  # (batch, num_answers, dim)

        scores = self.anchors.outfit_score(z_selected, uidx)
        return scores

    def train_group_fitb(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        index = utils.to(batch["index"], device)

        scores = self.score_function(uidx, data, cate, index)

        logprob = -F.log_softmax(scores, dim=-1)
        nll_c = logprob[:, 0].mean()
        loss = dict(nll_c=nll_c)

        overall_loss, metrics = utils.compute_weighted_losses(loss, self.loss_weight)

        accuracy = (torch.argmax(scores, dim=-1) == 0).float().mean()
        metrics["accuracy"] = accuracy.item()
        return overall_loss, metrics

    def eval_group_fitb(self, batch, device):
        uidx = utils.to(batch["uidx"], device)
        data = utils.to(batch["data"], device)
        cate = utils.to(batch["cate"], device)
        index = utils.to(batch["index"], device)  # b x num_questions
        scores = self.score_function(uidx, data, cate, index)
        return scores
