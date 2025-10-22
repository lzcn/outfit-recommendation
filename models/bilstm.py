import math
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchutils
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import datasets
import utils

logger = utils.get_logger(__name__)


class ContrastiveEmbeddingLoss(nn.Module):
    def __init__(self, i_dim, t_dim, c_dim, margin=0.2, word_embedding=False):
        """Compute visual-semantic embedding loss.

        Args:
            i_dim (int): Dimension for image data.
            t_dim (int): Dimension for text data.
            c_dim (int): Dimension for embedding space.
            margin (float, optional): Margin for loss. Defaults to 0.2.
            word_embedding (bool, optional): Whether input is word embedding. Defaults to False.
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
        loss = torchutils.loss.contrastive_loss(i_feat, t_feat, self.margin, mask=mask, reduction="mean")
        return loss


class BiLSTM(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        hidden_dim=512,
        num_layers=1,
        dropout=0.3,
        backbone=None,
        use_nn_feature=False,
        semantic_dim=2400,
        word_embedding=False,
        use_vse=False,
        vse_dim=512,
        margin=0.2,
        loss_weight=None,
        max_items=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        feat_dim = backbone.pop("dim")
        self.backbone = nn.Identity() if use_nn_feature else utils.backbone(**backbone)
        self.dense_input = nn.Sequential(nn.Linear(feat_dim, embed_dim), nn.Dropout(dropout))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.f_output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim))
        self.b_output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim))
        self.tau = math.sqrt(embed_dim)
        self.use_vse = use_vse
        if use_vse:
            self.vse = ContrastiveEmbeddingLoss(feat_dim, semantic_dim, vse_dim, margin, word_embedding=word_embedding)
        else:
            self.vse = nn.Identity()
        self.init()
        self.feat_cache = {}
        self.name_dict = None
        self.feat_list = None
        self.max_items = max_items

        default_loss_weight = {"vse": 1.0, "f_loss": 1.0, "b_loss": 1.0}
        if loss_weight is not None:
            default_loss_weight.update(loss_weight)

        self.loss_weight = default_loss_weight

    def init(self):
        """U(-0.08, 0.08) in original paper."""
        scale = 0.08
        for name, param in self.dense_input.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -scale, scale)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -scale, scale)
        for name, param in self.f_output.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -scale, scale)
        for name, param in self.b_output.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -scale, scale)

    @torch.no_grad()
    def cache_batch_feature(self, seq_data: torch.Tensor, seq_name: List[str], seq_size: torch.Tensor):
        """To build features cache, we need to run through all data."""
        # get image features
        b, t, *im_size = seq_data.shape
        seq_feat = self.backbone(seq_data.view(-1, *im_size))
        seq_feat = self.dense_input(seq_feat)
        # b x t x d
        seq_feat = seq_feat.view(b, t, -1).cpu().numpy()
        for feat, name, size in zip(seq_feat, seq_name, seq_size):
            name = name.split(",")
            assert len(feat) == len(name)
            for i in range(size):
                self.feat_cache[name[i]] = feat[i]

    def clear_feature_cache(self):
        """Clear feature cache."""
        self.feat_cache = {}
        self.name_dict = None
        self.feat_list = None

    def build_feature_cache(self):
        """Build feature cache."""
        feat_list = []
        name_dict = {}
        for i, key in enumerate(self.feat_cache.keys()):
            name_dict[key] = i
            feat_list.append(self.feat_cache[key])
        # add zero-embedding
        feat_list.append(np.zeros(self.embed_dim, dtype=np.float32))
        feat_list = np.vstack(feat_list)
        feat_list = torch.from_numpy(feat_list).cuda()
        self.name_dict = name_dict
        self.feat_list = feat_list

    @torch.no_grad()
    def setup_for_evaluation(self, data_cfg, device):
        """Build feature cache for validation or testing."""
        changes = {"dataset": {"data_mode": "PositiveOutfit", "pos_mode": "Identity"}}
        data_cfg = OmegaConf.merge(data_cfg, changes)
        logger.debug(f"Building feature cache with config:\n{OmegaConf.to_yaml(data_cfg)}")
        dataloader = datasets.dataloader.OutfitDataModule(**data_cfg)

        self.eval()

        self.feat_cache = {}
        self.name_dict = None
        self.feat_list = None

        dataloader.build()
        for batch in tqdm(dataloader, desc="Build feature cache"):
            seq_name = batch["name"]
            seq_data = batch["data"].to(device)
            seq_size = batch["size"]
            # get image features
            b, t, *im_size = seq_data.shape
            seq_feat = self.backbone(seq_data.view(-1, *im_size))
            seq_feat = self.dense_input(seq_feat)
            # b x t x d
            seq_feat = seq_feat.view(b, t, -1).cpu().numpy()
            for feat, name, size in zip(seq_feat, seq_name, seq_size):
                name = name.split(",")
                assert len(feat) == len(name)
                for i in range(size):
                    self.feat_cache[name[i]] = feat[i]

        feat_list = []
        name_dict = {}
        for i, key in enumerate(self.feat_cache.keys()):
            name_dict[key] = i
            feat_list.append(self.feat_cache[key])
        # add zero-embedding
        feat_list.append(np.zeros(self.embed_dim, dtype=np.float32))
        feat_list = np.vstack(feat_list)
        feat_list = torch.from_numpy(feat_list).cuda()
        self.name_dict = name_dict
        self.feat_list = feat_list

    @torch.no_grad()
    def compatibility(self, sequence: List[str]):
        """Compute the compatibility score for a given sequence of item names."""

        def _get_fIndex(index):
            x = torch.arange(len(index))
            y = torch.LongTensor(index[1:] + [-1])
            return x, y

        def _get_bIndex(index):
            x = torch.arange(len(index))
            y = torch.LongTensor([-1] + index[:-1])
            return x, y

        seq_index = [self.name_dict[name] for name in sequence]
        # 1 x t x d
        seq_feat = self.feat_list[seq_index].unsqueeze(0)
        output, _ = self.lstm(seq_feat)
        dim = output.shape[-1] // 2
        f_input, b_input = output.squeeze(dim=0).split(dim, dim=-1)
        f_output = self.f_output(f_input)
        b_output = self.b_output(b_input)
        # re-scale the softmax with number of dimension
        feat_target = self.feat_list.permute(1, 0)
        fw_logprob = F.log_softmax(torch.mm(f_output, feat_target) / self.tau, dim=-1)
        bw_logprob = F.log_softmax(torch.mm(b_output, feat_target) / self.tau, dim=-1)
        # The first one is used to predict the second item
        f_score = fw_logprob[_get_fIndex(seq_index)].mean()
        b_score = bw_logprob[_get_bIndex(seq_index)].mean()
        score = f_score + b_score
        return score

    @torch.no_grad()
    def item_scores(self, sequence: List[str]):
        """Compute the item scores for a given sequence of item names."""

        def _get_fIndex(index):
            x = torch.arange(len(index))
            y = torch.LongTensor(index[1:] + [-1])
            return x, y

        def _get_bIndex(index):
            x = torch.arange(len(index))
            y = torch.LongTensor([-1] + index[:-1])
            return x, y

        seq_index = [self.name_dict[name] for name in sequence]
        num_items = len(seq_index)
        # 1 x t x d
        seq_feat = self.feat_list[seq_index].unsqueeze(0)
        output, _ = self.lstm(seq_feat)
        dim = output.shape[-1] // 2
        f_input, b_input = output.squeeze(dim=0).split(dim, dim=-1)
        f_output = self.f_output(f_input)
        b_output = self.b_output(b_input)
        # re-scale the softmax with number of dimension
        feat_target = self.feat_list.permute(1, 0)
        fw_logprob = F.log_softmax(torch.mm(f_output, feat_target) / self.tau, dim=-1)
        bw_logprob = F.log_softmax(torch.mm(b_output, feat_target) / self.tau, dim=-1)
        # The first one is used to predict the second item
        # fw: [1, 2, ..., n - 1, X]
        # bw: [X, 0, 1, ..., n - 2]
        # X represents the unknown
        f_score = fw_logprob[_get_fIndex(seq_index)].cpu()
        b_score = bw_logprob[_get_bIndex(seq_index)].cpu()
        f_score[-1] = b_score[1]
        b_score[0] = f_score[-2]
        scores = torch.zeros(self.max_items)
        scores[:num_items] = f_score.roll(1) + b_score.roll(-1)
        return -scores

    def batch_item_scores(self, seq_name, seq_size):
        scores = []
        for names, size in zip(seq_name, seq_size):
            sequence = names.split(",")[:size]
            score = self.item_scores(sequence)
            scores.append(score)
        scores = torch.stack(scores, dim=0)
        return scores

    def batch_compatibility(self, seq_name, seq_size):
        scores = []
        for names, size in zip(seq_name, seq_size):
            sequence = names.split(",")[:size]
            score = self.compatibility(sequence)
            scores.append(score)
        return torch.stack(scores)

    @torch.no_grad()
    def _batch_compatibility(self, seq_name: List[str], seq_size: torch.Tensor):
        def _get_fIndex(indexes):
            x = torch.arange(len(indexes))
            y = torch.LongTensor(indexes[1:] + [-1])
            return x, y

        def _get_bIndex(indexes):
            x = torch.arange(len(indexes))
            y = torch.LongTensor([-1] + indexes[:-1])
            return x, y

        seq_feat = []
        seq_index = []
        for names in seq_name:
            index = [self.name_dict[key] for key in names.split(",")]
            feats = self.feat_list[index]
            seq_index.append(index)
            seq_feat.append(feats)
        # b x t x d
        seq_feat = torch.stack(seq_feat, dim=0)
        seq_index = torch.LongTensor(seq_index)
        # pack the sequences
        sorted_seq_size, sorted_indices = torch.sort(seq_size, descending=True)
        sorted_seq_index = seq_index.index_select(0, sorted_indices)
        # sort features
        sorted_indices = sorted_indices.to(seq_feat.device)
        sorted_seq_feat = seq_feat.index_select(0, sorted_indices)
        # pack features
        packed_seq_feat = pack_padded_sequence(sorted_seq_feat, sorted_seq_size, batch_first=True)
        # forward and backward hidden states
        packed_output, _ = self.lstm(packed_seq_feat)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # b x t x d, in sorted order
        f_input, b_input = lstm_output.split(self.hidden_dim, dim=-1)
        f_output = self.f_output(f_input)
        b_output = self.b_output(b_input)
        # 1 x d x s
        feat_target = self.feat_list.permute(1, 0).unsqueeze(0)
        # b x t x s
        f_logp = F.log_softmax(torch.matmul(f_output, feat_target) / self.tau, dim=-1)
        b_logp = F.log_softmax(torch.matmul(b_output, feat_target) / self.tau, dim=-1)
        # save scores
        scores = [0] * len(seq_size)
        for i, size in enumerate(sorted_seq_size):
            index = sorted_seq_index[i].tolist()[:size]
            f_score = f_logp[i][_get_fIndex(index)].mean()
            b_score = b_logp[i][_get_bIndex(index)].mean()
            score = f_score + b_score
            scores[sorted_indices[i]] = score.item()
        return scores

    def forward(
        self, seq_data: Union[torch.Tensor, List[torch.Tensor]], seq_lens: torch.Tensor, seq_type: torch.Tensor
    ) -> dict:
        """Run bi-lstm.

        Args:
            seq_data (Union[torch.Tensor, List[torch.Tensor]]): batch_size x num_items x data_shape
            seq_lens (torch.Tensor): batch_size x item_size
            seq_type (torch.Tensor): batch_size x item_size

        """

        def _get_fIndex(size, start_id):
            """Get the target index for forward loss.
            0 -> 1 + start_id
            1 -> 2 + start_id
            ...
            n-1 -> -1  # zero embeding
            """
            x = torch.arange(size)
            y = torch.arange(size) + 1 + start_id
            y[-1] = -1
            return x, y

        def _get_bIndex(size, start_id):
            """Get the target index for forward loss.
            0 -> -1   # zero embeding
            1 -> 0 + start_id
            ...
            n-1 -> (n-2) + start_id
            """
            x = torch.arange(size)
            y = torch.arange(size) - 1 + start_id
            y[0] = -1
            return x, y

        if self.use_vse:
            i_data, t_data = seq_data
            seq_data = i_data
        # extract features with backbone
        b, t, *data_shape = seq_data.shape
        # (b t d)
        seq_data = seq_data.view(-1, *data_shape)
        i_data = self.backbone(seq_data)
        im_feats = self.dense_input(i_data)
        im_feats = im_feats.view(b, t, -1)
        if self.use_vse:
            vse_loss = self.vse(i_data, t_data, seq_type)
            loss = {"vse": vse_loss.view(-1, 1)}
        else:
            loss = {}
        # pack the sequences
        sorted_seq_lens, sorted_indices = torch.sort(seq_lens, descending=True)
        sorted_seq_lens = sorted_seq_lens.cpu()
        # sorted_indices = sorted_indices.to(im_feats.device)
        # sort features
        sorted_im_feats = im_feats.index_select(0, sorted_indices)
        # pack features
        packed_im_feats = pack_padded_sequence(sorted_im_feats, sorted_seq_lens, batch_first=True)
        # forward and backward hidden states
        packed_output, _ = self.lstm(packed_im_feats)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # b x t x d, in sorted order
        f_input, b_input = lstm_output.split(self.hidden_dim, dim=-1)
        f_output = self.f_output(f_input)
        b_output = self.b_output(b_input)

        # merge all batch features add zero embedding in the last
        im_target = [x[:size] for x, size in zip(sorted_im_feats, sorted_seq_lens)]
        im_target.append(sorted_im_feats.new_zeros(1, self.embed_dim))
        # d x n, n is the number of items in one batch
        im_target = torch.cat(im_target, dim=0).permute(1, 0)

        # compute the nll loss
        f_loss = []
        b_loss = []
        start_id = 0
        for i, size in enumerate(sorted_seq_lens):
            # t x n
            f_logp = F.log_softmax(torch.mm(f_output[i], im_target) / self.tau, dim=-1)
            b_logp = F.log_softmax(torch.mm(b_output[i], im_target) / self.tau, dim=-1)
            f_loss.append(-f_logp[_get_fIndex(size, start_id)].mean())
            b_loss.append(-b_logp[_get_bIndex(size, start_id)].mean())
            start_id += size
        f_loss = torch.stack(f_loss, dim=0).view(-1)
        b_loss = torch.stack(b_loss, dim=0).view(-1)
        loss.update(f_loss=f_loss, b_loss=b_loss)
        return loss

    def train_step(self, batch: dict, device):
        seq_data = batch["data"].to(device)
        seq_type = batch["cate"].to(device)
        seq_lens = batch["size"].to(device)

        loss_dict = self(seq_data, seq_lens, seq_type)

        total_loss = sum(self.loss_weight.get(k, 1.0) * v.mean() for k, v in loss_dict.items())
        metrics = {k: v.mean().item() for k, v in loss_dict.items()}

        return total_loss, metrics

    @torch.no_grad()
    def eval_compatibility(self, batch, device=None):
        """Compute compatibility scores for a batch of outfits.
        Requires feature cache to be built.
        """
        scores = self.batch_compatibility(batch["name"], batch["size"])
        return scores

    @torch.no_grad()
    def eval_outlier(self, batch, device=None):
        """Compute the outlier score for each item in the batch.
        Requires feature cache to be built.

        """
        scores = []
        for names, size in zip(batch["name"], batch["size"]):
            sequence = names.split(",")[:size]
            score = self.item_scores(sequence)
            scores.append(score)
        scores = torch.stack(scores, dim=0)
        return scores
