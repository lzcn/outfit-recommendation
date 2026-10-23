import os
import stat
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from utils.logger import get_logger

logger = get_logger(__name__)


def load_pretrained(net: nn.Module, path_or_state_dict, weights_only=True, strict=False) -> nn.Module:
    """Load weights loosely or strictly and log any mismatches."""

    if isinstance(path_or_state_dict, (str, os.PathLike)):
        logger.info("Loading pre-trained model from %s.", path_or_state_dict)
        state_dict = torch.load(path_or_state_dict, map_location="cpu", weights_only=weights_only)
    else:
        logger.info("Loading pre-trained model from state dict.")
        state_dict = path_or_state_dict

    real_net = net.module if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else net

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # delete "module." if the model is wrapped by DataParallel or DistributedDataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    if strict:
        real_net.load_state_dict(state_dict, strict=True)
        logger.info("All checkpoint keys and shapes matched (strict loading).")
        return net

    net_param = real_net.state_dict()
    unmatched_keys = [k for k, v in state_dict.items() if k in net_param and v.shape != net_param[k].shape]
    if unmatched_keys:
        state_dict = {k: v for k, v in state_dict.items() if k not in unmatched_keys}

    load_result = real_net.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    if unmatched_keys:
        missing_keys = [k for k in missing_keys if k not in set(unmatched_keys)]

    if not (missing_keys or unexpected_keys or unmatched_keys):
        logger.info("All keys matched.")
    else:
        if missing_keys:
            logger.info("Params not found in checkpoint (%d): %s", len(missing_keys), ", ".join(missing_keys))
        if unexpected_keys:
            logger.info("Params not used by model (%d): %s", len(unexpected_keys), ", ".join(unexpected_keys))
        if unmatched_keys:
            logger.info("Params with mismatched shape (%d): %s", len(unmatched_keys), ", ".join(unmatched_keys))

    if strict:
        assert not (missing_keys or unexpected_keys or unmatched_keys), "Strict loading failed."
    return net


class ModelSaver:
    """Save model checkpoints to disk with automatic best model tracking.

    Args:
        dirname (str): Directory for checkpoints.
        filename_prefix (str, optional): Filename prefix.
        score_name (str, optional): Optional metric name included in checkpoint filenames.
        n_saved (int, optional): Number of models to keep. Defaults to 5.
        save_latest (bool, optional): Save latest model. Defaults to False.
        save_best (bool, optional): Save best model copy. Defaults to False.
        mode (str, optional): "max" or "min" for best selection.
        atomic (bool, optional): Use atomic writes for safety. Defaults to True.
        create_dir (bool, optional): Create dirname if needed. Defaults to True.
        require_empty (bool, optional): Require empty directory. Defaults to True.

    Examples:
        .. code-block:: python

            saver = ModelSaver(dirname="checkpoints", filename_prefix="model")
            for epoch in range(max_epochs):
                # train
                score = evaluate(model)
                saver.save(model, score, epoch)
            best_model = saver.best_checkpoint
            model.load_state_dict(torch.load(best_model))
    """

    def __init__(
        self,
        dirname: str,
        filename_prefix: str = None,
        score_name: str = None,
        n_saved: int = 5,
        save_latest: bool = False,
        save_best: bool = False,
        mode="max",
        atomic: bool = True,
        create_dir: bool = True,
        require_empty: bool = True,
    ):
        super().__init__()
        score = np.inf if mode == "min" else -np.inf
        self.dirname = dirname
        self.filename_prefix = filename_prefix
        self.score_name = score_name
        self.n_saved = n_saved
        self.save_latest = save_latest
        self.save_best = save_best
        self.mode = mode
        self.history = [(score, False) for _ in range(n_saved)]
        self.best_checkpoint: str = None
        self.atomic = atomic

        if create_dir and not os.path.exists(dirname):
            os.makedirs(dirname)
        if require_empty and os.listdir(dirname):
            raise ValueError(f"Directory {dirname} is not empty")

    def filename(self, score: float = None, epoch: int = None, latest=False, best=False) -> str:
        prefix = f"{self.filename_prefix}_" if self.filename_prefix else ""
        score_name = f"{self.score_name}_" if self.score_name else ""
        if latest:
            return f"{self.dirname}/{prefix}latest.pt"
        if best:
            return f"{self.dirname}/{prefix}best.pt"
        if score is None and epoch is not None:
            filename = f"{self.dirname}/{prefix}epoch_{epoch}.pt"
        elif score is not None and epoch is None:
            filename = f"{self.dirname}/{prefix}{score_name}{score:.4f}.pt"
        else:
            filename = f"{self.dirname}/{prefix}{score_name}{score:.4f}_epoch_{epoch}.pt"
        return filename

    def _is_worst(self, score):
        # assume that self.history is sorted with first being the worst
        return (score < self.history[0][0]) if self.mode == "max" else (score > self.history[0][0])

    def _is_best(self, score):
        # assume that self.history is sorted with last item being the best
        return (score >= self.history[-1][0]) if self.mode == "max" else (score <= self.history[-1][0])

    def _sort_history(self):
        # sort the history from worst to best
        if self.mode == "max":
            self.history.sort(key=lambda x: x[0])
        else:
            self.history.sort(key=lambda x: x[0], reverse=True)

    def _save_func(self, checkpoint: dict, path: Path, func: callable) -> None:
        if not self.atomic:
            func(checkpoint, path)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
            tmp_file = tmp.file
            tmp_name = tmp.name
            try:
                func(checkpoint, tmp_file)
            except BaseException:
                tmp.close()
                os.remove(tmp_name)
                raise
            else:
                tmp.close()
                os.replace(tmp.name, path)
                # append group/others read mode
                os.chmod(path, os.stat(path).st_mode | stat.S_IRGRP | stat.S_IROTH)

    def save(self, model: nn.Module, score, epoch=None):
        """Save model checkpoint to disk.

        Filename format: {filename_prefix}[_{score_name}]_{score:.4f}[_{epoch}].pt
        Optional components: [_{score_name}] and [_{epoch}]

        Args:
            model (nn.Module): Model to save.
            score (float, Optional): Model score.
            epoch (Number, Optional): Current epoch.
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self._is_worst(score) and not self.save_latest:
            return
        state_dict = model.state_dict()
        if self.save_latest:
            self._save_func(state_dict, self.filename(latest=True), torch.save)
        if self.save_best and self._is_best(score):
            self._save_func(state_dict, self.filename(best=True), torch.save)
        filename = self.filename(score, epoch)
        if self._is_worst(score):
            pass
        else:
            # replace the worst model
            if self.history[0][1] and os.path.exists(self.history[0][1]):
                os.remove(self.history[0][1])
            self._save_func(state_dict, filename, torch.save)
            self.history[0] = (score, filename)
        self._sort_history()
        self.best_checkpoint = self.history[-1][-1]
