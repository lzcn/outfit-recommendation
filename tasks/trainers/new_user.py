from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import datasets
import utils
from tasks.registries import register_trainer
from tasks.trainers.base import BaseTrainer

logger = utils.get_logger(__name__)


def config_clone(cfg):
    container = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = OmegaConf.create(container)
    return new_cfg


@register_trainer("new_user")
class NewUserTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._build_model()
        self._setup_training()
        self._setup_data_loaders()
        self._setup_logging()
        self.iterations = 0

    def _build_model(self) -> Any:
        """Build and initialize the model."""
        self.model = instantiate(self.cfg.model).to(self.cfg.task.device)

        pretrained = self.cfg.task.get("load_pretrained")
        if pretrained:
            utils.load_pretrained(self.model, pretrained)
        logger.info(self.model)

    def _setup_training(self) -> None:
        """Only setup optimizer and scheduler for finetuning."""
        if self.cfg.task.new_user == "finetune":
            self.model.new_user_finetune()
            super()._setup_training()

    def _setup_data_loaders(self) -> None:
        """Setup train, validation and test data loaders."""
        cfg = self.cfg
        self.train_loader = datasets.dataloader.OutfitDataModule(**cfg.data.train)
        self.valid_loader = datasets.dataloader.OutfitDataModule(**cfg.data.valid)
        self.test_loader = datasets.dataloader.OutfitDataModule(**cfg.data.test)

    def _setup_logging(self) -> None:
        """Setup TensorBoard writer and model saver."""
        if self.cfg.task.new_user == "finetune":
            super()._setup_logging()

    def get_best_model(self):
        if self.cfg.task.new_user == "finetune":
            return super().get_best_model()
        return self.model

    def train(self) -> None:
        cfg = self.cfg.task
        device = cfg.device
        num_outfits = cfg.get("n_outfits", 1)
        if cfg.new_user == "finetune":
            self.model.new_user_finetune()
            self.super().train()
        if cfg.new_user == "closed_form":
            self.model.build_new_user_closed_form(self.train_loader, device, num_outfits)
        if cfg.new_user == "adhoc":
            self.model.build_new_user_adhoc(self.train_loader, device, num_outfits)
        if cfg.new_user == "omega_only":
            self.model.build_new_user_omega_only(self.train_loader, device, num_outfits)
        if cfg.new_user == "shrink":
            self.model.build_new_user_shrink(self.train_loader, device, num_outfits)
        if cfg.new_user == "rocchio":
            self.model.build_new_user_rocchio(self.train_loader, device, num_outfits)
        if cfg.new_user == "ridge":
            self.model.build_new_user_ridge(self.train_loader, device, num_outfits)
