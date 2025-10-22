import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import attr

from datasets.param.base import Param, to_param
from datasets.param.optim import OptimParam
from datasets.param.reader import DataReaderParam

LOGGER = logging.getLogger(__name__)


@attr.s
class OutfitDataParam(Param):
    #: dataset format
    data_mode: str = attr.ib(default="PairwiseOutfit")
    #: dataset param
    data_param: dict = attr.ib(factory=dict)
    #: positive tuples generation mode
    pos_mode: str = attr.ib(default=None)
    #: negative tuples generation mode
    neg_mode: str = attr.ib(default=None)
    #: configuration for positive tuples generation
    pos_param: dict = attr.ib(factory=dict)
    #: configuration for negative tuples generation
    neg_param: dict = attr.ib(factory=dict)


@dataclass
class OutfitDataConfig:
    #: dataset format
    data_mode: str = "PairwiseOutfit"
    #: dataset param
    data_param: Dict = field(default_factory=dict)
    #: positive tuples generation mode
    pos_mode: str = "Fix"
    #: negative tuples generation mode
    neg_mode: str = "Fix"
    #: configuration for positive tuples generation
    pos_param: Dict = field(default_factory=dict)
    #: configuration for negative tuples generation
    neg_param: Dict = field(default_factory=dict)


@attr.s
class OutfitLoaderParam(Param):
    """Configuration class for data loader.

    Examples:

        .. code-block:: python

            kwargs = {
                "num_users": 1,
                "max_size": 8,
                "num_types": 8,
                "data_root": "data/maryland-polyvore",
                "readers": [
                    {
                        "reader": "TensorLMDB",
                        "path": "data/maryland-polyvore/features/resnet34",
                    },
                ],
                "dataset": {
                    "data_mode": "PointWise",
                    "posi_mode": "Fix",
                    "nega_mode": "RandomMix",
                },
                "batch_size": 64,
                "shuffle": False,
                "num_workers": 4,
                "phase": "train",
            }

            data_param = DataParam(**kwargs)
            outfit_data = OutfitData(data_param)
            for batch in outfit_data:
                // ....
    """

    # basic configurations
    #: data split of ["train", "valid", "test"]
    phase: str = attr.ib(default="train")
    #: data root, where tuples files are saved
    data_root: str = attr.ib(default="polyvore-u/processed/original/tuples_32", repr=False)
    tuple_data: str = attr.ib(default=None)
    #: readers :class:`DataReaderParam`
    readers: List[DataReaderParam] = attr.ib(factory=list)
    #: dataset :class:`OutfitDataParam`
    dataset: OutfitDataParam = attr.ib(factory=dict, converter=OutfitDataParam.from_dict)
    #: number of item categories
    num_types: int = attr.ib(default=10)
    #: max number of items in an outfit
    max_items: int = attr.ib(default=None)
    #: number of users
    num_users: int = attr.ib(default=1, converter=int)
    #: number of choice for FITB task
    num_fitb_choices: int = attr.ib(default=4, converter=int)
    # for data-lodaer
    #: batch size for dataloader
    batch_size: int = attr.ib(default=64, converter=int)
    #: number of workers for dataloader
    num_workers: int = attr.ib(default=8, converter=int)
    drop_last: bool = attr.ib(default=False)
    #: whether to shuffle the dataset
    shuffle: bool = attr.ib(default=None)
    # non-configurable attributes
    pos_fn: str = attr.ib(init=False)
    neg_fn: str = attr.ib(init=False)
    # pos_fitb_fn: str = attr.ib(init=False, repr=False)
    # neg_fitb_fn: str = attr.ib(init=False, repr=False)
    # fitb_fn: str = attr.ib(init=False, repr=False)
    # retrieval_fn: str = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        data_root = os.path.expanduser(self.data_root)
        if self.tuple_data is not None:
            data_root = os.path.join(data_root, self.tuple_data)
        # reader parameters
        self.readers = [dict()] if len(self.readers) == 0 else self.readers
        self.readers = [DataReaderParam.from_dict(param) for param in self.readers]
        self.pos_fn = os.path.join(data_root, f"{self.phase}_pos")
        self.neg_fn = os.path.join(data_root, f"{self.phase}_neg")
        self.pos_fitb_fn = os.path.join(data_root, f"{self.phase}_pos_fitb")
        self.neg_fitb_fn = os.path.join(data_root, f"{self.phase}_neg_fitb")
        self.fitb_fn = os.path.join(data_root, f"{self.phase}_fitb")
        self.retrieval_fn = os.path.join(data_root, "retrieval")
        self.item_list_fn = os.path.join(data_root, "items.json")


@dataclass
class OutfitLoaderConfig:
    phase: str = "train"
    data_root: str = "polyvore-u/processed/original/tuples_32"
    tuple_data: str = field(default=None)
    readers: List[DataReaderParam] = field(default_factory=list)
    dataset: OutfitDataConfig = field(default_factory=OutfitDataConfig)
    num_types: int = 10
    max_items: Optional[int] = None
    num_users: int = 1
    num_fitb_choices: int = 4
    batch_size: int = 64
    num_workers: int = 8
    drop_last: bool = False
    shuffle: Optional[bool] = False
    # post attributes which are not configurable
    pos_fn: str = field(init=False, repr=False)
    neg_fn: str = field(init=False, repr=False)

    def __post_init__(self):
        data_root = os.path.expanduser(self.data_root)
        if self.tuple_data is not None:
            data_root = os.path.join(data_root, self.tuple_data)
        self.readers = [DataReaderParam.from_dict(param) for param in self.readers]
        self.pos_fn = os.path.join(data_root, f"{self.phase}_pos")
        self.neg_fn = os.path.join(data_root, f"{self.phase}_neg")
        self.pos_fitb_fn = os.path.join(data_root, f"{self.phase}_pos_fitb")
        self.neg_fitb_fn = os.path.join(data_root, f"{self.phase}_neg_fitb")
        self.fitb_fn = os.path.join(data_root, f"{self.phase}_fitb")
        self.retrieval_fn = os.path.join(data_root, "retrieval")
        self.item_list_fn = os.path.join(data_root, "items.json")


@attr.s
class RunParam(Param):
    r"""Configuration interface for training/testing."""

    epochs: int = attr.ib(default=100)
    data_param: OutfitLoaderParam = attr.ib(factory=dict, converter=to_param)
    valid_data_param: OutfitLoaderParam = attr.ib(factory=dict)
    test_data_param: OutfitLoaderParam = attr.ib(factory=dict)
    train_data_param: OutfitLoaderParam = attr.ib(factory=dict)
    net_param: Param = attr.ib(factory=dict, converter=to_param)
    optim_param: OptimParam = attr.ib(factory=dict)
    summary_interval: int = attr.ib(default=10)
    display_interval: int = attr.ib(default=50)
    load_trained: str = attr.ib(default=None)
    log_dir: str = attr.ib(default="summaries/log")
    log_level: str = attr.ib(default="INFO")
    gpus: Union[int, list] = attr.ib(default=0)
    num_runs: int = attr.ib(default=1)

    def __attrs_post_init__(self):
        if isinstance(self.data_param, Param):
            self.train_data_param = attr.evolve(self.data_param, **self.train_data_param)
            self.valid_data_param = attr.evolve(self.data_param, **self.valid_data_param)
            self.test_data_param = attr.evolve(self.data_param, **self.test_data_param)
        self.optim_param = OptimParam(**self.optim_param)
        # for gpus
        gpus = [self.gpus] if isinstance(self.gpus, int) else self.gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        self.gpus = list(range(len(gpus)))
