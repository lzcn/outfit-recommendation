import json
import os
import warnings
from typing import Dict, List

from torch.utils.data import DataLoader

import utils
from datasets.dataset import getOutfitData
from datasets.datum import Datum, getReader
from datasets.param import DataReaderConfig, OutfitDataConfig
from datasets.utils import load_outfit_tuples

logger = utils.logger.get_logger(__name__)


class OutfitDataModule(object):
    """Build an outfit dataset and a PyTorch DataLoader.

    Tuple files are stored under ``{data_root}/{partition}/``:

    - Standard modes read ``{phase}_pos`` and optional ``{phase}_neg``.
    - FITB with fixed candidates reads ``{phase}_pos_fitb`` and
      ``{phase}_neg_fitb``; online FITB reads ``{phase}_pos``.
    - Retrieval reads ``{phase}_pos``.

    The same partition directory contains ``items.json`` for item-key lookup.
    FITB and retrieval disable shuffling to preserve candidate order.
    ``num_workers`` controls DataLoader worker processes.
    """

    def __init__(
        self,
        data_root=None,
        partition="original",  # original outfit tuples partition
        dataset: OutfitDataConfig = None,
        readers: List[DataReaderConfig] = None,
        num_users=1,  # personalized recommendation when num_users > 1
        num_types=8,  # number of item types
        max_items=None,  # max number of items in an outfit
        phase="train",
        batch_size=16,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    ):

        dataset = dataset or OutfitDataConfig()
        readers = readers or [DataReaderConfig()]

        self.data_root = data_root
        self.partition = partition
        self.phase = phase
        self.num_users = num_users
        self.num_types = num_types
        self.max_items = max_items
        self._extra_info: Dict[str, int] = {}

        logger.info("Initializing OutfitDataModule(phase=%s, mode=%s)", phase, dataset.data_mode)

        # load outfit tuples
        pos_data, neg_data, shuffle = self._prepare_tuples(dataset, shuffle)
        if pos_data is not None:
            logger.info("Loaded positive tuples shape: %s", pos_data.shape)
        if neg_data is not None:
            logger.info("Loaded negative tuples shape: %s", neg_data.shape)

        # load item list
        json_file = os.path.join(data_root, self.partition, "items.json")
        if not os.path.exists(json_file):
            self.item_list = None
        else:
            with open(json_file, "r") as f:
                item_list = json.load(f)
            self.item_list = item_list
            assert self.num_types == len(item_list)

        # build datums
        datums: List[Datum] = []
        for reader_param in readers:
            reader = getReader(**reader_param)
            datums.append(Datum(item_list=self.item_list, reader=reader))
        self.datums = datums

        # build dataset and dataloader
        self.dataset = getOutfitData(datum=datums, param=dataset, pos_data=pos_data, neg_data=neg_data, phase=phase)
        logger.info(
            "Initializing DataLoader (batch_size=%d, num_workers=%d, shuffle=%s, drop_last=%s)",
            batch_size,
            num_workers,
            shuffle,
            drop_last,
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )
        logger.info("Data module ready!")

    def _prepare_tuples(self, dataset: OutfitDataConfig, shuffle=False):
        # specific preparation for different tasks
        self._extra_info = {}
        if dataset.data_mode == "FITB":
            if dataset.neg_mode == "Fix":
                pos_file = os.path.join(self.data_root, self.partition, f"{self.phase}_pos_fitb")
                neg_file = os.path.join(self.data_root, self.partition, f"{self.phase}_neg_fitb")
                pos_data = load_outfit_tuples(pos_file)
                neg_data = load_outfit_tuples(neg_file)
                num_answers = len(neg_data) // len(pos_data) + 1
                num_questions = len(pos_data)
                logger.info("Summary for fill-in-the-blank data set (Offline tuples)")
                logger.info("Number of FITB questions: %d", num_questions)
                logger.info("Number of FITB answers: %d", num_answers)
                self._extra_info = {"num_answers": num_answers, "num_questions": num_questions}
                return pos_data, neg_data, False
            else:
                pos_file = os.path.join(self.data_root, self.partition, f"{self.phase}_pos")
                pos_data = load_outfit_tuples(pos_file)
                num_answers = dataset.neg_param.get("ratio", 1) + 1
                num_questions = len(pos_data)
                logger.info("Summary for fill-in-the-blank data set (Online tuples)")
                logger.info("Number of FITB questions: %d", num_questions)
                logger.info("Number of FITB answers: %d", num_answers)
                self._extra_info = {"num_answers": num_answers, "num_questions": num_questions}
                return pos_data, None, False

        if dataset.data_mode == "Retrieval":
            file = os.path.join(self.data_root, self.partition, f"{self.phase}_pos")
            pos_data = load_outfit_tuples(file)
            self._extra_info = {"num_retrieval_outfits": len(pos_data)}
            logger.info("Summary for retrieval data set")
            logger.info("Number of retrieval outfits: %d", len(pos_data))
            return pos_data, None, False

        # default behavior
        pos_file = os.path.join(self.data_root, self.partition, f"{self.phase}_pos")
        neg_file = os.path.join(self.data_root, self.partition, f"{self.phase}_neg")
        pos_data = load_outfit_tuples(pos_file)
        neg_data = load_outfit_tuples(neg_file)
        return pos_data, neg_data, shuffle

    def next(self):
        """Generate outfits."""
        self.dataset.next()

    def build(self):
        """Generate outfits."""
        warnings.warn("build() is deprecated; use next() instead.", DeprecationWarning, stacklevel=2)
        self.dataset.next()

    def __len__(self):
        return self.num_batch

    @property
    def num_batch(self) -> int:
        try:
            loader = self.dataloader
            return len(loader)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} needs attribute dataloader")

    @property
    def num_sample(self) -> int:
        try:
            dataset = self.dataset
            return len(dataset)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} needs attribute dataset")

    def __iter__(self):
        for batch in self.dataloader:
            yield batch

    @property
    def num_answers(self) -> int:
        """Number of FITB answer choices per question."""
        return self._extra_info.get("num_answers")

    @property
    def num_questions(self) -> int:
        """Number of FITB questions in the dataset."""
        return self._extra_info.get("num_questions")

    @property
    def num_retrieval_outfits(self) -> int:
        """Number of outfits for retrieval ranking."""
        return self._extra_info.get("num_retrieval_outfits")
