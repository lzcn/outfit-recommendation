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
    """Unified data loader for outfit recommendation tasks.

    Provides a consistent interface for loading outfit datasets across
    multiple modes: pairwise, pointwise, fill-in-the-blank (FITB),
    retrieval, and others. Automatically constructs the dataset
    and DataLoader for training or evaluation.

    FITB and Retrieval modes enforce deterministic ordering for
    accurate metric computation. Other modes use standard positive
    and negative tuples.

    Parameters
    ----------
    data_root : str
        Root directory of the dataset.
    partition : str, default="original"
        Subdirectory or split name that contains tuple files.
    dataset : OutfitDataConfig, optional
        Configuration describing how the dataset should be constructed.
        If None, defaults to an empty OutfitDataConfig.
    readers : list of DataReaderConfig, optional
        List of data reader configurations defining how features are read.
        If None, defaults to a single DataReaderConfig.
    phase : str, default="train"
        Dataset phase indicator ("train", "val", "test").
    num_users : int, default=1
        Number of users for personalized recommendation settings.
    batch_size : int, default=16
        Batch size used for DataLoader.
    num_workers : int, default=4
        Number of worker threads for DataLoader.
    shuffle : bool, default=False
        Whether to shuffle data. Overridden internally for FITB/Retrieval modes.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.
    **kwargs : dict
        Additional arguments for flexibility in extended subclasses.

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        Constructed dataset instance.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batched access to the dataset.
    num_batch : int
        Number of batches per epoch.
    num_sample : int
        Total number of samples in the dataset.
    _extra_info : dict
        Internal storage for auxiliary statistics such as
        number of FITB questions or retrieval outfits.

    Notes
    -----
    Required files vary by dataset mode:

    1. Default modes (Pairwise, Pointwise, NPair, Triplet, Outlier):
        {partition}/{phase}_pos
        {partition}/{phase}_neg
        Standard positive and negative outfit tuples.

    2. FITB (Fill-In-The-Blank):
        Offline (neg_mode == "Fix"):
            {partition}/{phase}_pos_fitb
            {partition}/{phase}_neg_fitb
        Online (other neg_modes):
            {partition}/{phase}_pos
        FITB enforces shuffle=False for deterministic evaluation,
        allowing tuples to be reshaped for accuracy computation.

    3. Retrieval:
        {partition}/retrieval
        Retrieval enforces shuffle=False for consistent ranking.

    Common companion file:
        {data_root}/items.json - defines item key mapping for readers.
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
