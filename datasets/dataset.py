import logging
import multiprocessing as mp
import time
from typing import List

import numpy as np
import torch

from datasets.datum import Datum
from datasets.generator import getGenerator
from datasets.param import OutfitDataParam

from . import utils

_dataset_registry = {}


class BaseOutfitData(object):
    """Base class for outfit datasets.

    Subclasses must implement:
        - process: Process positive and negative tuples.
        - names: Get outfit names.
        - __getitem__: Get a data sample.
        - __len__: Get dataset length.

    Args:
        datum (List[Datum]): List of data readers.
        pos_data (np.ndarray): Positive outfit tuples.
        neg_data (np.ndarray, optional): Negative outfit tuples. Defaults to None.
        phase (str, optional): Dataset phase (only for logging). Defaults to "train".
        data_param (dict, optional): Parameters for data generation. Defaults to None.
        pos_mode (str, optional): Positive generator mode. Defaults to "Fix".
        neg_mode (str, optional): Negative generator mode. Defaults to "RandomMix".
        pos_param (dict, optional): Parameters for positive generator. Defaults to None.
        neg_param (dict, optional): Parameters for negative generator. Defaults to None.

    Notes:
        Data Structure:
            Datasets contain positive and negative tuples. Generators control how tuples
            are created and refreshed each epoch. Subclasses return either pairwise outfits
            (positive and negative pairs) or pointwise outfits (single labeled outfits).

        Custom Generation:
            Some specialized datasets require task-specific negative generation that generic
            generators cannot achieve, so they implement custom generation logic.

        Example - FITB Task:
            Each positive outfit is paired with n negatives that differ only at one specific
            position (coordinated generation). The model scores all candidates, and accuracy
            is computed by checking if the positive (first answer) ranks highest.
    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        _dataset_registry[cls.__name__] = cls

    def __init__(
        self,
        datum: List[Datum],
        pos_data: np.ndarray,
        neg_data: np.ndarray = None,
        phase="train",
        data_param: dict = None,
        pos_mode: str = None,
        neg_mode: str = None,
        pos_param: dict = None,
        neg_param: dict = None,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.datum = datum
        self.phase = phase

        self.data_param = data_param or {}

        # save the initial data
        self.ini_pos = pos_data
        self.ini_neg = neg_data

        # placeholders for current data
        self.pos_data = pos_data
        self.neg_data = neg_data

        self.num_type = utils.infer_num_type(pos_data) if pos_data is not None else 0
        self.max_size = utils.infer_max_size(pos_data) if pos_data is not None else self.max_size
        self.sections = [1, 1, self.max_size, self.max_size]

        pos_param = pos_param or {}
        neg_param = neg_param or {}
        data_param = data_param or {}
        self.pos_generator = getGenerator(pos_mode, data=pos_data, **pos_param)
        self.neg_generator = getGenerator(neg_mode, data=neg_data, **neg_param)

        # multiprocessing
        self._num_mp_workers = data_param.get("mp_workers", 0)
        self._multiprocessing = self._num_mp_workers > 0
        self._queue = None
        self._processes = []
        self._stop_event = None
        if self._multiprocessing:
            self._queue = mp.Queue(maxsize=self._num_mp_workers * 2)
            self._stop_event = mp.Event()
            for i in range(self._num_mp_workers):
                proc = mp.Process(target=self._produce, args=(i,), daemon=True)
                proc.start()
                self._processes.append(proc)

        self.next()

    def _produce(self, worker_id):
        """Produce data with multiprocessing."""
        # Set different random seed for each worker (ensure within valid range)
        seed = (int(time.time() * 1000) + worker_id) % (2**32)
        np.random.seed(seed)

        while not self._stop_event.is_set():
            if self._queue.full():
                time.sleep(0.01)
                continue
            ini_data = self.ini_pos
            pos_data = self.pos_generator(ini_data) if self.pos_generator is not None else self.pos_data
            neg_data = self.neg_generator(pos_data) if self.neg_generator is not None else self.neg_data
            self._queue.put((pos_data, neg_data))

    def next(self):
        """Prepare tuples for next epoch."""
        if self._multiprocessing:
            if self._queue.qsize() == 0:
                self.logger.warning("Queue is empty, prefetching data may be slow.")
            self.pos_data, self.neg_data = self._queue.get()
        else:
            if self.pos_generator is not None:
                self.pos_data = self.pos_generator(self.ini_pos)
            if self.neg_generator is not None:
                self.neg_data = self.neg_generator(self.pos_data)

        # get data statistics
        if self.pos_data is not None:
            self.num_type = utils.infer_num_type(self.pos_data)
            self.max_size = utils.infer_max_size(self.pos_data)
            self.sections = [1, 1, self.max_size, self.max_size]
            self.logger.info(f"Positive outfit tuples ready for next epoch ({self.phase}): shape={self.pos_data.shape}")

        if self.neg_data is not None:
            self.logger.info(f"Negative outfit tuples ready for next epoch ({self.phase}): shape={self.neg_data.shape}")

        self.process()

    def cleanup(self):
        """Cleanup multiprocessing resources."""
        if self._multiprocessing and self._processes:
            self._stop_event.set()
            for proc in self._processes:
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
            self._processes.clear()

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def process(self):
        raise NotImplementedError

    def names(self, n):
        raise NotImplementedError

    def __getitem__(self, n):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class PairwiseOutfit(BaseOutfitData):
    def __getitem__(self, n):
        # in current implementation, pos_types == neg_types
        pos_items, pos_types = self.pos_items[n], self.pos_types[n]
        neg_items, neg_types = self.neg_items[n], self.neg_types[n]
        pos_args = (pos_items, pos_types, self.max_size)
        neg_args = (neg_items, neg_types, self.max_size)
        data = []
        if len(self.datum) == 1:
            datum = self.datum[0]
            data = torch.stack([datum.get_data(*pos_args), datum.get_data(*neg_args)], dim=0)
        else:
            data = [torch.stack([datum.get_data(*pos_args), datum.get_data(*neg_args)], dim=0) for datum in self.datum]
        # item category: shape 2 x n
        cate = torch.stack([torch.tensor(pos_types), torch.tensor(neg_types)], dim=0)
        # outfit size: shape 2
        size = torch.tensor([self.pos_sizes[n], self.neg_sizes[n]])
        name = (",".join(self.datum[0].get_key(*pos_args)), ",".join(self.datum[0].get_key(*neg_args)))
        return dict(data=data, name=name, size=size, uidx=self.uidxs[n], cate=cate)

    def __len__(self):
        if not hasattr(self, "uidxs"):
            return 1
        return len(self.uidxs)

    def process(self):
        # split tuples
        pos_uidx, pos_sizes, pos_items, pos_types = utils.split_tuple(self.pos_data)
        neg_uidx, neg_sizes, neg_items, neg_types = utils.split_tuple(self.neg_data)
        # check tuples
        ratio = len(self.neg_data) // len(self.pos_data)
        assert (ratio * len(self.pos_data)) == len(self.neg_data)
        assert (pos_uidx.repeat(ratio, axis=0) == neg_uidx).all()
        # save tuples
        self.uidxs = neg_uidx
        # positive data
        self.pos_sizes = pos_sizes.repeat(ratio, axis=0)
        self.pos_items = pos_items.repeat(ratio, axis=0)
        self.pos_types = pos_types.repeat(ratio, axis=0)
        # negative data
        self.neg_sizes = neg_sizes
        self.neg_items = neg_items
        self.neg_types = neg_types


class NPairOutfit(BaseOutfitData):
    """N-pair outfit dataset.

    Returns a list of outfits with the first being positive
    and the rest being negatives.
    """

    def __getitem__(self, n):
        # in current implementation, pos_types == neg_types
        items = []
        types = []
        size = []
        name = []
        # for positive outfit
        pos_items, pos_types = self.pos_items[n], self.pos_types[n]
        items.append(pos_items)
        types.append(pos_types)
        size.append(self.pos_sizes[n])
        name.append(",".join(self.datum[0].get_key(pos_items, pos_types, self.max_size)))
        # for negative outfits
        neg_index = [n * self.num_neg + i for i in range(self.num_neg)]
        for idx in neg_index:
            items.append(self.neg_items[idx])
            types.append(self.neg_types[idx])
            size.append(self.neg_sizes[idx])
            name.append(",".join(self.datum[0].get_key(self.neg_items[idx], self.neg_types[idx], self.max_size)))
        data = []
        for datum in self.datum:
            x = [datum.get_data(item_id, item_type, self.max_size) for item_id, item_type in zip(items, types)]
            data.append(torch.stack(x, dim=0))
        # [num_modalities] x (1 + num_neg) x data_shape
        data = data[0] if len(self.datum) == 1 else torch.stack(data, dim=0)
        # item category: shape (1 + num_neg) x n
        cate = torch.stack([torch.tensor(x) for x in types], dim=0)
        # outfit size: shape (1 + num_neg)
        size = torch.tensor(size)
        name = tuple(name)
        return dict(data=data, name=name, size=size, uidx=self.uidxs[n], cate=cate)

    def __len__(self):
        if not hasattr(self, "uidxs"):
            return 1
        return len(self.uidxs)

    def process(self):
        # split tuples
        pos_uidx, pos_sizes, pos_items, pos_types = utils.split_tuple(self.pos_data)
        neg_uidx, neg_sizes, neg_items, neg_types = utils.split_tuple(self.neg_data)
        # check tuples
        ratio = len(self.neg_data) // len(self.pos_data)
        assert (ratio * len(self.pos_data)) == len(self.neg_data)
        assert (pos_uidx.repeat(ratio, axis=0) == neg_uidx).all()
        # number of negatives
        self.num_neg = ratio
        # save tuples
        self.uidxs = pos_uidx
        # positive data
        self.pos_sizes = pos_sizes
        self.pos_items = pos_items
        self.pos_types = pos_types
        # negative data
        self.neg_sizes = neg_sizes
        self.neg_items = neg_items
        self.neg_types = neg_types


class SubsetData(PairwiseOutfit):
    def __getitem__(self, n):
        # in current implementation, pos_types == neg_types
        item_size = self.pos_sizes[n]
        index = np.random.randint(item_size)
        sub_items, sub_types = list(self.pos_items[n]), list(self.pos_types[n])
        pos_item = sub_items.pop(index)
        pos_type = sub_types.pop(index)
        neg_item, neg_type = self.neg_items[n][index], self.neg_types[n][index]
        if len(self.datum) == 1:
            datum = self.datum[0]
            sub_data = datum.get_data(sub_items, sub_types, self.max_size - 1)
            pos_data = datum.get_item(pos_item, pos_type)
            neg_data = datum.get_item(neg_item, neg_type)
        else:
            # m x n x *
            sub_data = [datum.get_data(sub_items, sub_types, self.max_size - 1) for datum in self.datum]
            pos_data = [datum.get_item(pos_item, pos_type) for datum in self.datum]
            neg_data = [datum.get_item(neg_item, neg_type) for datum in self.datum]
        cate = torch.tensor(sub_types)
        return dict(
            data=sub_data,
            size=item_size - 1,
            uidx=self.uidxs[n],
            cate=cate,
            pos_data=pos_data,
            neg_data=neg_data,
            pos_cate=pos_type,
            neg_cate=neg_type,
        )


class SubsetDataFixType(PairwiseOutfit):
    def __getitem__(self, n):
        # in current implementation, pos_types == neg_types
        item_size = self.pos_sizes[n]
        index = np.random.randint(item_size)
        for index in range(item_size):
            if self.pos_types[n][index] == 0:
                break
        sub_items, sub_types = list(self.pos_items[n]), list(self.pos_types[n])
        pos_item = sub_items.pop(index)
        pos_type = sub_types.pop(index)
        neg_item, neg_type = self.neg_items[n][index], self.neg_types[n][index]
        if len(self.datum) == 1:
            datum = self.datum[0]
            sub_data = datum.get_data(sub_items, sub_types, self.max_size - 1)
            pos_data = datum.get_item(pos_item, pos_type)
            neg_data = datum.get_item(neg_item, neg_type)
        else:
            # m x n x *
            sub_data = [datum.get_data(sub_items, sub_types, self.max_size - 1) for datum in self.datum]
            pos_data = [datum.get_item(pos_item, pos_type) for datum in self.datum]
            neg_data = [datum.get_item(neg_item, neg_type) for datum in self.datum]
        cate = torch.tensor(sub_types)
        return dict(
            data=sub_data,
            size=item_size - 1,
            uidx=self.uidxs[n],
            cate=cate,
            pos_data=pos_data,
            neg_data=neg_data,
            pos_cate=pos_type,
            neg_cate=neg_type,
        )


class PointwiseOutfit(BaseOutfitData):
    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        # m x n x data_shape
        data = [datum.get_data(items, types, self.max_size) for datum in self.datum]
        data = data[0] if len(self.datum) == 1 else data
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            label=self.labels[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size)),
            cate=cate,
        )

    def __len__(self):
        return len(self.uidxs)

    def process(self):
        tuples = np.vstack((self.pos_data, self.neg_data))
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(tuples)
        self.labels = np.array(([1] * len(self.pos_data) + [0] * len(self.neg_data)))


class PositiveOutfit(PointwiseOutfit):
    def build(self):
        self.logger.info("Generating positive tuples.")
        self.pos_data = self.pos_generator(self.ini_pos)
        self.logger.info("Positive tuples shape: {}".format(self.pos_data.shape))
        self.max_size = utils.infer_max_size(self.pos_data)
        self.sections = [1, 1, self.max_size, self.max_size]
        self.process()

    def process(self):
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(self.pos_data)
        self.labels = np.array([1] * len(self.pos_data))


class NegativeOutfit(PointwiseOutfit):
    def process(self):
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(self.neg_data)
        self.labels = np.array([0] * len(self.neg_data))


class FITB(PointwiseOutfit):
    """Pointwise dataset for Fill-In-The-Blank task.

    pos_data: [o_1, ..., o_i, ..., o_n], where o_i is positive outfit
    neg_data: [
        o_11, ..., o_1j, ..., o_1s,
        ...,
        o_i1, ..., o_ij, ..., o_is,
        ...
        o_n1, ..., o_nj, ..., o_ns,
    ], where o_ij is the j-th negative outfit for i-th positive
    """

    def process(self):
        num_questions = len(self.pos_data)
        num_negatives = len(self.neg_data) // len(self.pos_data)
        num_answers = num_negatives + 1
        self.logger.info("Number of FITB questions: %s", num_questions)
        self.logger.info("Number of FITB answers: %s", num_answers)
        pos_data = self.pos_data.reshape((num_questions, 1, -1))
        neg_data = self.neg_data.reshape((num_questions, num_negatives, -1))
        # num_questions x num_answers
        outfits = np.concatenate((pos_data, neg_data), axis=1).reshape((num_questions * num_answers, -1))
        pos_label = np.ones((num_questions, 1), dtype=np.int64)
        neg_label = np.zeros((num_questions, num_negatives), dtype=np.int64)
        self.labels = np.hstack((pos_label, neg_label)).reshape((-1, 1))
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(outfits)
        # find the index for answer item
        mask = neg_data[:, 0, :] == pos_data[:, 0, :]
        # the index of answer item
        index_mask = ~utils.split_tuple(mask)[2]
        assert index_mask.sum() == num_questions, f"Number of answer items {index_mask.sum()} != {num_questions}"
        _, item_index = np.where(index_mask)
        self.item_index = item_index.repeat(num_answers)
        assert len(self.item_index) == len(self.labels)

    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        # m x n x data_shape
        data = [datum.get_data(items, types, self.max_size) for datum in self.datum]
        data = data[0] if len(self.datum) == 1 else data
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            label=self.labels[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size)),
            cate=cate,
            index=self.item_index[n],
        )


class GroupFITB(BaseOutfitData):
    def process(self):
        num_questions = len(self.pos_data)
        num_negatives = len(self.neg_data) // len(self.pos_data)
        num_answers = num_negatives + 1
        self.logger.info("Number of FITB questions: %s", num_questions)
        self.logger.info("Number of FITB answers: %s", num_answers)
        pos_data = self.pos_data.reshape((num_questions, 1, -1))
        neg_data = self.neg_data.reshape((num_questions, num_negatives, -1))
        # num_questions x num_answers
        outfits = np.concatenate((pos_data, neg_data), axis=1).reshape((num_questions * num_answers, -1))
        pos_label = np.ones((num_questions, 1), dtype=np.int64)
        neg_label = np.zeros((num_questions, num_negatives), dtype=np.int64)
        self.labels = np.hstack((pos_label, neg_label)).reshape((-1, 1))
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(outfits)
        # find the index for answer item
        mask = neg_data[:, 0, :] == pos_data[:, 0, :]
        # the index of answer item
        index_mask = ~utils.split_tuple(mask)[2]
        assert index_mask.sum() == num_questions, f"Number of answer items {index_mask.sum()} != {num_questions}"
        _, item_index = np.where(index_mask)
        self.item_index = item_index.repeat(num_answers)
        self.num_answers = num_answers
        self.num_questions = num_questions
        assert len(self.item_index) == len(self.labels)

    def __len__(self):
        return self.num_questions

    def __getitem__(self, n):
        all_data = []
        all_cate = []
        pos_idx = n * self.num_answers
        pos_items, pos_types = self.items[pos_idx], self.types[pos_idx]
        # m x n x data_shape
        pos_data = [datum.get_data(pos_items, pos_types, self.max_size) for datum in self.datum]
        pos_data = pos_data[0] if len(self.datum) == 1 else pos_data
        all_data.append(pos_data)
        all_cate.append(torch.tensor(pos_types))
        for idx in range(pos_idx + 1, pos_idx + self.num_answers):
            neg_items, neg_types = self.items[idx], self.types[idx]
            neg_datum = [datum.get_data(neg_items, neg_types, self.max_size) for datum in self.datum]
            neg_datum = neg_datum[0] if len(self.datum) == 1 else neg_datum
            all_data.append(neg_datum)
            all_cate.append(torch.tensor(neg_types))

        all_data = torch.stack(all_data, dim=0)  # num_answers x data_shape
        all_cate = torch.stack(all_cate, dim=0)  # num_answers x num_items
        return dict(
            size=self.sizes[pos_idx],
            uidx=self.uidxs[pos_idx],
            data=all_data,
            cate=all_cate,
            index=self.item_index[pos_idx],
        )


class OutfitCompletion(PointwiseOutfit):
    """Pointwise outfit for completion task.

    Alternative implementation of FITB with different output format.

    outfits: [o_1, ..., o_i, ..., o_n], where o_i is the incomplete outfit
    items: [
        i_11, ..., i_1j, ..., i_1s,
        ...,
        i_i1, ..., o_ij, ..., i_is,
        ...
        i_n1, ..., o_nj, ..., i_ns,
    ], where i_ij is the j-th selection for i-th outfit and i_i1 is ground truth.
    """

    def generate(self):
        num_answers = self.data_param.get("num_answers", 4)
        full_sample = self.data_param.get("full_sample", False)
        type_aware = self.data_param.get("type_aware", True)
        ini_data = self.ini_pos
        pos_data = self.pos_generator(ini_data) if self.pos_generator is not None else self.pos_data
        num_questions = len(pos_data)
        pos_uidxs, pos_sizes, pos_items, pos_types = utils.split_tuple(pos_data)
        item_list = utils.get_item_list(pos_data)
        num_types = utils.infer_num_type(pos_data)
        pool_items = []
        pool_types = []
        query_uidxs = []
        query_sizes = []
        query_items = []
        query_types = []
        if not self._multiprocessing:
            # pbar = tqdm(desc="Generating {} tuples".format(self.phase), total=num_questions)
            # self.logger.info("Number of incomplete outfits: {green}%d{reset}", num_questions)
            # self.logger.info("Number of choices: {green}%d{reset}", num_answers)
            self.logger.info("Number of incomplete outfits: %d", num_questions)
            self.logger.info("Number of choices: %d", num_answers)
        for uidx, size, items, types in zip(pos_uidxs, pos_sizes, pos_items, pos_types):
            replaces = range(size) if full_sample else np.random.choice(size, 1)
            for replace_index in replaces:
                sampled_items = items.tolist()
                sampled_types = types.tolist()
                target_type = sampled_types[replace_index]
                target_item = sampled_items[replace_index]
                ans_items = []
                ans_types = []
                for _ in range(num_answers - 1):
                    ans_item = target_item
                    ans_type = target_type
                    while ans_item == target_item:
                        # random sample an item
                        if type_aware:
                            ans_type = target_type
                            ans_item = np.random.choice(item_list[target_type])
                        else:
                            ans_type = np.random.randint(num_types)
                            ans_item = np.random.choice(item_list[ans_type])
                    # replace item and type
                    ans_items.append(ans_item)
                    ans_types.append(ans_type)
                # append incomplete outfit
                query_uidxs.append(uidx)
                query_sizes.append(size - 1)
                query_items.append(sampled_items[:replace_index] + sampled_items[replace_index + 1 :])
                query_types.append(sampled_types[:replace_index] + sampled_types[replace_index + 1 :])
                # append answer items
                pool_items.append([target_item] + ans_items)
                pool_types.append([target_type] + ans_types)
            # if not self._multiprocessing:
            #     pbar.update()
        query_uidxs = np.array(query_uidxs)
        query_sizes = np.array(query_sizes)
        query_types = np.array(query_types)
        query_items = np.array(query_items)
        pool_items = np.array(pool_items).reshape(-1, num_answers)
        pool_types = np.array(pool_types).reshape(-1, num_answers)
        if not self._multiprocessing:
            self.logger.info("Shape of incomplete outfit {}".format(query_items.shape))
            self.logger.info("Shape of pool outfit {}".format(pool_items.shape))
            # pbar.close()
        return query_uidxs, query_sizes, query_items, query_types, pool_items, pool_types

    def _produce(self, worker_id):
        seed = (int(time.time() * 1000) + worker_id) % (2**32)
        np.random.seed(seed)
        while not self._stop_event.is_set():
            if self._queue.full():
                time.sleep(0.01)
                continue
            self._queue.put(self.generate())

    def next(self):
        data_source = self._queue.get() if self._multiprocessing else self.generate()
        self.uidxs, self.sizes, self.query_items, self.query_types, self.pool_items, self.pool_types = data_source
        self.logger.info("Outfit completion tuples ready for next epoch (%s).", self.phase)

    def __getitem__(self, n):
        # get pool data
        pool_items = self.pool_items[n]
        pool_types = self.pool_types[n]
        pool_data = [datum.get_data(pool_items, pool_types) for datum in self.datum]
        pool_data = pool_data[0] if len(self.datum) == 1 else pool_data
        pool_cate = torch.tensor(pool_types)
        # m x n x data_shape
        items, types = self.query_items[n], self.query_types[n]
        data = [datum.get_data(items, types, self.max_size - 1) for datum in self.datum]
        data = data[0] if len(self.datum) == 1 else data
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size - 1)),
            cate=cate,
            pool_data=pool_data,
            pool_cate=pool_cate,
            pool_name=",".join(self.datum[0].get_key(pool_items, pool_types)),
        )


class Retrieval(PointwiseOutfit):
    """Pointwise dataset for retrieval task."""

    def process(self):
        num_outfits = len(self.pos_data)
        num_negatives = len(self.neg_data) // len(self.pos_data)
        num_retrieval = num_negatives + 1
        self.logger.info("Number of retrieval queries: %s", num_outfits)
        self.logger.info("Number of retrieval items: %s", num_retrieval)
        pos_data = self.pos_data.reshape((num_outfits, 1, -1))
        neg_data = self.neg_data.reshape((num_outfits, num_negatives, -1))
        # num_questions x num_answers
        outfits = np.concatenate((pos_data, neg_data), axis=1).reshape((num_outfits * num_retrieval, -1))
        pos_label = np.ones((num_outfits, 1), dtype=np.int64)
        neg_label = np.zeros((num_outfits, num_negatives), dtype=np.int64)
        self.labels = np.hstack((pos_label, neg_label)).reshape((-1, 1))
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(outfits)
        # find the index for answer item
        mask = neg_data[:, 0, :] == pos_data[:, 0, :]
        # the index of answer item
        index_mask = ~utils.split_tuple(mask)[2]
        assert index_mask.sum() == num_outfits
        _, item_index = np.where(index_mask)
        self.item_index = item_index.repeat(num_retrieval)
        assert len(self.item_index) == len(self.labels)

    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        # m x n x data_shape
        data = [datum.get_data(items, types, self.max_size) for datum in self.datum]
        data = data[0] if len(self.datum) == 1 else data
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            label=self.labels[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size)),
            cate=cate,
            index=self.item_index[n],
        )


class OutlierOutfit(PointwiseOutfit):
    """Generate outfits with outlier items for detection task."""

    def generate(self):
        type_aware = self.data_param.get("type_aware", True)

        pos_data = self.ini_pos if self.pos_generator is None else self.pos_generator(self.ini_pos)

        uidxs, sizes, pos_items, pos_types = utils.split_tuple(pos_data)
        item_list = utils.get_item_list(pos_data)
        num_types = utils.infer_num_type(pos_data)

        pos_set = set(map(tuple, pos_items))
        r_items = []
        r_types = []
        r_uidxs = []
        r_yidxs = []
        r_sizes = []
        for uidx, size, items, types in zip(uidxs, sizes, pos_items, pos_types):
            # replace each item
            for ridx in range(size):
                sampled_types = types.copy()
                sampled_items = items.copy()
                while tuple(sampled_items) in pos_set:
                    target_type = sampled_types[ridx]
                    target_item = sampled_items[ridx]
                    # random sample an item
                    sampled_item = target_item
                    while sampled_item == target_item:
                        if type_aware:
                            sampled_type = target_type
                            sampled_item = np.random.choice(item_list[target_type])
                        else:
                            sampled_type = np.random.randint(num_types)
                            sampled_item = np.random.choice(item_list[sampled_type])
                    # replace item and type
                    sampled_items[ridx] = sampled_item
                    sampled_types[ridx] = sampled_type
                r_sizes.append(size)
                r_uidxs.append(uidx)
                r_yidxs.append(ridx)
                r_items.append(sampled_items)
                r_types.append(sampled_types)
        r_items = np.array(r_items)
        r_types = np.array(r_types)
        r_sizes = np.array(r_sizes)
        return np.array(r_uidxs), r_sizes, r_items, r_types, np.array(r_yidxs)

    def _produce(self, worker_id):
        seed = (int(time.time() * 1000) + worker_id) % (2**32)
        np.random.seed(seed)
        while not self._stop_event.is_set():
            if self._queue.full():
                time.sleep(0.01)
                continue
            self._queue.put(self.generate())

    def next(self):
        if self._multiprocessing:
            data_source = self._queue.get()
            self.uidxs, self.sizes, self.items, self.types, self.y = data_source
            self.logger.info("Outlier outfit tuples ready for next epoch (%s).", self.phase)
            self.max_size = np.max(self.sizes)
        else:
            if self.pos_generator is not None:
                self.logger.info(f"Generating {self.phase} positive tuples.")
                self.pos_data = self.pos_generator(self.ini_pos)
                self.logger.info("Positive tuples shape: {}".format(self.pos_data.shape))
                self.num_type = utils.infer_num_type(self.pos_data)
                self.max_size = utils.infer_max_size(self.pos_data)
                self.sections = [1, 1, self.max_size, self.max_size]
            if self.neg_generator is not None:
                self.logger.info(f"Generating {self.phase} negative tuples.")
                self.neg_data = self.neg_generator(self.pos_data)
                self.logger.info("Negative tuples shape: {}".format(self.neg_data.shape))
            self.process()

    def consume(self):
        self.uidxs, self.sizes, self.items, self.types, self.y = self._queue.get()

    def process(self):
        ratio = len(self.neg_data) // len(self.pos_data)
        pos_data = self.pos_data.repeat(ratio, axis=0)
        neg_data = self.neg_data
        pos_items = utils.split_tuple(pos_data)[2]
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(neg_data)
        x, y = np.where(pos_items != self.items)
        assert len(x) == len(self.items), "Only support replacing one item currently"
        self.y = y

    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        # m x n x data_shape
        data = [datum.get_data(items, types, self.max_size) for datum in self.datum]
        data = data[0] if len(self.datum) == 1 else data
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size)),
            cate=cate,
            y=self.y[n],
            label=1,  # fake label
        )


class AddonOutlierOutfit(PointwiseOutfit):
    """Generate outfits with additional outlier items."""

    def process(self):
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(self.neg_data)
        self.y = np.zeros(len(self.uidxs), dtype=np.int64)

    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        # m x n x data_shape
        data = [datum.get_data(items, types, self.max_size) for datum in self.datum]
        data = data[0] if len(self.datum) == 1 else data
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size)),
            cate=cate,
            y=0,
            y_true=0,
        )


class SequenceOutfit(PositiveOutfit):
    """Sequence-based outfit dataset."""

    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        data = self.datum[0].get_data(items, types, self.max_size)
        data = torch.stack(data, dim=0)
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            label=self.labels[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size)),
            cate=cate,
        )


class ItemTriplet(BaseOutfitData):
    """Triplet dataset for item-level learning.

    Returns a triplet (anchor, positive, negative) where
    positive and negative are from the same category.
    """

    def __init__(
        self,
        datum: List[Datum],
        pos_data: np.ndarray = None,
        neg_data: np.ndarray = None,
        phase="train",
        data_param: dict = None,
        pos_mode: str = None,
        neg_mode: str = None,
        pos_param: dict = None,
        neg_param: dict = None,
        **kwargs,
    ):
        super().__init__(datum, pos_data, neg_data, phase, data_param, pos_mode, neg_mode, pos_param, neg_param)
        self.logger.info("Building conditions for {} data.".format(phase))
        # build conditions for type-pair
        indx, indy = np.triu_indices(self.num_type, k=1)
        anc_type = np.hstack((indx, indy))
        cmp_type = np.hstack((indy, indx))
        conditions = dict()
        for i, j in zip(anc_type, cmp_type):
            conditions[(i, j)] = len(conditions)
        self.conditions = conditions

    def __getitem__(self, n):
        triplet = []
        for datum in self.datum:
            anc_data = datum.get_item(item_id=self.anc_idx[n], item_type=self.anc_type[n])
            pos_data = datum.get_item(item_id=self.pos_idx[n], item_type=self.cmp_type[n])
            neg_data = datum.get_item(item_id=self.neg_idx[n], item_type=self.cmp_type[n])
            triplet.append((anc_data, pos_data, neg_data))
        triplet = triplet[0] if len(self.datum) == 1 else triplet
        # get triplet
        # conditions are the same for triplet since:
        # 1. pos and neg are from the same category
        # 2. only anc->pos and anc->neg pairs are considered
        if (self.anc_type[n], self.cmp_type[n]) not in self.conditions:
            condition = 0
        else:
            condition = self.conditions[(self.anc_type[n], self.cmp_type[n])]
        types = torch.LongTensor([self.anc_type[n], self.cmp_type[n]])
        return dict(data=triplet, condition=condition, label=-1, types=types)

    def __len__(self):
        return len(self.anc_idx)

    def process(self):
        # split tuples
        _, _, pos_items, pos_types = utils.split_tuple(self.pos_data)
        _, _, neg_items, neg_types = utils.split_tuple(self.neg_data)
        # check tuples
        assert len(pos_items) == len(neg_items)
        assert (pos_types == neg_types).all()
        indx, indy = np.triu_indices(self.max_size, k=1)
        i = np.hstack((indx, indy))
        j = np.hstack((indy, indx))
        # triplet (xi, xj, xk) where xj and xk are from the same category
        anc_item = pos_items[:, i].flatten()
        pos_item = pos_items[:, j].flatten()
        neg_item = neg_items[:, j].flatten()
        anc_type = pos_types[:, i].flatten()
        cmp_type = pos_types[:, j].flatten()
        # check validation
        valid_items = (anc_type != -1) * (cmp_type != -1)
        valid_pairs = (anc_item != pos_item) * (anc_item != neg_item) * (pos_item != neg_item)
        valid = (valid_items * valid_pairs).nonzero()
        self.anc_idx = anc_item[valid]
        self.pos_idx = pos_item[valid]
        self.neg_idx = neg_item[valid]
        self.anc_type = anc_type[valid]
        self.cmp_type = cmp_type[valid]


class MultiTaskDataset(BaseOutfitData):
    def __init__(
        self,
        datum: List[Datum],
        pos_data: np.ndarray,
        neg_data: np.ndarray = None,
        phase="train",
        data_param: dict = None,
        pos_mode: str = None,
        neg_mode: str = None,
        pos_param: dict = None,
        neg_param: dict = None,
        **kwargs,
    ):

        self._num_threads = 0
        self._multiprocessing = False

        self.outlier = OutlierOutfit(
            datum,
            pos_data=pos_data,
            neg_data=neg_data,
            phase=phase,
            pos_mode=None,
            neg_mode="RandomReplace",
            neg_param={"ratio": 1, "type_aware": True, "num_replace": 1},
        )

        self.outfits = NPairOutfit(
            datum,
            pos_data=pos_data,
            neg_data=neg_data,
            phase=phase,
            data_param={"mp_workers": 4},
            pos_mode=None,
            neg_mode="RandomMix",
            neg_param={"ratio": 32, "type_aware": True},
        )
        self.completion = OutfitCompletion(
            datum,
            pos_data=pos_data,
            neg_data=neg_data,
            phase=phase,
            data_param={"num_answers": 32, "type_aware": True, "mp_workers": 4},
            pos_mode=None,
            neg_mode=None,
            neg_param=None,
        )
        self.group_fitb = GroupFITB(
            datum,
            pos_data=pos_data,
            neg_data=neg_data,
            phase=phase,
            data_param={"mp_workers": 4},
            pos_mode="Fix",
            neg_mode="FITB",
            neg_param={"ratio": 31, "type_aware": True},
        )
        self.next()
        assert len(self.outfits) == len(self.completion) == len(self.outlier)

    def next(self):
        self.outfits.next()
        self.completion.next()
        self.outlier.next()
        self.group_fitb.next()

    def __getitem__(self, idx):
        outfits = self.outfits[idx]
        completion = self.completion[idx]
        outiler = self.outlier[idx]
        group_fitb = self.group_fitb[idx]
        return dict(
            uidx=outfits["uidx"],
            data_r=outfits["data"],
            cate_r=outfits["cate"],
            data_d=outiler["data"],
            cate_d=outiler["cate"],
            y_d=outiler["y"],
            data_c=completion["data"],
            cate_c=completion["cate"],
            data_p=completion["pool_data"],
            cate_p=completion["pool_cate"],
            data_g=group_fitb["data"],
            cate_g=group_fitb["cate"],
            index_g=group_fitb["index"],
        )

    def __len__(self):
        return len(self.outfits)


def getOutfitData(
    datum: List[Datum], param: OutfitDataParam, pos_data: np.ndarray, neg_data: np.ndarray = None, phase="train"
) -> BaseOutfitData:
    """Factory function for outfit dataset creation.

    Args:
        datum (List[Datum]): List of feature readers.
        param (DataBuilderParam): Dataset parameters.
        pos_data (np.ndarray): Positive tuples.
        neg_data (np.ndarray, optional): Negative tuples. Defaults to None.
        phase (str): Phase information for logging.

    Returns:
        BaseOutfitData: Dataset instance.
    """
    return _dataset_registry[param.data_mode](datum, pos_data, neg_data, phase, **param)
