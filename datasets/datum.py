import json
import os
from typing import List, Optional

import torch

from datasets.param import OutfitLoaderParam
from datasets.reader import DataReader, getReader


def load_json(fn: str) -> List[List[str]]:
    """Load item list JSON file.

    The file should contain a list of lists, where each sub-list represents
    all item keys of one category:
        [
            ["top_001", "top_002", "top_003"],    # category 0
            ["bottom_001", "bottom_002"],         # category 1
            ["shoes_001", "shoes_002"]            # category 2
        ]

    Args:
        fn (str): Path to the JSON file.

    Returns:
        List[List[str]]: A nested list of item keys.
    """
    path = os.path.expanduser(fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(c, list) for c in data):
        raise ValueError(f"Expected list of lists in {path}, got {type(data).__name__}")
    return data


class Datum(object):
    r"""Wrapper for a data reader.

    The data reader (datasets.reader.DataReader) must implement:
        reader(key: str) -> torch.Tensor

    Args:
        item_list: item keys indexed by [category][item_id].
                   If None, the key is `str(item_id)`.
        reader:    a callable that returns a tensor given a key.
    """

    def __init__(self, item_list: Optional[List[List[str]]] = None, reader: DataReader = None) -> None:
        self.item_list = item_list
        self.reader = reader

    def get_key(self, item_ids: List, item_types: List, max_size: int = 0) -> List[str]:
        """Return keys for the data reader.

        Args:
            item_ids (List): item list
            item_types (List): item types
            max_size (int, optional): max size of items. Defaults to 0.

        Returns:
            List[str]: list of item keys
        """
        keys: List[str] = []
        target_size = max(max_size, len(item_ids))
        for item, cate in zip(item_ids, item_types):
            if cate == -1:
                continue
            else:
                key = self.item_list[cate][item] if self.item_list is not None else str(item)
                keys.append(key)
        while len(keys) < target_size:
            keys.append(keys[-1])
        return keys

    def get_item(self, item_id: int, item_type: int) -> torch.Tensor:
        r"""Get the data of single item.

        Args:
            item_id (int): item id
            item_type (int): item type.

        Returns:
            torch.Tensor: item data
        """
        key = self.item_list[item_type][item_id] if self.item_list is not None else str(item_id)
        return self.reader(key)

    def get_data(self, item_id: List, item_type: List, max_size: int = 1) -> torch.Tensor:
        r"""Get the data for an outfit.

        Args:
            item_id (List): item ids
            item_type (List): item types.
            max_size (int, optional): max size. Defaults to 1.

        Returns:
            torch.Tensor: shape of (max_size, *data_shape)
        """
        keys = self.get_key(item_id, item_type, max_size)
        data = torch.stack([self.reader(key) for key in keys], dim=0)
        return data


def getDatum(param: OutfitLoaderParam) -> List[Datum]:
    """Datum factory.

    Args:
        param (OutfitLoaderParam): outfit data loader parameters

    Returns:
        List[Datum]: a list of datum
    """
    datums = []
    if os.path.exists(param.item_list_fn):
        item_list = load_json(param.item_list_fn)
    else:
        item_list = None
    for reader_param in param.readers:
        reader = getReader(param=reader_param)
        datums.append(Datum(item_list, reader))
    return datums


def build_datums(item_list_fn: Optional[str], readers: List[dict]) -> List[Datum]:
    """Construct Datum objects.

    Each Datum wraps a DataReader and optional item list.
    The item list defines how item IDs map to keys used by the readers.

    Args:
        item_list_fn (Optional[str]): Path to the JSON file containing item lists.
            If None or the file doesn't exist, item_list will be None.
        readers (List[dict]): List of reader parameter dicts. Each will be passed
            to `getReader(param=reader_param)`.

    Returns:
        List[Datum]: Initialized Datum instances.
    """
    # Load item list if available
    item_list = None
    if item_list_fn and os.path.exists(item_list_fn):
        item_list = load_json(item_list_fn)

    # Create one Datum per reader
    datums: List[Datum] = []
    for reader_param in readers:
        reader = getReader(**reader_param)
        datums.append(Datum(item_list=item_list, reader=reader))

    return datums
