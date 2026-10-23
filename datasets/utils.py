"""
Outfit Tuples Utilities
=======================

Tuple format (headerless CSV):
    [user_id, size, item_1, ..., item_n, type_1, ..., type_n]

    - user_id : int, user index
    - size    : int, number of valid items (<= n)
    - item    : int, index into `item_list` (padded -1)
    - type    : int, category index (padded -1)
    - n       : max outfit length, inferred as (cols - 2) // 2

Legacy format (no `size` column) is auto-converted.

Note on `item_list`
-------------------
`item` is only an index; the real image/feature key is stored in `item_list`.
Use the category-specific lists in `items.json` to resolve indices to keys.
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np

import utils

NONE_TYPE = -1  # padding value used for both item indices and types
logger = utils.logger.get_logger(__name__)


def _upgrade_legacy(arr: np.ndarray) -> np.ndarray:
    """If legacy format (no size col), insert size inferred from (-1)-padding."""
    if arr.ndim != 2:
        raise ValueError(f"tuples must be 2D, got {arr.shape}")
    n_cols = arr.shape[1]
    # new: cols = 2 + 2*n (even). old: 1 + 2*n (odd)
    if n_cols % 2 == 0:
        return arr
    # legacy format, insert size col
    users = arr[:, :1]
    items, types = np.split(arr[:, 1:], 2, axis=1)
    sizes = (items != NONE_TYPE).sum(axis=1, keepdims=True)
    out = np.hstack([users, sizes, items, types])
    logger.info("Upgraded legacy tuples to [user,size,items...,types...] shape=%s", out.shape)
    return out


def load_csv(fn, num_skip=0, converter=None):
    """Load data in csv format.

    Args:
        fn (str): file name
        num_skip (int, optional): number of lines to skip. Defaults to 0.
        converter (Callable, optional): function to convert each element. Defaults to None.

    Returns:
        List: data
    """
    fn = os.path.expanduser(fn)
    with open(fn, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for _ in range(num_skip):
            next(reader)
        data = list(reader)
        if converter is not None:
            data = [list(map(converter, line)) for line in data]
    return data


def load_outfit_tuples(path: str) -> np.ndarray:
    """Load tuples into ndarray."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        logger.debug("File %s does not exist", path)
        return None
    logger.debug("Loading tuples: %s", path)
    arr = np.loadtxt(path, delimiter=",", dtype=int, ndmin=2)
    arr = _upgrade_legacy(arr)
    return arr


def split_tuple(tuples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (users, sizes, items, types):
        users: (N,)
        sizes: (N,)
        items: (N, K)  [-1 padded]
        types: (N, K)  [-1 padded]
    """
    k = (tuples.shape[1] - 2) // 2
    users = tuples[:, 0]
    sizes = tuples[:, 1]
    items = tuples[:, 2 : 2 + k]
    types = tuples[:, 2 + k : 2 + 2 * k]
    return users, sizes, items, types


def infer_max_size(tuples: np.ndarray) -> int:
    """Infer the max item size from outfit tuples.

    Args:
        tuples (np.ndarray): fashion outfit tuples

    Returns:
        int: max size
    """
    max_shape = (tuples.shape[1] - 2) // 2
    sizes = split_tuple(tuples)[1]
    max_size = np.max(sizes)
    if max_size != max_shape:
        logger.warning("Tuples is not compact, i.e. the last {} columns are all -1s".format(max_shape - max_size))
    return max_size


def infer_max_shape(tuples: np.ndarray) -> int:
    """Infer the max size from oufit shape.

    Args:
        tuples (np.ndarray): fashion outfit tuples

    Returns:
        int: max size
    """
    max_shape = (tuples.shape[1] - 2) // 2
    return max_shape


def infer_num_type(tuples: np.ndarray) -> int:
    """Infer the number of categories from data

    Args:
        tuples (np.ndarray): fashion outfit tuples

    Returns:
        int: number of categories.
    """
    item_types = set(split_tuple(tuples)[-1].flatten())
    if NONE_TYPE in item_types:
        num_type = len(item_types) - 1
    else:
        num_type = len(item_types)
    return num_type


# def get_item_list(data: np.ndarray) -> List[np.ndarray]:
#     _, _, items, types = split_tuple(data)
#     items = items.ravel()
#     types = types.ravel()
#     # map -1 -> last bucket if present
#     if np.any(types == -1):
#         max_type = types.max()
#         mapped = np.where(types == -1, max_type + 1, types)
#     else:
#         mapped = types
#     num_buckets = mapped.max() + 1
#     # collect unique item_ids per bucket
#     item_list = []
#     for t in range(num_buckets):
#         sel = items[mapped == t]
#         item_list.append(np.unique(sel))
#     return item_list


def get_item_list(data: np.ndarray) -> List[np.ndarray]:
    """Return item list for in each fashion category.

    Append non-fashion category in last.

    Args:
        data (np.ndarray): fashion outfit tuples

    Returns:
        List[np.ndarray]: a list of fashion items
    """
    _, _, item_ids, item_types = split_tuple(data)
    num_list = len(set(item_types.flatten()))
    item_set = [set() for _ in range(num_list)]
    for idxs, types in zip(item_ids, item_types):
        for idx, c in zip(idxs, types):
            item_set[c].add(idx)
    item_list = [np.array(list(s)) for s in item_set]
    return item_list


def rearrange(items, types) -> List[list]:
    new_items, new_types = [], []
    for item_id, item_type in zip(items, types):
        if item_type == NONE_TYPE:
            continue
        new_items.append(item_id)
        new_types.append(item_type)
    while len(new_items) < len(items):
        new_items.append(NONE_TYPE)
        new_types.append(NONE_TYPE)
    return new_items, new_types
