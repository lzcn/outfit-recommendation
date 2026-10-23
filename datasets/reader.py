import logging
import os
import pickle
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Union

import lmdb
import numpy as np
import PIL
import six
import torch
from torchvision import transforms

from datasets.param import DataReaderParam

LOGGER = logging.getLogger(__name__)
_LMDB_ENVS = {}
_LMDB_PID = os.getpid()


def _load_pkl_data(path):
    path = os.path.expanduser(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def _open_lmdb_env(path):
    global _LMDB_PID
    if _LMDB_PID != os.getpid():
        for env in _LMDB_ENVS.values():
            env.close()
        _LMDB_ENVS.clear()
        _LMDB_PID = os.getpid()
    path = os.path.realpath(os.path.expanduser(path))
    if path not in _LMDB_ENVS:
        _LMDB_ENVS[path] = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    return _LMDB_ENVS[path]


class DataReader(metaclass=ABCMeta):
    """Callable reader for different data types.

    Args:
        path (str): Data path, format depends on reader type.
        data_transform (Callable, optional): Data transform function. Defaults to identity.
        default (optional): Value returned for a missing key. If None, raise KeyError.
    """

    def __init__(self, path: str, data_transform: Callable = None, default=None):
        self.default = default
        self.path = path
        self.data_transform = data_transform

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_env" in state:
            state["_env"] = None
        return state

    @abstractmethod
    def load(self, key) -> Any:
        """Load raw data before transformation.

        Args:
            key (str): Data identifier, format varies by reader type.

        Returns:
            Any: Raw data before transform.
        """
        raise NotImplementedError

    def __call__(self, key):
        data = self.load(key)
        if data is None:
            if self.default is None:
                raise KeyError(f"Missing data key {key!r} in {self.path}")
            return self.default
        if self.data_transform is None:
            return data
        return self.data_transform(data)


class TensorLMDBReader(DataReader):
    """Reader for tensor data with LMDB backend.

    Data is stored in float32 format.
    """

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)
        self.path = path
        self._env = None

    def load(self, key) -> torch.Tensor:
        """Load tensor from LMDB with key-value storage.

        Args:
            key (str): Data key.

        Returns:
            torch.Tensor: Output tensor.
        """
        self._env = _open_lmdb_env(self.path)
        with self._env.begin(write=False) as txn:
            buf = txn.get(key.encode())
            if buf is None:
                return None
        feature = np.frombuffer(buf, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(feature.copy()).view(-1)


class ImagePILReader(DataReader):
    """Reader for images using PIL.

    Args:
        path (str): Root directory for images.
        data_transform (Callable, optional): Data transform. Defaults to identity.
    """

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)

    def load(self, name: str) -> PIL.Image.Image:
        """Load PIL Image from file.

        Args:
            name (str): Relative image path under self.path.

        Returns:
            PIL.Image.Image: Loaded image before transform.
        """
        # read from raw image
        path = os.path.join(self.path, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            img = PIL.Image.open(f).convert("RGB")
        return img


class ImageLMDBReader(DataReader):
    """Reader for images with LMDB backend.

    Args:
        path (str): Directory for LMDB data.
    """

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)
        self._env = None

    def load(self, name: str) -> PIL.Image.Image:
        """Load image from LMDB using key-value storage.

        Args:
            name (str): Data key.

        Returns:
            PIL.Image.Image: Loaded image before transform.
        """
        self._env = _open_lmdb_env(self.path)
        with self._env.begin(write=False) as txn:
            imgbuf = txn.get(name.encode())
            if imgbuf is None:
                return None
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = PIL.Image.open(buf).convert("RGB")
        return img


class TensorPKLReader(DataReader):
    """Reader for tensor data."""

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)
        self._data = _load_pkl_data(path)

    def load(self, name) -> torch.Tensor:
        if name not in self._data:
            return None
        feature = self._data[name].astype(np.float32)
        return torch.from_numpy(feature.copy())


class DummyReader(DataReader):
    """Dummy data reader for testing.

    Args:
        path (str): Unused path parameter.
        data_transform (Callable, optional): Data transform. Defaults to identity.
    """

    def __init__(self, path, data_transform=None, default=None):
        super().__init__(path, data_transform=data_transform, default=default)
        if default is None:
            self.default = torch.zeros(1)

    def load(self, name: str):
        return None


_Readers = {
    "ImageLMDB": ImageLMDBReader,
    "ImagePIL": ImagePILReader,
    "TensorLMDB": TensorLMDBReader,
    "TensorPKL": TensorPKLReader,
    "Dummy": DummyReader,
}


def _get_transforms(data_transform):
    if isinstance(data_transform, str):
        data_transform = eval(data_transform)
        if isinstance(data_transform, Callable):
            return data_transform
        else:
            return transforms.Compose(data_transform)
    elif isinstance(data_transform, Callable):
        return data_transform
    else:
        return None


def getReader(
    reader: str = None, path: str = None, data_transform: Union[str, Callable] = None, param: DataReaderParam = None
) -> DataReader:
    """Factory for DataReader instances.

    Available reader types:
        - "ImageLMDB": Images with LMDB backend.
        - "ImagePIL": Images with PIL backend.
        - "TensorLMDB": Float32 data with LMDB backend.
        - "TensorPKL": Float32 data saved with pickle.

    Args:
        reader (str): Reader type.
        path (str): Data path.
        data_transform (Union[str, Callable], optional): Transform function or eval string. Defaults to None.
        param (DataReaderParam, optional): Reader parameters. Defaults to None.

    Returns:
        DataReader: Callable DataReader instance.

    Examples:

        .. code-block:: python

            reader = "ImageLMDB"
            path = "/path/to/lmdb"
            data_transform = '''
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            '''
            reader = getReader(reader, path, data_transform)
            data = reader(key)

        In this example, data_transform is defined as a string.
    """
    if param is not None:
        reader = param.reader
        path = param.path
        data_transform = param.data_transform
    if reader not in _Readers:
        raise ValueError("reader must be on of {}".format("|".join(_Readers.keys())))
    data_transform = _get_transforms(data_transform)
    return _Readers[reader](path, data_transform)
