from dataclasses import dataclass
from typing import Optional

import attr

from .base import Param


@attr.s
class DataReaderParam(Param):
    """Parameter class for :class:`outfit_datasets.reader.DataReader`.

    It supports an alternative way to initialize a DataReader instance::

        param = DataReaderParam(reader="ImageLMDB", path="data", data_transform="identity")
        reader = getReader(param=param)

    """

    reader = attr.ib(default="Dummy")
    path = attr.ib(default=None)
    data_transform = attr.ib(default=None)


@dataclass
class DataReaderConfig:
    """
    Parameter class for :class:`outfit_datasets.reader.DataReader`.

    It supports an alternative way to initialize a DataReader instance:

        param = DataReaderConfig(reader="ImageLMDB", path="data", data_transform="identity")
        reader = getReader(param=param)
    """

    reader: str = "Dummy"
    path: Optional[str] = None
    data_transform: Optional[str] = None
