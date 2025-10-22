import logging
import os
from typing import IO, Any

import attr
import yaml

_factory = {}

_FACTORY_ATTR_NAME = "factory"
LOGGER = logging.getLogger(__name__)


def _configurable_filter(attribute: attr.Attribute, value: Any) -> bool:
    if attribute.name == _FACTORY_ATTR_NAME and value is None:
        return False
    return attribute.init


def _printable_filter(attribute: attr.Attribute, value: Any) -> bool:
    if attribute.name == _FACTORY_ATTR_NAME and value is None:
        return False
    return attribute.repr


def _format_display(opt: Any, num=1, symbol="  "):
    indent = symbol * num
    if isinstance(opt, dict):
        repr_list = ["{}: {}".format(k, _format_display(v, num + 1)) for k, v in opt.items()]
        lsign = "{"
        rsign = "}"
        if sum(map(len, repr_list)) < 10:
            string = lsign + ", ".join(repr_list) + rsign
        else:
            string = lsign + "\n"
            for repr in repr_list:
                string += "{}{},\n".format(indent, repr)
            string += symbol * (num - 1) + rsign
    elif isinstance(opt, list):
        repr_list = [_format_display(v, num + 1) for v in opt]
        lsign = "["
        rsign = "]"
        if sum(map(len, repr_list)) < 10:
            string = lsign + ", ".join(repr_list) + rsign
        else:
            string = lsign + "\n"
            for repr in repr_list:
                string += "{}{},\n".format(indent, repr)
            string += symbol * (num - 1) + rsign
    else:
        string = str(opt)
    return string


class YAMLoader(yaml.SafeLoader):
    """YAML Loader with include constructor."""

    def __init__(self, stream: IO) -> None:

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


@attr.s
class Param(object):
    """Base configuration class with hierarchical structure.

    Subclasses are automatically registered by class name.
    Built using the attrs package. For attributes that are Param
    subclasses, use to_param as converter::

        attr.ib(converter=to_param)

    See OutfitDataParam and OutfitLoaderParam for examples.

    Note:
        Only attributes with repr=True are displayed.
        Attributes with init=True are configurable.

    Attributes:
        factory (str): Class name for initialization.
    """

    # class name for param
    factory: str = attr.ib(default=None, kw_only=True)

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__ in _factory:
            LOGGER.warning("%s is already used", cls.__name__)
        _factory[cls.__name__] = cls

    def __str__(self):
        d = attr.asdict(self, filter=_printable_filter)
        return self.__class__.__name__ + ": " + _format_display(d)

    def asdict(self):
        """Return configurable attributes with init=True."""
        return attr.asdict(self, filter=_configurable_filter)

    def serialize(self):
        """Serialize configurable settings to YAML format."""
        return yaml.dump(self.asdict())

    @classmethod
    def evolve(cls, inst=None, **changes):
        """Create new instance based on existing instance with changes.

        Calling evolve() without inst creates a default instance.

        Examples::

                param = Param.evolve()  # default instance
                param = Param.evolve(x=1)  # default with partial keywords
                param = Param.evolve(param, x=2)  # apply changes to param

        Args:
            inst (Param, optional): Instance of class. Defaults to None.
            changes: Keyword changes.

        Returns:
            Param: New instance.
        """
        inst = cls() if inst is None else inst
        return attr.evolve(inst, **changes)

    @classmethod
    def new(cls, value=None):
        """Create new instance.

        If value is None, returns instance with default settings.

        Args:
            value: Configuration dict or existing instance.

        Returns:
            New instance.
        """
        if value is None:
            return cls()
        elif isinstance(value, cls):
            return value
        return cls(**value)

    @classmethod
    def from_yaml(cls, fn):
        """Create new instance from YAML file.

        Args:
            fn (str): Filename.

        Returns:
            Param: New instance.
        """
        with open(fn, "r") as fn:
            kwargs = yaml.load(fn, Loader=YAMLoader)
            param = cls(**kwargs)
        return param

    @classmethod
    def from_dict(cls, value=None):
        """Create new instance from dictionary.

        Args:
            value (Any, optional): If None, returns None. Defaults to None.

        Used as converter for attr.ib:
            - Default instance: attr.ib(factory=dict, converter=cls.from_dict)
            - No default: attr.ib(default=None, converter=cls.from_dict)

        Returns:
            New instance or None.
        """
        if value is None or isinstance(value, cls):
            return value
        return cls(**value)


def to_param(value) -> Param:
    """Convert dictionary to Param instance if possible.

    If value is dict with "factory" key matching a Param subclass,
    returns an instance. Otherwise returns value unchanged.

    Args:
        value (Any): Configuration dict or other value.

    Returns:
        Param instance or original value.

    Examples::

        value = {"factory": "OptimParam", ...}
        param = to_param(value)
        assert isinstance(param, OptimParam)
    """
    class_name = value.get(_FACTORY_ATTR_NAME)
    if isinstance(value, dict) and class_name in _factory:
        return _factory[class_name](**value)
    return value


def converter(value) -> Param:
    """Convert value to Param instance using factory attribute.

    If value has factory attribute matching a Param subclass,
    returns an instance. Otherwise returns value unchanged.

    Args:
        value (Any): Configuration object with factory attribute.

    Returns:
        Param instance or original value.

    Examples::

        value = {"factory": "OptimParam", ...}
        param = converter(value)
        assert isinstance(param, OptimParam)
    """
    if value is None:
        return None
    class_name = value.factory
    if class_name in _factory:
        return _factory[class_name](**value)
    return value
