import functools
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import hydra
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from .distributed import rank_zero_only


def get_output_dir() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def instantiate(
    registry: Dict[str, Union[str, Callable]],
    config: Union[str, Dict[str, Any]],
    *args: Any,
    partial: bool = False,
    wrap: Optional[Callable] = None,
    **kwargs: Any
) -> Any:
    """Instantiate class or callable from registry and configuration.

    Dynamically creates an object or partial function based on runtime
    configuration. The registry maps identifiers to classes, functions,
    or module paths for flexible component creation.

    Returns target(*args, **config, **kwargs) or its partial function.
    target = registry[config["_name_"]] if config is dict else registry[config]

    Args:
        registry (Dict[str, Union[str, Callable]]): Maps string identifiers
            to class types, factory functions, or import paths.
        config (Union[str, Dict[str, Any], None]): Configuration for instantiation:
            - String: directly maps to registry key.
            - Dict: must have "_name_" key mapping to registry. Other keys
              are constructor arguments.
            - None: returns None (useful for optional components).
        *args (Any): Positional arguments passed to constructor. Override
            arguments from config.
        partial (bool, optional): If True, returns functools.partial object
            for delayed instantiation. Defaults to False.
        wrap (Optional[Callable], optional): Callable to wrap the target
            before instantiation. Useful for decorators, logging, etc.
            Defaults to None.
        **kwargs (Any): Keyword arguments passed to constructor. Override
            arguments from config.

    Returns:
        Any: Instantiated object, wrapped object, or functools.partial object.

    Note:
        - Use partial=True for delayed instantiation or incomplete parameters.
        - wrap can apply decorators or monitoring tools to the target.
    """
    # Case 1: no config
    if config is None:
        return None
    # Case 2a: string means _name_ was overloaded
    if isinstance(config, str):
        _name_ = None
        _target_ = registry[config]
        config = {}
    # Case 2b: grab the desired callable from name
    else:
        _name_ = config.pop("_name_")
        _target_ = registry[_name_]

    # Retrieve the right constructor automatically based on type
    if isinstance(_target_, str):
        fn = hydra.utils.get_method(path=_target_)
    elif isinstance(_target_, Callable):
        fn = _target_
    else:
        raise NotImplementedError("instantiate target must be string or callable")

    # Instantiate object
    if wrap is not None:
        fn = wrap(fn)
    obj = functools.partial(fn, *args, **config, **kwargs)

    # Restore _name_
    if _name_ is not None:
        config["_name_"] = _name_

    if partial:
        return obj
    else:
        return obj()


def omegaconf_filter_keys_inplace(cfg, fn=None):
    """Remove keys from config where fn(key) returns False.

    Modifies config in-place recursively.

    Args:
        cfg: OmegaConf configuration object.
        fn: Filter function. If None, keeps all keys. Defaults to None.
    """
    if fn is None:
        fn = lambda k: True

    if isinstance(cfg, (dict, DictConfig)):
        # Remove keys that don't pass the filter (in-place deletion)
        keys_to_remove = [k for k in cfg if not fn(k)]
        for k in keys_to_remove:
            del cfg[k]
        # Recursively filter nested configs (modifies nested objects in-place)
        for v in cfg.values():
            omegaconf_filter_keys_inplace(v, fn)

    elif isinstance(cfg, (list, ListConfig)):
        # Recursively filter items in lists (items are modified in-place)
        for item in cfg:
            omegaconf_filter_keys_inplace(item, fn)


def process_config(config: DictConfig) -> DictConfig:
    """Process config: register resolvers, filter internal keys, enable modifications.

    Args:
        config: Hydra configuration object.

    Returns:
        Processed configuration.
    """

    # Register custom OmegaConf resolvers
    resolvers = {
        "eval": eval,
        "len": len,
        "add": lambda a, b: a + b,
    }
    for name, func in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, func)

    # Remove internal keys (those starting with __)
    omegaconf_filter_keys_inplace(config, lambda k: not k.startswith("__"))

    # Allow adding new keys dynamically
    OmegaConf.set_struct(config, False)

    return config


@rank_zero_only
def print_config(config: DictConfig, resolve: bool = True, save_cfg=True) -> None:
    """Print configuration tree using Rich library.

    Args:
        config (DictConfig): Hydra configuration.
        resolve (bool, optional): Resolve reference fields. Defaults to True.
        save_cfg (bool, optional): Save tree to file. Defaults to True.
    """
    base_style = "#d4d4d4"
    field_style = "#569cd6"
    tree_style = base_style
    guide_style = base_style

    tree = rich.tree.Tree("CONFIG", style=tree_style, guide_style=guide_style)
    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=field_style, guide_style=guide_style)

        config_section = config.get(field)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        else:
            branch_content = str(config_section)
        # inherit background color from the terminal
        branch.add(rich.syntax.Syntax(branch_content, "yaml", background_color="default"))

    rich.print(tree)

    if save_cfg:
        file = get_output_dir() / "config_tree.txt"
        with open(file, "w") as fp:
            rich.print(tree, file=fp)
