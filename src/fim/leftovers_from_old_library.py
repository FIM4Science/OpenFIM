import _imp  # Just the builtin component, NOT the full Python module
import sys
from copy import copy
from pathlib import Path
from typing import Optional

import yaml


try:
    import _frozen_importlib as _bootstrap
except ImportError:
    from . import _bootstrap

    _bootstrap._setup(sys, _imp)
else:
    # importlib._bootstrap is the built-in import, ensure we don't create
    # a second copy of the module.
    _bootstrap.__name__ = "importlib._bootstrap"
    _bootstrap.__package__ = "importlib"
    try:
        _bootstrap.__file__ = __file__.replace("__init__.py", "_bootstrap.py")
    except NameError:
        # __file__ is not guaranteed to be defined, e.g. if this code gets
        # frozen by a tool like cx_Freeze.
        pass
    sys.modules["importlib._bootstrap"] = _bootstrap


def save_in_yaml(cfg: dict, path: Path, filename: Optional[str] = None) -> None:
    "Save cfg (dict) as yaml in path with filename."

    # path to directory + filename provided
    if filename is not None:
        path.mkdir(parents=True, exist_ok=True)
        if not filename.endswith(".yaml"):
            filename = filename + ".yaml"
        cfg_path = Path(path) / filename

    # path to file directly (still create dirs if necessary)
    else:
        path = path.parents[0]
        path.mkdir(parents=True, exist_ok=True)
        cfg_path = path

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)


def create_class_instance(kwargs, *args) -> object:
    """
    Create and return an instance of a class specified in kwargs["class_path"].
    """

    if isinstance(kwargs, dict):
        clazz, kwargs = get_class(kwargs)

        if kwargs is None:
            instance = clazz(*args)
        else:
            instance = clazz(*args, **kwargs)

        return instance

    else:
        return kwargs


def get_class(kwargs: dict, class_key: Optional[str] = "class_path") -> tuple:
    """
    Get class from class_path specified in kwargs.
    """
    # jax evaluates pop twice
    kwargs_copy = copy(kwargs)

    class_path = kwargs_copy.pop(class_key)

    module_name, class_name = class_path.rsplit(".", 1)

    module = import_module(module_name)
    clazz = getattr(module, class_name)

    return clazz, kwargs_copy


def import_module(name, package=None):
    """Import a module.

    The 'package' argument is required when performing a relative import. It
    specifies the package to use as the anchor point from which to resolve the
    relative import to an absolute import.

    """
    level = 0
    if name.startswith("."):
        if not package:
            msg = "the 'package' argument is required to perform a relative " "import for {!r}"
            raise TypeError(msg.format(name))
        for character in name:
            if character != ".":
                break
            level += 1
    return _bootstrap._gcd_import(name[level:], package, level)


def load_config(path: str) -> dict:
    """
    Loads configuration from yaml file into dict.
    """
    with open(path, "r") as f:
        params = yaml.full_load(f)
    return params
