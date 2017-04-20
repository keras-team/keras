from inspect import isfunction, isclass, getmembers, ismodule, ismethod, signature, getmodule
import importlib
from itertools import compress
import pytest

from keras.utils.test_utils import keras_test

modules = ['keras.layers', 'keras.models', 'keras', 'keras.backend.tensorflow_backend']
accepted_name = ["from_config"]
accepted_module = ["keras.legacy.layers", "keras.utils.generic_utils"]


def handle_class(name, member):
    if name in accepted_name or member.__module__ in accepted_module:
        return
    assert member.__doc__ is not None, "class doesn't have any documentation {} {}".format(name, member.__module__, getmodule(member).__file__)
    for n, met in getmembers(member):
        if ismethod(met):
            handle_method(n, met)


def handle_function(name, member):
    if name in accepted_name or member.__module__ in accepted_module:
        return
    doc = member.__doc__
    assert doc is not None, "{} function doesn't have any documentation {} {}".format(name, member.__module__, getmodule(member).__file__)
    args = list(signature(member).parameters.keys())
    args_not_in_doc = [not arg in doc for arg in args]
    assert not any(args_not_in_doc), "{} {} arguments are not present in documentation ".format(name, list(compress(args, args_not_in_doc)),
                                                                                                member.__module__)


def handle_method(name, member):
    if name in accepted_name or member.__module__ in accepted_module:
        return
    handle_function(name, member)


def handle_module(mod):
    for name, mem in getmembers(mod):
        if isclass(mem):
            handle_class(name, mem)
        elif isfunction(mem):
            handle_function(name, mem)
        elif 'keras' in name and ismodule(mem):
            # Only test keras' modules
            handle_module(mem)


@keras_test
def test_doc():
    for module in modules:
        mod = importlib.import_module(module)
        handle_module(mod)


if __name__ == '__main__':
    pytest.main([__file__])
