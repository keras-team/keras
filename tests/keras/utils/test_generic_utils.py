import pytest
import keras
from keras import backend as K
from keras.utils.generic_utils import custom_object_scope, get_custom_objects, get_from_module


def test_custom_object_scope_adds_objects():
    get_custom_objects().clear()
    assert (len(get_custom_objects()) == 0)
    with custom_object_scope({"Test1": object, "Test2": object}, {"Test3": object}):
        assert (len(get_custom_objects()) == 3)
    assert (len(get_custom_objects()) == 0)


class CustomObject(object):
    def __init__(self):
        pass


def test_get_from_module_uses_custom_object():
    get_custom_objects().clear()
    assert (get_from_module("CustomObject", globals(), "test_generic_utils") == CustomObject)
    with pytest.raises(ValueError):
        get_from_module("TestObject", globals(), "test_generic_utils")
    with custom_object_scope({"TestObject": CustomObject}):
        assert (get_from_module("TestObject", globals(), "test_generic_utils") == CustomObject)


if __name__ == '__main__':
    pytest.main([__file__])
