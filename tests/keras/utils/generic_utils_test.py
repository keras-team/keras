import pytest
from keras.utils.generic_utils import custom_object_scope
from keras import activations
from keras import regularizers


def test_custom_objects_scope():

    def custom_fn():
        pass

    class CustomClass(object):
        pass

    with custom_object_scope({'CustomClass': CustomClass,
                              'custom_fn': custom_fn}):
        act = activations.get('custom_fn')
        assert act == custom_fn
        cl = regularizers.get('CustomClass')
        assert cl.__class__ == CustomClass


if __name__ == '__main__':
    pytest.main([__file__])
