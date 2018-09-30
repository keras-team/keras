import pytest
from keras.utils.test_utils import layer_test
from keras import layers
from keras import backend as K


def test_leaky_relu():
    for alpha in [0., .5, -1.]:
        layer_test(layers.LeakyReLU, kwargs={'alpha': alpha},
                   input_shape=(2, 3, 4))


def test_prelu():
    layer_test(layers.PReLU, kwargs={},
               input_shape=(2, 3, 4))


def test_prelu_share():
    layer_test(layers.PReLU, kwargs={'shared_axes': 1},
               input_shape=(2, 3, 4))


def test_elu():
    for alpha in [0., .5, -1.]:
        layer_test(layers.ELU, kwargs={'alpha': alpha},
                   input_shape=(2, 3, 4))


def test_thresholded_relu():
    layer_test(layers.ThresholdedReLU, kwargs={'theta': 0.5},
               input_shape=(2, 3, 4))


def test_softmax():
    for axis in [1, -1]:
        layer_test(layers.Softmax, kwargs={'axis': axis},
                   input_shape=(2, 3, 4))


def test_relu():
    layer_test(layers.ReLU,
               kwargs={'max_value': 10,
                       'negative_slope': 0.2,
                       'threshold': 3.0},
               input_shape=(2, 3, 4))
    layer_test(layers.ReLU,
               kwargs={'max_value': 6},
               input_shape=(2, 3, 4))
    layer_test(layers.ReLU,
               kwargs={'negative_slope': 0.2},
               input_shape=(2, 3, 4))

    # max_value of ReLU layer cannot be negative value
    with pytest.raises(ValueError):
        layer_test(layers.ReLU, kwargs={'max_value': -2.0},
                   input_shape=(2, 3, 4))

    # negative_slope of ReLU layer cannot be negative value
    with pytest.raises(ValueError):
        layer_test(layers.ReLU, kwargs={'negative_slope': -2.0},
                   input_shape=(2, 3, 4))


@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='TF-specific implementation.')
def test_relu_tf_ops():
    inputs = layers.Input((3,))
    # Test that `relu` op gets used.
    outputs = layers.ReLU()(inputs)
    assert outputs.op.name.lower().endswith('/relu')
    # Test that `leakyrelu` op gets used.
    outputs = layers.ReLU(negative_slope=0.2)(inputs)
    assert outputs.op.name.lower().endswith('/leakyrelu')
    # Test that `relu6` op gets used.
    outputs = layers.ReLU(max_value=6)(inputs)
    assert outputs.op.name.lower().endswith('/relu6')


if __name__ == '__main__':
    pytest.main([__file__])
