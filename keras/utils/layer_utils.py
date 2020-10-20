"""Utilities related to layer/model functionality.
"""
from .conv_utils import convert_kernel
from .. import backend as K
import numpy as np


from tensorflow.keras.utils import get_source_inputs
from tensorflow.python.keras.utils.layer_utils import print_summary


def count_params(weights):
    """Count the total number of scalars composing the weights.

    # Arguments
        weights: An iterable containing the weights on which to compute params

    # Returns
        The total number of scalars composing the weights
    """
    weight_ids = set()
    total = 0
    for w in weights:
        if id(w) not in weight_ids:
            weight_ids.add(id(w))
            total += int(K.count_params(w))
    return total


def convert_all_kernels_in_model(model):
    """Converts all convolution kernels in a model from Theano to TensorFlow.

    Also works from TensorFlow to Theano.

    # Arguments
        model: target model for the conversion.
    """
    # Note: SeparableConvolution not included
    # since only supported by TF.
    conv_classes = {
        'Conv1D',
        'Conv2D',
        'Conv3D',
        'Conv2DTranspose',
    }
    to_assign = []
    for layer in model.layers:
        if layer.__class__.__name__ in conv_classes:
            original_kernel = K.get_value(layer.kernel)
            converted_kernel = convert_kernel(original_kernel)
            to_assign.append((layer.kernel, converted_kernel))
    K.batch_set_value(to_assign)


def convert_dense_weights_data_format(dense,
                                      previous_feature_map_shape,
                                      target_data_format='channels_first'):
    """Utility useful when changing a convnet's `data_format`.

    When porting the weights of a convnet from one data format to the other,
    if the convnet includes a `Flatten` layer
    (applied to the last convolutional feature map)
    followed by a `Dense` layer, the weights of that `Dense` layer
    should be updated to reflect the new dimension ordering.

    # Arguments
        dense: The target `Dense` layer.
        previous_feature_map_shape: A shape tuple of 3 integers,
            e.g. `(512, 7, 7)`. The shape of the convolutional
            feature map right before the `Flatten` layer that
            came before the target `Dense` layer.
        target_data_format: One of "channels_last", "channels_first".
            Set it "channels_last"
            if converting a "channels_first" model to "channels_last",
            or reciprocally.
    """
    assert target_data_format in {'channels_last', 'channels_first'}
    kernel, bias = dense.get_weights()
    for i in range(kernel.shape[1]):
        if target_data_format == 'channels_first':
            c, h, w = previous_feature_map_shape
            original_fm_shape = (h, w, c)
            ki = kernel[:, i].reshape(original_fm_shape)
            ki = np.transpose(ki, (2, 0, 1))  # last -> first
        else:
            h, w, c = previous_feature_map_shape
            original_fm_shape = (c, h, w)
            ki = kernel[:, i].reshape(original_fm_shape)
            ki = np.transpose(ki, (1, 2, 0))  # first -> last
        kernel[:, i] = np.reshape(ki, (np.prod(previous_feature_map_shape),))
    dense.set_weights([kernel, bias])
