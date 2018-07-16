"""Input layer code (`Input` and `InputLayer`).
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from .base_layer import Layer
from .base_layer import Node
from .. import backend as K
from ..legacy import interfaces
from ..utils.generic_utils import unpack_singleton


class InputLayer(Layer):
    """Layer to be used as an entry point into a model.

    It can either wrap an existing tensor (pass an `input_tensor` argument)
    or create its a placeholder tensor (pass arguments `input_shape`
    or `batch_input_shape` as well as `dtype`).

    # Arguments
        input_shape: Shape tuple, not including the batch axis.
        batch_size: Optional input batch size (integer or None).
        batch_input_shape: Shape tuple, including the batch axis.
        dtype: Datatype of the input.
        input_tensor: Optional tensor to use as layer input
            instead of creating a placeholder.
        sparse: Boolean, whether the placeholder created
            is meant to be sparse.
        name: Name of the layer (string).
    """

    @interfaces.legacy_input_support
    def __init__(self, input_shape=None, batch_size=None,
                 batch_input_shape=None,
                 dtype=None, input_tensor=None, sparse=False, name=None):
        if not name:
            prefix = 'input'
            name = prefix + '_' + str(K.get_uid(prefix))
        super(InputLayer, self).__init__(dtype=dtype, name=name)

        self.trainable = False
        self.built = True
        self.sparse = sparse

        if input_shape and batch_input_shape:
            raise ValueError('Only provide the input_shape OR '
                             'batch_input_shape argument to '
                             'InputLayer, not both at the same time.')
        if input_tensor is not None and batch_input_shape is None:
            # If input_tensor is set, and batch_input_shape is not set:
            # Attempt automatic input shape inference.
            try:
                batch_input_shape = K.int_shape(input_tensor)
            except TypeError:
                if not input_shape and not batch_input_shape:
                    raise ValueError('InputLayer was provided '
                                     'an input_tensor argument, '
                                     'but its input shape cannot be '
                                     'automatically inferred. '
                                     'You should pass an input_shape or '
                                     'batch_input_shape argument.')
        if not batch_input_shape:
            if not input_shape:
                raise ValueError('An Input layer should be passed either '
                                 'a `batch_input_shape` or an `input_shape`.')
            else:
                batch_input_shape = (batch_size,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)

        if not dtype:
            if input_tensor is None:
                dtype = K.floatx()
            else:
                dtype = K.dtype(input_tensor)

        self.batch_input_shape = batch_input_shape
        self.dtype = dtype

        if input_tensor is None:
            self.is_placeholder = True
            input_tensor = K.placeholder(shape=batch_input_shape,
                                         dtype=dtype,
                                         sparse=self.sparse,
                                         name=self.name)
        else:
            self.is_placeholder = False
            input_tensor._keras_shape = batch_input_shape
        # Create an input node to add to self.outbound_node
        # and set output_tensors' _keras_history.
        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[batch_input_shape],
             output_shapes=[batch_input_shape])

    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'dtype': self.dtype,
                  'sparse': self.sparse,
                  'name': self.name}
        return config


def Input(shape=None, batch_shape=None,
          name=None, dtype=None, sparse=False,
          tensor=None):
    """`Input()` is used to instantiate a Keras tensor.

    A Keras tensor is a tensor object from the underlying backend
    (Theano, TensorFlow or CNTK), which we augment with certain
    attributes that allow us to build a Keras model
    just by knowing the inputs and outputs of the model.

    For instance, if a, b and c are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`

    The added Keras attributes are:
        `_keras_shape`: Integer shape tuple propagated
            via Keras-side shape inference.
        `_keras_history`: Last layer applied to the tensor.
            the entire layer graph is retrievable from that layer,
            recursively.

    # Arguments
        shape: A shape tuple (integer), not including the batch size.
            For instance, `shape=(32,)` indicates that the expected input
            will be batches of 32-dimensional vectors.
        batch_shape: A shape tuple (integer), including the batch size.
            For instance, `batch_shape=(10, 32)` indicates that
            the expected input will be batches of 10 32-dimensional vectors.
            `batch_shape=(None, 32)` indicates batches of an arbitrary number
            of 32-dimensional vectors.
        name: An optional name string for the layer.
            Should be unique in a model (do not reuse the same name twice).
            It will be autogenerated if it isn't provided.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
        sparse: A boolean specifying whether the placeholder
            to be created is sparse.
        tensor: Optional existing tensor to wrap into the `Input` layer.
            If set, the layer will not create a placeholder tensor.

    # Returns
        A tensor.

    # Example

    ```python
    # this is a logistic regression in Keras
    x = Input(shape=(32,))
    y = Dense(16, activation='softmax')(x)
    model = Model(x, y)
    ```
    """
    if not batch_shape and tensor is None:
        assert shape is not None, ('Please provide to Input either a `shape`'
                                   ' or a `batch_shape` argument. Note that '
                                   '`shape` does not include the batch '
                                   'dimension.')
    if shape is not None and not batch_shape:
        batch_shape = (None,) + tuple(shape)
    if not dtype:
        dtype = K.floatx()
    input_layer = InputLayer(batch_input_shape=batch_shape,
                             name=name, dtype=dtype,
                             sparse=sparse,
                             input_tensor=tensor)
    # Return tensor including _keras_shape and _keras_history.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = input_layer._inbound_nodes[0].output_tensors
    return unpack_singleton(outputs)
