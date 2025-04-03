"""Wrapper layer to apply every temporal slice of an input."""

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.core.wrapper import Wrapper
from keras.src.layers.layer import Layer


@keras_export("keras.layers.TimeDistributed")
class TimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every temporal slice of an input.

    Every input should be at least 3D, and the dimension of index one of the
    first input will be considered to be the temporal dimension.

    Consider a batch of 32 video samples, where each sample is a 128x128 RGB
    image with `channels_last` data format, across 10 timesteps.
    The batch input shape is `(32, 10, 128, 128, 3)`.

    You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
    of the 10 timesteps, independently:

    >>> inputs = layers.Input(shape=(10, 128, 128, 3), batch_size=32)
    >>> conv_2d_layer = layers.Conv2D(64, (3, 3))
    >>> outputs = layers.TimeDistributed(conv_2d_layer)(inputs)
    >>> outputs.shape
    (32, 10, 126, 126, 64)

    Because `TimeDistributed` applies the same instance of `Conv2D` to each of
    the timestamps, the same set of weights are used at each timestamp.

    Args:
        layer: a `keras.layers.Layer` instance.

    Call arguments:
        inputs: Input tensor of shape (batch, time, ...) or nested tensors,
            and each of which has shape (batch, time, ...).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            wrapped layer (only if the layer supports this argument).
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked. This argument is passed to the
            wrapped layer (only if the layer supports this argument).
    """

    def __init__(self, layer, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                "Please initialize `TimeDistributed` layer with a "
                f"`keras.layers.Layer` instance. Received: {layer}"
            )
        super().__init__(layer, **kwargs)
        self.supports_masking = True

    def _get_child_input_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
            raise ValueError(
                "`TimeDistributed` Layer should be passed an `input_shape` "
                f"with at least 3 dimensions, received: {input_shape}"
            )
        return (input_shape[0], *input_shape[2:])

    def compute_output_shape(self, input_shape):
        child_input_shape = self._get_child_input_shape(input_shape)
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        return (child_output_shape[0], input_shape[1], *child_output_shape[1:])

    def build(self, input_shape):
        child_input_shape = self._get_child_input_shape(input_shape)
        super().build(child_input_shape)
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = ops.shape(inputs)
        mask_shape = None if mask is None else ops.shape(mask)
        batch_size = input_shape[0]
        timesteps = input_shape[1]

        # For TF backend with graph mode and `partial_batch_size`, skip
        # evaluation of `batch_size` as it can be a `strided_slice` and
        # not a constant.
        if backend.backend() == "tensorflow":
            from keras.src.utils.module_utils import tensorflow as tf

            if (
                not tf.executing_eagerly
                and mask_shape is not None
                and mask_shape[1:2] != (timesteps,)
            ):
                raise ValueError(
                    "`TimeDistributed` Layer should be passed a `mask` of "
                    f"shape ({batch_size}, {timesteps}, ...), "
                    f"received: mask.shape={mask_shape}"
                )
        elif mask_shape is not None and mask_shape[:2] != (
            batch_size,
            timesteps,
        ):
            raise ValueError(
                "`TimeDistributed` Layer should be passed a `mask` of "
                f"shape ({batch_size}, {timesteps}, ...), "
                f"received: mask.shape={mask_shape}"
            )

        def time_distributed_transpose(data):
            """Swaps the timestep and batch dimensions of a tensor."""
            axes = [1, 0, *range(2, len(data.shape))]
            return ops.transpose(data, axes=axes)

        inputs = time_distributed_transpose(inputs)
        if mask is not None:
            mask = time_distributed_transpose(mask)

        def step_function(i):
            kwargs = {}
            if self.layer._call_has_mask_arg and mask is not None:
                kwargs["mask"] = mask[i]
            if self.layer._call_has_training_arg:
                kwargs["training"] = training
            return self.layer.call(inputs[i], **kwargs)

        # Implementation #1: is the time axis is static, use a Python for loop.

        if inputs.shape[0] is not None:
            outputs = ops.stack(
                [step_function(i) for i in range(inputs.shape[0])]
            )
            return time_distributed_transpose(outputs)

        # Implementation #2: use backend.vectorized_map.

        outputs = backend.vectorized_map(step_function, ops.arange(timesteps))
        return time_distributed_transpose(outputs)
