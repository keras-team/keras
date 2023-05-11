import numpy as np
import tensorflow as tf

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


@keras_core_export("keras_core.layers.RandomFlip")
class RandomFlip(Layer):
    """A preprocessing layer which randomly flips images during training.

    This layer will flip the images horizontally and or vertically based on the
    `mode` attribute. During inference time, the output will be identical to
    input. Call the layer with `training=True` to flip the input.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        mode: String indicating which flip mode to use. Can be `"horizontal"`,
            `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a
            left-right flip and `"vertical"` is a top-bottom flip. Defaults to
            `"horizontal_and_vertical"`
        seed: Integer. Used to create a random seed.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    def __init__(
        self, mode=HORIZONTAL_AND_VERTICAL, seed=None, name=None, **kwargs
    ):
        super().__init__(name=name)
        self.seed = seed or backend.random.make_default_seed()
        self.layer = tf.keras.layers.RandomFlip(
            mode=mode,
            name=name,
            seed=self.seed,
            **kwargs,
        )

    def call(self, inputs, training=True):
        if not isinstance(inputs, (tf.Tensor, np.ndarray, list, tuple)):
            inputs = tf.convert_to_tensor(np.array(inputs))
        outputs = self.layer.call(inputs)
        if backend.backend() != "tensorflow":
            outputs = backend.convert_to_tensor(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(self.layer.compute_output_shape(input_shape))

    def get_config(self):
        config = self.layer.get_config()
        config.update({"seed": self.seed})
        return config
