import numpy as np
import tensorflow as tf

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer
from keras_core.utils import backend_utils


@keras_core_export("keras_core.layers.RandomTranslation")
class RandomTranslation(Layer):
    """A preprocessing layer which randomly translates images during training.

    This layer will apply random translations to each image during training,
    filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
      height_factor: a float represented as fraction of value, or a tuple of
          size 2 representing lower and upper bound for shifting vertically. A
          negative value means shifting image up, while a positive value means
          shifting image down. When represented as a single positive float, this
          value is used for both the upper and lower bound. For instance,
          `height_factor=(-0.2, 0.3)` results in an output shifted by a random
          amount in the range `[-20%, +30%]`.  `height_factor=0.2` results in an
          output height shifted by a random amount in the range `[-20%, +20%]`.
      width_factor: a float represented as fraction of value, or a tuple of size
          2 representing lower and upper bound for shifting horizontally. A
          negative value means shifting image left, while a positive value means
          shifting image right. When represented as a single positive float,
          this value is used for both the upper and lower bound. For instance,
          `width_factor=(-0.2, 0.3)` results in an output shifted left by 20%,
          and shifted right by 30%. `width_factor=0.2` results
          in an output height shifted left or right by 20%.
      fill_mode: Points outside the boundaries of the input are filled according
          to the given mode
          (one of `{"constant", "reflect", "wrap", "nearest"}`).
          - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
              reflecting about the edge of the last pixel.
          - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
              filling all values beyond the edge with the same constant value
              k = 0.
          - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
              wrapping around to the opposite edge.
          - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
              the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
          `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
          boundaries when `fill_mode="constant"`.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.
    """

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.layer = tf.keras.layers.RandomTranslation(
            height_factor=height_factor,
            width_factor=width_factor,
            fill_mode=fill_mode,
            interpolation=interpolation,
            seed=seed,
            fill_value=fill_value,
            name=name,
            **kwargs,
        )
        self.supports_jit = False
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, training=True):
        if not isinstance(inputs, (tf.Tensor, np.ndarray, list, tuple)):
            inputs = tf.convert_to_tensor(backend.convert_to_numpy(inputs))
        outputs = self.layer.call(inputs, training=training)
        if (
            backend.backend() != "tensorflow"
            and not backend_utils.in_tf_graph()
        ):
            outputs = backend.convert_to_tensor(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(self.layer.compute_output_shape(input_shape))

    def get_config(self):
        return self.layer.get_config()
