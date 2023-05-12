import numpy as np
import tensorflow as tf

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.RandomZoom")
class RandomZoom(Layer):
    """A preprocessing layer which randomly zooms images during training.

    This layer will randomly zoom in or out on each axis of an image
    independently, filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    **Note:** This layer wraps `tf.keras.layers.RandomZoom`. It cannot
    be used as part of the compiled computation graph of a model with
    any backend other than TensorFlow.
    It can however be used with any backend when running eagerly.
    It can also always be used as part of an input preprocessing pipeline
    with any backend (outside the model itself), which is how we recommend
    to use this layer.

    Args:
        height_factor: a float represented as fraction of value,
            or a tuple of size 2 representing lower and upper bound
            for zooming vertically. When represented as a single float,
            this value is used for both the upper and
            lower bound. A positive value means zooming out,
            while a negative value
            means zooming in. For instance, `height_factor=(0.2, 0.3)`
            result in an output zoomed out by a random amount
            in the range `[+20%, +30%]`.
            `height_factor=(-0.3, -0.2)` result in an output zoomed
            in by a random amount in the range `[+20%, +30%]`.
        width_factor: a float represented as fraction of value,
            or a tuple of size 2 representing lower and upper bound
            for zooming horizontally. When
            represented as a single float, this value is used
            for both the upper and
            lower bound. For instance, `width_factor=(0.2, 0.3)`
            result in an output
            zooming out between 20% to 30%.
            `width_factor=(-0.3, -0.2)` result in an
            output zooming in between 20% to 30%. `None` means
            i.e., zooming vertical and horizontal directions
            by preserving the aspect ratio. Defaults to `None`.
        fill_mode: Points outside the boundaries of the input are
            filled according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
            - *reflect*: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about
                the edge of the last pixel.
            - *constant*: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond
                the edge with the same constant value k = 0.
            - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
                wrapping around to the opposite edge.
            - *nearest*: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.

    Example:

    >>> input_img = np.random.random((32, 224, 224, 3))
    >>> layer = keras_core.layers.RandomZoom(.5, .2)
    >>> out_img = layer(input_img)

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    """

    def __init__(
        self,
        height_factor,
        width_factor=None,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.seed = seed or backend.random.make_default_seed()
        self.layer = tf.keras.layers.RandomZoom(
            height_factor=height_factor,
            width_factor=width_factor,
            fill_mode=fill_mode,
            interpolation=interpolation,
            seed=self.seed,
            name=name,
            fill_value=fill_value,
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
