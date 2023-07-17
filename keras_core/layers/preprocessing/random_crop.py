import numpy as np

from keras_core import backend
from keras_core import random
from keras_core.api_export import keras_core_export
from keras_core.layers.preprocessing.tf_data_layer import TFDataLayer
from keras_core.random.seed_generator import SeedGenerator
from keras_core.utils import backend_utils
from keras_core.utils import image_utils
from keras_core.utils.module_utils import tensorflow as tf


@keras_core_export("keras_core.layers.RandomCrop")
class RandomCrop(TFDataLayer):
    """A preprocessing layer which randomly crops images during training.

    During training, this layer will randomly choose a location to crop images
    down to a target size. The layer will crop all the images in the same batch
    to the same cropping location.

    At inference time, and during training if an input image is smaller than the
    target size, the input will be resized and cropped so as to return the
    largest possible window in the image that matches the target aspect ratio.
    If you need to apply random cropping at inference time, set `training` to
    True when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    **Note:** This layer wraps `tf.keras.layers.RandomCrop`. It cannot
    be used as part of the compiled computation graph of a model with
    any backend other than TensorFlow.
    It can however be used with any backend when running eagerly.
    It can also always be used as part of an input preprocessing pipeline
    with any backend (outside the model itself), which is how we recommend
    to use this layer.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`.

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        seed: Integer. Used to create a random seed.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    def __init__(
        self, height, width, seed=None, data_format=None, name=None, **kwargs
    ):
        if not tf.available:
            raise ImportError(
                "Layer RandomCrop requires TensorFlow. "
                "Install it via `pip install tensorflow`."
            )

        super().__init__(name=name, **kwargs)
        self.height = height
        self.width = width
        self.seed = seed or backend.random.make_default_seed()
        self.seed_generator = SeedGenerator(seed)
        self.data_format = backend.standardize_data_format(data_format)

        if self.data_format == "channels_first":
            self.heigh_axis = -2
            self.width_axis = -1
        elif self.data_format == "channels_last":
            self.height_axis = -3
            self.width_axis = -2

        self.supports_masking = False
        self.supports_jit = False
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, training=True):
        if not isinstance(inputs, (tf.Tensor, np.ndarray, list, tuple)):
            inputs = self.backend.convert_to_tensor(
                backend.convert_to_numpy(inputs)
            )

        input_shape = self.backend.shape(inputs)
        is_batched = len(input_shape) > 3
        inputs = (
            self.backend.expand_dims(inputs, axis=0)
            if not is_batched
            else inputs
        )

        h_diff = input_shape[self.height_axis] - self.height
        w_diff = input_shape[self.width_axis] - self.width

        def random_crop():
            dtype = input_shape.dtype
            rands = random.uniform(
                [2], 0, dtype.max, dtype, seed=self.self.seed_generator
            )
            h_start = rands[0] % (h_diff + 1)
            w_start = rands[1] % (w_diff + 1)
            if self.data_format == "channels_last":
                return inputs[
                    :,
                    h_start : h_start + self.height,
                    w_start : w_start + self.width,
                ]
            else:
                return inputs[
                    :,
                    :,
                    h_start : h_start + self.height,
                    w_start : w_start + self.width,
                ]

        def resize():
            outputs = image_utils.smart_resize(
                inputs,
                [self.height, self.width],
                data_format=self.data_format,
                backend_module=self.backend,
            )
            # smart_resize will always output float32, so we need to re-cast.
            return self.backend.cast(outputs, self.compute_dtype)

        outputs = self.backend.cond(
            self.backend.all((training, h_diff >= 0, w_diff >= 0)),
            random_crop,
            resize,
        )

        if (
            self.backend.backend() != "tensorflow"
            and not backend_utils.in_tf_graph()
        ):
            outputs = self.backend.convert_to_tensor(outputs)
        return outputs

    def compute_output_shape(self, input_shape, *args, **kwargs):
        input_shape = list(input_shape)
        input_shape[self.height_axis] = self.height
        input_shape[self.width_axis] = self.width
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "seed": self.seed,
                "data_format": self.data_format,
            }
        )
        return config
