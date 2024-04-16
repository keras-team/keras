from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
from keras.src.utils import image_utils


@keras_export("keras.layers.RandomCrop")
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
        super().__init__(name=name, **kwargs)
        self.height = height
        self.width = width
        self.seed = (
            seed if seed is not None else backend.random.make_default_seed()
        )
        self.generator = SeedGenerator(seed)
        self.data_format = backend.standardize_data_format(data_format)

        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
        elif self.data_format == "channels_last":
            self.height_axis = -3
            self.width_axis = -2

        self.supports_masking = False
        self.supports_jit = False
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, training=True):
        inputs = self.backend.cast(inputs, self.compute_dtype)
        input_shape = self.backend.shape(inputs)
        is_batched = len(input_shape) > 3
        if not is_batched:
            inputs = self.backend.numpy.expand_dims(inputs, axis=0)

        h_diff = input_shape[self.height_axis] - self.height
        w_diff = input_shape[self.width_axis] - self.width

        def random_crop():
            input_height, input_width = (
                input_shape[self.height_axis],
                input_shape[self.width_axis],
            )

            seed_generator = self._get_seed_generator(self.backend._backend)
            h_start = self.backend.cast(
                self.backend.random.uniform(
                    (),
                    0,
                    maxval=float(input_height - self.height + 1),
                    seed=seed_generator,
                ),
                "int32",
            )
            w_start = self.backend.cast(
                self.backend.random.uniform(
                    (),
                    0,
                    maxval=float(input_width - self.width + 1),
                    seed=seed_generator,
                ),
                "int32",
            )
            if self.data_format == "channels_last":
                return self.backend.core.slice(
                    inputs,
                    self.backend.numpy.stack([0, h_start, w_start, 0]),
                    [
                        self.backend.shape(inputs)[0],
                        self.height,
                        self.width,
                        self.backend.shape(inputs)[3],
                    ],
                )
            else:
                return self.backend.core.slice(
                    inputs,
                    self.backend.numpy.stack([0, 0, h_start, w_start]),
                    [
                        self.backend.shape(inputs)[0],
                        self.backend.shape(inputs)[1],
                        self.height,
                        self.width,
                    ],
                )

        def resize():
            outputs = image_utils.smart_resize(
                inputs,
                [self.height, self.width],
                data_format=self.data_format,
                backend_module=self.backend,
            )
            # smart_resize will always output float32, so we need to re-cast.
            return self.backend.cast(outputs, self.compute_dtype)

        if isinstance(h_diff, int) and isinstance(w_diff, int):
            if training and h_diff >= 0 and w_diff >= 0:
                outputs = random_crop()
            else:
                outputs = resize()
        else:
            predicate = self.backend.numpy.logical_and(
                training,
                self.backend.numpy.logical_and(h_diff >= 0, w_diff >= 0),
            )
            outputs = self.backend.cond(
                predicate,
                random_crop,
                resize,
            )

        if not is_batched:
            outputs = self.backend.numpy.squeeze(outputs, axis=0)
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
