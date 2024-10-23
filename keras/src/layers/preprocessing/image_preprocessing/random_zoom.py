from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomZoom")
class RandomZoom(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly zooms images during training.

    This layer will randomly zoom in or out on each axis of an image
    independently, filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format,
        or `(..., channels, height, width)`, in `"channels_first"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`,
        or `(..., channels, target_height, target_width)`,
        in `"channels_first"` format.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        height_factor: a float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound for zooming vertically.
            When represented as a single float, this value is used for both the
            upper and lower bound. A positive value means zooming out, while a
            negative value means zooming in. For instance,
            `height_factor=(0.2, 0.3)` result in an output zoomed out by a
            random amount in the range `[+20%, +30%]`.
            `height_factor=(-0.3, -0.2)` result in an output zoomed in by a
            random amount in the range `[+20%, +30%]`.
        width_factor: a float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound for zooming horizontally.
            When represented as a single float, this value is used for both the
            upper and lower bound. For instance, `width_factor=(0.2, 0.3)`
            result in an output zooming out between 20% to 30%.
            `width_factor=(-0.3, -0.2)` result in an output zooming in between
            20% to 30%. `None` means i.e., zooming vertical and horizontal
            directions by preserving the aspect ratio. Defaults to `None`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about the edge of the last
                pixel.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond
                the edge with the same constant value k specified by
                `fill_value`.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The input is extended by wrapping around to the opposite edge.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
            Note that when using torch backend, `"reflect"` is redirected to
            `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not
            support `"reflect"`.
            Note that torch backend does not support `"wrap"`.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float that represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Example:

    >>> input_img = np.random.random((32, 224, 224, 3))
    >>> layer = keras.layers.RandomZoom(.5, .2)
    >>> out_img = layer(input_img)
    """

    _USE_BASE_FACTOR = False
    _FACTOR_VALIDATION_ERROR = (
        "The `height_factor` and `width_factor` arguments "
        "should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. "
    )
    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        height_factor,
        width_factor=None,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.height_factor = height_factor
        self.height_lower, self.height_upper = self._set_factor(
            height_factor, "height_factor"
        )
        self.width_factor = width_factor
        if width_factor is not None:
            self.width_lower, self.width_upper = self._set_factor(
                width_factor, "width_factor"
            )
        if fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.data_format = backend.standardize_data_format(data_format)
        self.supports_jit = False

    def _set_factor(self, factor, factor_name):
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR
                    + f"Received: {factor_name}={factor}"
                )
            self._check_factor_range(factor[0])
            self._check_factor_range(factor[1])
            lower, upper = sorted(factor)
        elif isinstance(factor, (int, float)):
            self._check_factor_range(factor)
            factor = abs(factor)
            lower, upper = [-factor, factor]
        else:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: {factor_name}={factor}"
            )
        return lower, upper

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < -1.0:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR
                + f"Received: input_number={input_number}"
            )

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training:
            return self._zoom_inputs(images, transformation)
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        raise NotImplementedError

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None
        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        images_shape = self.backend.shape(images)
        if len(images_shape) == 4:
            zoom_factor_shape = (images_shape[0], 1)
        else:
            zoom_factor_shape = (1, 1)

        if not training:
            return {
                "height_zoom": self.backend.numpy.zeros(zoom_factor_shape),
                "width_zoom": self.backend.numpy.zeros(zoom_factor_shape),
            }
        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        height_zoom = self.backend.random.uniform(
            minval=1.0 + self.height_lower,
            maxval=1.0 + self.height_upper,
            shape=zoom_factor_shape,
            seed=seed,
        )
        if self.width_factor is not None:
            width_zoom = self.backend.random.uniform(
                minval=1.0 + self.width_lower,
                maxval=1.0 + self.width_upper,
                shape=zoom_factor_shape,
                seed=seed,
            )
        else:
            width_zoom = height_zoom
        return {
            "height_zoom": height_zoom,
            "width_zoom": width_zoom,
        }

    def _zoom_inputs(self, inputs, transformation):
        if transformation is None:
            return inputs

        width_zoom = transformation["width_zoom"]
        height_zoom = transformation["height_zoom"]
        zooms = self.backend.cast(
            self.backend.numpy.concatenate([width_zoom, height_zoom], axis=1),
            dtype="float32",
        )

        inputs_shape = self.backend.shape(inputs)
        unbatched = len(inputs_shape) == 3
        if unbatched:
            inputs = self.backend.numpy.expand_dims(inputs, axis=0)
            inputs_shape = self.backend.shape(inputs)
        if self.data_format == "channels_first":
            height = inputs_shape[-2]
            width = inputs_shape[-1]
        else:
            height = inputs_shape[-3]
            width = inputs_shape[-2]

        outputs = self.backend.image.affine_transform(
            inputs,
            transform=self._get_zoom_matrix(zooms, height, width),
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

        if unbatched:
            outputs = self.backend.numpy.squeeze(outputs, axis=0)
        return outputs

    def _get_zoom_matrix(self, zooms, image_height, image_width):
        num_zooms = self.backend.shape(zooms)[0]
        # The zoom matrix looks like:
        #     [[zx 0 0]
        #      [0 zy 0]
        #      [0 0 1]]
        # where the last entry is implicit.
        # zoom matrices are always float32.
        x_offset = ((self.backend.cast(image_width, "float32") - 1.0) / 2.0) * (
            1.0 - zooms[:, 0:1]
        )
        y_offset = (
            (self.backend.cast(image_height, "float32") - 1.0) / 2.0
        ) * (1.0 - zooms[:, 1:])
        return self.backend.numpy.concatenate(
            [
                zooms[:, 0:1],
                self.backend.numpy.zeros((num_zooms, 1)),
                x_offset,
                self.backend.numpy.zeros((num_zooms, 1)),
                zooms[:, 1:],
                y_offset,
                self.backend.numpy.zeros((num_zooms, 2)),
            ],
            axis=1,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "interpolation": self.interpolation,
            "seed": self.seed,
            "fill_value": self.fill_value,
            "data_format": self.data_format,
        }
        return {**base_config, **config}
