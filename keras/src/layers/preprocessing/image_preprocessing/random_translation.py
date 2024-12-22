from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    clip_to_image_size,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    convert_format,
)
from keras.src.random.seed_generator import SeedGenerator
from keras.src.utils import backend_utils


@keras_export("keras.layers.RandomTranslation")
class RandomTranslation(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly translates images during training.

    This layer will apply random translations to each image during training,
    filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

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
            size 2 representing lower and upper bound for shifting vertically. A
            negative value means shifting image up, while a positive value means
            shifting image down. When represented as a single positive float,
            this value is used for both the upper and lower bound. For instance,
            `height_factor=(-0.2, 0.3)` results in an output shifted by a random
            amount in the range `[-20%, +30%]`. `height_factor=0.2` results in
            an output height shifted by a random amount in the range
            `[-20%, +20%]`.
        width_factor: a float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound for shifting horizontally.
            A negative value means shifting image left, while a positive value
            means shifting image right. When represented as a single positive
            float, this value is used for both the upper and lower bound. For
            instance, `width_factor=(-0.2, 0.3)` results in an output shifted
            left by 20%, and shifted right by 30%. `width_factor=0.2` results
            in an output height shifted left or right by 20%.
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
        fill_value: a float represents the value to be filled outside the
            boundaries when `fill_mode="constant"`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
    """

    _USE_BASE_FACTOR = False
    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. "
    )
    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.height_factor = height_factor
        self.height_lower, self.height_upper = self._set_factor(
            height_factor, "height_factor"
        )
        self.width_factor = width_factor
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
            return self._translate_inputs(images, transformation)
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def get_transformed_x_y(self, x, y, transform):
        a0, a1, a2, b0, b1, b2, c0, c1 = self.backend.numpy.split(
            transform, 8, axis=-1
        )

        k = c0 * x + c1 * y + 1
        x_transformed = (a0 * x + a1 * y + a2) / k
        y_transformed = (b0 * x + b1 * y + b2) / k
        return x_transformed, y_transformed

    def get_shifted_bbox(self, bounding_boxes, w_shift_factor, h_shift_factor):
        bboxes = bounding_boxes["boxes"]
        x1, x2, x3, x4 = self.backend.numpy.split(bboxes, 4, axis=-1)

        w_shift_factor = self.backend.convert_to_tensor(
            w_shift_factor, dtype=x1.dtype
        )
        h_shift_factor = self.backend.convert_to_tensor(
            h_shift_factor, dtype=x1.dtype
        )

        if len(bboxes.shape) == 3:
            w_shift_factor = self.backend.numpy.expand_dims(w_shift_factor, -1)
            h_shift_factor = self.backend.numpy.expand_dims(h_shift_factor, -1)

        bounding_boxes["boxes"] = self.backend.numpy.concatenate(
            [
                x1 - w_shift_factor,
                x2 - h_shift_factor,
                x3 - w_shift_factor,
                x4 - h_shift_factor,
            ],
            axis=-1,
        )
        return bounding_boxes

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        if training:
            if backend_utils.in_tf_graph():
                self.backend.set_backend("tensorflow")

            if self.data_format == "channels_first":
                height_axis = -2
                width_axis = -1
            else:
                height_axis = -3
                width_axis = -2

            input_height, input_width = (
                transformation["input_shape"][height_axis],
                transformation["input_shape"][width_axis],
            )

            bounding_boxes = convert_format(
                bounding_boxes,
                source=self.bounding_box_format,
                target="xyxy",
                height=input_height,
                width=input_width,
            )

            translations = transformation["translations"]
            transform = self._get_translation_matrix(translations)

            w_shift_factor, h_shift_factor = self.get_transformed_x_y(
                0, 0, transform
            )
            bounding_boxes = self.get_shifted_bbox(
                bounding_boxes, w_shift_factor, h_shift_factor
            )

            bounding_boxes = clip_to_image_size(
                bounding_boxes=bounding_boxes,
                height=input_height,
                width=input_width,
                bounding_box_format="xyxy",
            )

            bounding_boxes = convert_format(
                bounding_boxes,
                source="xyxy",
                target=self.bounding_box_format,
                height=input_height,
                width=input_width,
            )

            self.backend.reset()

        return bounding_boxes

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
        unbatched = len(images_shape) == 3
        if unbatched:
            images_shape = self.backend.shape(images)
            batch_size = 1
        else:
            batch_size = images_shape[0]
        if self.data_format == "channels_first":
            height = images_shape[-2]
            width = images_shape[-1]
        else:
            height = images_shape[-3]
            width = images_shape[-2]

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        height_translate = self.backend.random.uniform(
            minval=self.height_lower,
            maxval=self.height_upper,
            shape=[batch_size, 1],
            seed=seed,
        )
        height_translate = self.backend.numpy.multiply(height_translate, height)
        width_translate = self.backend.random.uniform(
            minval=self.width_lower,
            maxval=self.width_upper,
            shape=[batch_size, 1],
            seed=seed,
        )
        width_translate = self.backend.numpy.multiply(width_translate, width)
        translations = self.backend.cast(
            self.backend.numpy.concatenate(
                [width_translate, height_translate], axis=1
            ),
            dtype="float32",
        )
        return {"translations": translations, "input_shape": images_shape}

    def _translate_inputs(self, inputs, transformation):
        if transformation is None:
            return inputs

        inputs_shape = self.backend.shape(inputs)
        unbatched = len(inputs_shape) == 3
        if unbatched:
            inputs = self.backend.numpy.expand_dims(inputs, axis=0)

        translations = transformation["translations"]
        outputs = self.backend.image.affine_transform(
            inputs,
            transform=self._get_translation_matrix(translations),
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

        if unbatched:
            outputs = self.backend.numpy.squeeze(outputs, axis=0)
        return outputs

    def _get_translation_matrix(self, translations):
        num_translations = self.backend.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # translation matrices are always float32.
        return self.backend.numpy.concatenate(
            [
                self.backend.numpy.ones((num_translations, 1)),
                self.backend.numpy.zeros((num_translations, 1)),
                -translations[:, 0:1],
                self.backend.numpy.zeros((num_translations, 1)),
                self.backend.numpy.ones((num_translations, 1)),
                -translations[:, 1:],
                self.backend.numpy.zeros((num_translations, 2)),
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
