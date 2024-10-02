import numpy as np

from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomRotation")
class RandomRotation(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly rotates images during training.

    This layer will apply random rotations to each image, filling empty space
    according to `fill_mode`.

    By default, random rotations are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    rotations at inference time, pass `training=True` when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        factor: a float represented as fraction of 2 Pi, or a tuple of size 2
            representing lower and upper bound for rotating clockwise and
            counter-clockwise. A positive values means rotating
            counter clock-wise,
            while a negative value means clock-wise.
            When represented as a single
            float, this value is used for both the upper and lower bound.
            For instance, `factor=(-0.2, 0.3)`
            results in an output rotation by a random
            amount in the range `[-20% * 2pi, 30% * 2pi]`.
            `factor=0.2` results in an
            output rotating by a random amount
            in the range `[-20% * 2pi, 20% * 2pi]`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
            - *reflect*: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about
                the edge of the last pixel.
            - *constant*: `(k k k k | a b c d | k k k k)`
                The input is extended by
                filling all values beyond the edge with
                the same constant value k = 0.
            - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
                wrapping around to the opposite edge.
            - *nearest*: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
    """

    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        data_format=None,
        **kwargs,
    ):
        super().__init__(factor=factor, data_format=data_format, **kwargs)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.supports_jit = False

        if self.fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if self.interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training:
            return self.backend.image.affine_transform(
                images=images,
                transform=transformation["rotation_matrix"],
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                data_format=self.data_format,
            )
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        boxes = bounding_boxes["boxes"]
        shape = self.backend.shape(boxes)
        ones = self.backend.ones((shape[0], shape[1], 1, 1))
        homogeneous_boxes = self.backend.concatenate([boxes, ones], axis=2)
        transformed_boxes = self.backend.matmul(
            transformation["rotation_matrix"], homogeneous_boxes
        )
        # Convert back to xyxy format
        transformed_boxes = (
            transformed_boxes[:, :, :2, :] / transformed_boxes[:, :, 2:3, :]
        )
        transformed_boxes = self.backend.reshape(
            transformed_boxes, (shape[0], shape[1], 4)
        )
        return {"boxes": transformed_boxes, "labels": bounding_boxes["labels"]}

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    """
    Assume an angle ø, then rotation matrix is defined by
    | cos(ø)   -sin(ø)  x_offset |
    | sin(ø)    cos(ø)  y_offset |
    |   0         0         1    |

    This function is returning the 8 elements barring the final 1 as a 1D array
    """

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None
        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        shape = self.backend.core.shape(images)
        if len(shape) == 4:
            if self.data_format == "channels_last":
                batch_size = shape[0]
                image_height = shape[1]
                image_width = shape[2]
            else:
                batch_size = shape[1]
                image_height = shape[2]
                image_width = shape[3]
        else:
            batch_size = 1
            if self.data_format == "channels_last":
                image_height = shape[0]
                image_width = shape[1]
            else:
                image_height = shape[1]
                image_width = shape[2]

        lower = self.factor[0] * 2.0 * self.backend.convert_to_tensor(np.pi)
        upper = self.factor[1] * 2.0 * self.backend.convert_to_tensor(np.pi)

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)
        angle = self.backend.random.uniform(
            shape=(batch_size,),
            minval=lower,
            maxval=upper,
            seed=seed,
        )

        cos_theta = self.backend.numpy.cos(angle)
        sin_theta = self.backend.numpy.sin(angle)
        image_height = self.backend.core.cast(image_height, cos_theta.dtype)
        image_width = self.backend.core.cast(image_width, cos_theta.dtype)

        x_offset = (
            (image_width - 1)
            - (cos_theta * (image_width - 1) - sin_theta * (image_height - 1))
        ) / 2.0

        y_offset = (
            (image_height - 1)
            - (sin_theta * (image_width - 1) + cos_theta * (image_height - 1))
        ) / 2.0

        rotation_matrix = self.backend.numpy.concatenate(
            [
                self.backend.numpy.cos(angle)[:, None],
                -self.backend.numpy.sin(angle)[:, None],
                x_offset[:, None],
                self.backend.numpy.sin(angle)[:, None],
                self.backend.numpy.cos(angle)[:, None],
                y_offset[:, None],
                self.backend.numpy.zeros((batch_size, 2)),
            ],
            axis=1,
        )
        if len(shape) == 3:
            rotation_matrix = self.backend.numpy.squeeze(
                rotation_matrix, axis=0
            )
        return {"rotation_matrix": rotation_matrix}

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "data_format": self.data_format,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
