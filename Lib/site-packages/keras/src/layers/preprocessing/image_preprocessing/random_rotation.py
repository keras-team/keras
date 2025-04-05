from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes import (
    converters,
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
            amount in the range `[-20% * 360, 30% * 360]`.
            `factor=0.2` results in an
            output rotating by a random amount
            in the range `[-20% * 360, 20% * 360]`.
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
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        if training:
            ops = self.backend
            boxes = bounding_boxes["boxes"]
            height = transformation["image_height"]
            width = transformation["image_width"]
            batch_size = transformation["batch_size"]
            boxes = converters.affine_transform(
                boxes=boxes,
                angle=transformation["angle"],
                translate_x=ops.numpy.zeros([batch_size]),
                translate_y=ops.numpy.zeros([batch_size]),
                scale=ops.numpy.ones([batch_size]),
                shear_x=ops.numpy.zeros([batch_size]),
                shear_y=ops.numpy.zeros([batch_size]),
                height=height,
                width=width,
            )

            bounding_boxes["boxes"] = boxes
            bounding_boxes = converters.clip_to_image_size(
                bounding_boxes,
                height=height,
                width=width,
                bounding_box_format="xyxy",
            )
            bounding_boxes = converters.convert_format(
                bounding_boxes,
                source="xyxy",
                target=self.bounding_box_format,
                height=height,
                width=width,
            )
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def get_random_transformation(self, data, training=True, seed=None):
        ops = self.backend
        if not training:
            return None
        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        shape = ops.core.shape(images)
        if len(shape) == 4:
            batch_size = shape[0]
            if self.data_format == "channels_last":
                image_height = shape[1]
                image_width = shape[2]
            else:
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

        if seed is None:
            seed = self._get_seed_generator(ops._backend)
        lower = self.factor[0] * 360.0
        upper = self.factor[1] * 360.0
        angle = ops.random.uniform(
            shape=(batch_size,),
            minval=lower,
            maxval=upper,
            seed=seed,
        )
        center_x, center_y = 0.5, 0.5
        rotation_matrix = self._compute_affine_matrix(
            center_x=center_x,
            center_y=center_y,
            angle=angle,
            translate_x=ops.numpy.zeros([batch_size]),
            translate_y=ops.numpy.zeros([batch_size]),
            scale=ops.numpy.ones([batch_size]),
            shear_x=ops.numpy.zeros([batch_size]),
            shear_y=ops.numpy.zeros([batch_size]),
            height=image_height,
            width=image_width,
        )
        if len(shape) == 3:
            rotation_matrix = self.backend.numpy.squeeze(
                rotation_matrix, axis=0
            )
        return {
            "angle": angle,
            "rotation_matrix": rotation_matrix,
            "image_height": image_height,
            "image_width": image_width,
            "batch_size": batch_size,
        }

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
