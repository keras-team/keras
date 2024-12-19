from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomContrast")
class RandomContrast(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly adjusts contrast during training.

    This layer will randomly adjust the contrast of an image or images
    by a random factor. Contrast is adjusted independently
    for each channel of each image during training.

    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    in integer or floating point dtype.
    By default, the layer will output floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound.
            When represented as a single float, lower = upper.
            The contrast factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,
            the output will be `(x - mean) * factor + mean`
            where `mean` is the mean value of the channel.
        value_range: the range of values the incoming images will have.
            Represented as a two-number tuple written `[low, high]`. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        seed: Integer. Used to create a random seed.
    """

    _FACTOR_BOUNDS = (0, 1)

    def __init__(self, factor, value_range=(0, 255), seed=None, **kwargs):
        super().__init__(**kwargs)
        self._set_factor(factor)
        self.value_range = value_range
        self.seed = seed
        self.generator = SeedGenerator(seed)

    def get_random_transformation(self, data, training=True, seed=None):
        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        images_shape = self.backend.shape(images)
        rank = len(images_shape)
        if rank == 3:
            factor_shape = (1, 1, 1)
        elif rank == 4:
            # Keep only the batch dim. This will ensure to have same adjustment
            # with in one image, but different across the images.
            factor_shape = [images_shape[0], 1, 1, 1]
        else:
            raise ValueError(
                "Expected the input image to be rank 3 or 4. Received "
                f"inputs.shape={images_shape}"
            )

        if not training:
            return {"contrast_factor": self.backend.numpy.zeros(factor_shape)}

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        factor = self.backend.random.uniform(
            shape=factor_shape,
            minval=1.0 - self.factor[0],
            maxval=1.0 + self.factor[1],
            seed=seed,
            dtype=self.compute_dtype,
        )
        return {"contrast_factor": factor}

    def transform_images(self, images, transformation, training=True):
        if training:
            constrast_factor = transformation["contrast_factor"]
            outputs = self._adjust_constrast(images, constrast_factor)
            outputs = self.backend.numpy.clip(
                outputs, self.value_range[0], self.value_range[1]
            )
            self.backend.numpy.reshape(outputs, self.backend.shape(images))
            return outputs
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks

    def _adjust_constrast(self, inputs, contrast_factor):
        if self.data_format == "channels_first":
            height_axis = -2
            width_axis = -1
        else:
            height_axis = -3
            width_axis = -2
        # reduce mean on height
        inp_mean = self.backend.numpy.mean(
            inputs, axis=height_axis, keepdims=True
        )
        # reduce mean on width
        inp_mean = self.backend.numpy.mean(
            inp_mean, axis=width_axis, keepdims=True
        )

        outputs = (inputs - inp_mean) * contrast_factor + inp_mean
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
