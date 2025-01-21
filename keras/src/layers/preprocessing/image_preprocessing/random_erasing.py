from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator
import tensorflow as tf


@keras_export("keras.layers.RandomErasing")
class RandomErasing(BaseImagePreprocessingLayer):
    """CutMix data augmentation technique.

    CutMix is a data augmentation method where patches are cut and pasted
    between two images in the dataset, while the labels are also mixed
    proportionally to the area of the patches.

    Args:
        factor: A single float or a tuple of two floats between 0 and 1.
            If a tuple of numbers is passed, a `factor` is sampled
            between the two values.
            If a single float is passed, a value between 0 and the passed
            float is sampled. These values define the range from which the
            mixing weight is sampled. A higher factor increases the variability
            in patch sizes, leading to more diverse and larger mixed patches.
            Defaults to 1.
        seed: Integer. Used to create a random seed.

    References:
       - [CutMix paper]( https://arxiv.org/abs/1905.04899).
    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)

    def __init__(self, factor=0.3, seed=None, data_format=None, **kwargs):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self.seed = seed
        self.generator = SeedGenerator(seed)

        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
            self.channel_axis = -3
        else:
            self.height_axis = -3
            self.width_axis = -2
            self.channel_axis = -1

    def transform_images(self, images, transformation=None, training=True):

        batch_size, image_height, image_width, image_channel = (
            images.shape[0],
            images.shape[self.height_axis],
            images.shape[self.width_axis],
            images.shape[self.channel_axis])

        area = image_height * image_width

        scale = (0., 1.)
        ratio = (0., 1.)

        min_area = area * scale[0]
        max_area = area * scale[1]
        min_aspect_ratio = ratio[0]
        max_aspect_ratio = ratio[1]

        target_area = self.backend.random.uniform((),
                                                  min_area,
                                                  max_area,
                                                  dtype=self.compute_dtype)

        aspect_ratio = self.backend.random.uniform((),
                                                   min_aspect_ratio,
                                                   max_aspect_ratio,
                                                   dtype=self.compute_dtype)

        h = self.backend.cast(self.backend.numpy.sqrt(target_area * aspect_ratio), dtype='int32')
        w = self.backend.cast(self.backend.numpy.sqrt(target_area / aspect_ratio), dtype='int32')

        x = self.backend.random.randint((), 0, image_height - h)
        y = self.backend.random.randint((), 0, image_width - w)

        v = self.backend.random.normal(shape=[h, w, image_channel])

        images = self.backend.convert_to_numpy(images)
        images[..., x:x + h, y:y + w, :] = v

        tf.print(x, y, w, h)
        images = self.backend.convert_to_tensor(images)
        images = self.backend.cast(images, self.compute_dtype)
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

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
