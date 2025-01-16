from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    convert_format,
)
from keras.src.random import SeedGenerator


@keras_export("keras.layers.CutMix")
class CutMix(BaseImagePreprocessingLayer):
    """CutMix implements the CutMix data augmentation technique.

    Args:
        alpha: Float between 0 and 1. Inverse scale parameter for the gamma
            distribution. This controls the shape of the distribution from which
            the smoothing values are sampled. Defaults to 1.0, which is a
            recommended value when training an imagenet1k classification model.
        seed: Integer. Used to create a random seed.

    References:
       - [CutMix paper]( https://arxiv.org/abs/1905.04899).
    """

    def __init__(self, alpha=1.0, seed=None, data_format=None, **kwargs):
        super().__init__(data_format=data_format, **kwargs)
        self.alpha = alpha
        self.seed = seed
        self.generator = SeedGenerator(seed)

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data

        images_shape = self.backend.shape(images)
        if len(images_shape) == 3:
            return None

        batch_size = images_shape[0]
        if self.data_format == "channels_first":
            image_height = images_shape[-2]
            image_width = images_shape[-1]
        else:
            image_height = images_shape[-3]
            image_width = images_shape[-2]

        seed = seed or self._get_seed_generator(self.backend._backend)

        permutation_order = self.backend.random.shuffle(
            self.backend.numpy.arange(0, batch_size, dtype="int64"),
            seed=seed,
        )

        mix_weight = self.backend.random.beta(
            (batch_size,), self.alpha, self.alpha, seed=seed
        )

        ratio = self.backend.numpy.sqrt(1 - mix_weight)
        cut_height = self.backend.cast(
            ratio * image_height, dtype=self.compute_dtype
        )
        cut_width = self.backend.cast(
            ratio * image_width, dtype=self.compute_dtype
        )

        random_center_height = self.backend.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_height,
            dtype=self.compute_dtype,
        )
        random_center_width = self.backend.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_width,
            dtype=self.compute_dtype,
        )

        return {
            "permutation_order": permutation_order,
            "cut_height": cut_height,
            "cut_width": cut_width,
            "random_center_height": random_center_height,
            "random_center_width": random_center_width,
            "input_shape": (batch_size, image_height, image_width),
        }

    def transform_images(self, images, transformation=None, training=True):
        if training:
            if transformation is not None:
                images = self._cut_mix(images, transformation)
        images = self.backend.cast(images, self.compute_dtype)
        return images

    def _cut_mix(self, images, transformation):
        def _axis_mask(starts, ends, mask_len, batch_size):
            axis_indices = self.backend.numpy.arange(0, mask_len)
            axis_indices = self.backend.numpy.expand_dims(axis_indices, 0)
            axis_indices = self.backend.numpy.tile(
                axis_indices, [batch_size, 1]
            )

            axis_mask = self.backend.numpy.greater_equal(
                axis_indices, starts
            ) & self.backend.numpy.less(axis_indices, ends)
            return axis_mask

        def corners_to_mask(bounding_boxes, mask_shape):
            batch_size, mask_height, mask_width = mask_shape
            x0, y0, x1, y1 = self.backend.numpy.split(
                bounding_boxes, 4, axis=-1
            )

            w_mask = _axis_mask(x0, x1, mask_width, batch_size)
            h_mask = _axis_mask(y0, y1, mask_height, batch_size)

            w_mask = self.backend.numpy.expand_dims(w_mask, axis=1)
            h_mask = self.backend.numpy.expand_dims(h_mask, axis=2)
            masks = self.backend.numpy.logical_and(w_mask, h_mask)
            return masks

        images = self.backend.cast(images, self.compute_dtype)

        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1

        permutation_order = transformation["permutation_order"]
        random_center_width = transformation["random_center_width"]
        random_center_height = transformation["random_center_height"]
        cut_width = transformation["cut_width"]
        cut_height = transformation["cut_height"]
        input_shape = transformation["input_shape"]

        xywh = self.backend.numpy.stack(
            [random_center_width, random_center_height, cut_width, cut_height],
            axis=1,
        )
        corners = convert_format(xywh, source="center_xywh", target="xyxy")
        is_rectangle = corners_to_mask(corners, input_shape)
        is_rectangle = self.backend.numpy.expand_dims(
            is_rectangle, channel_axis
        )

        images = self.backend.numpy.where(
            is_rectangle,
            self.backend.numpy.take(images, permutation_order, axis=0),
            images,
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
        raise NotImplementedError()

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
