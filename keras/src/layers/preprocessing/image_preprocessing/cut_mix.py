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
        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data

        images_shape = self.backend.shape(images)

        if len(images_shape) == 3:
            batch_size = 1
        else:
            batch_size = self.backend.shape(images)[0]

        seed = seed or self._get_seed_generator(self.backend._backend)

        permutation_order = self.backend.random.shuffle(
            self.backend.numpy.arange(0, batch_size, dtype="int64"),
            seed=seed,
        )

        mix_weight = self.backend.random.beta(
            (batch_size,), self.alpha, self.alpha, seed=seed
        )
        return {
            "mix_weight": mix_weight,
            "permutation_order": permutation_order,
        }

    def transform_images(self, images, transformation=None, training=True):
        if training:
            images = self._cutmix(images)
        return images

    def _cutmix(self, images):
        def _axis_mask(starts, ends, mask_len):
            batch_size = self.backend.shape(starts)[0]
            axis_indices = self.backend.numpy.arange(
                0, mask_len, dtype=starts.dtype
            )
            axis_indices = self.backend.numpy.expand_dims(axis_indices, 0)
            axis_indices = self.backend.numpy.tile(
                axis_indices, [batch_size, 1]
            )

            axis_mask = self.backend.numpy.greater_equal(
                axis_indices, starts
            ) & self.backend.numpy.less(axis_indices, ends)
            return axis_mask

        def corners_to_mask(bounding_boxes, mask_shape):
            mask_width, mask_height = mask_shape
            x0, y0, x1, y1 = self.backend.numpy.split(
                bounding_boxes, 4, axis=-1
            )

            w_mask = _axis_mask(x0, x1, mask_width)
            h_mask = _axis_mask(y0, y1, mask_height)

            w_mask = self.backend.numpy.expand_dims(w_mask, axis=1)
            h_mask = self.backend.numpy.expand_dims(h_mask, axis=2)
            masks = self.backend.numpy.logical_and(w_mask, h_mask)
            return masks

        def fill_rectangle(
            images, centers_x, centers_y, widths, heights, fill_values
        ):
            images_shape = self.backend.cast(
                self.backend.shape(images), dtype=self.compute_dtype
            )
            images_height = images_shape[1]
            images_width = images_shape[2]

            xywh = self.backend.numpy.stack(
                [centers_x, centers_y, widths, heights], axis=1
            )
            corners = convert_format(xywh, source="center_xywh", target="xyxy")
            mask_shape = (images_width, images_height)
            is_rectangle = corners_to_mask(corners, mask_shape)
            is_rectangle = self.backend.numpy.expand_dims(is_rectangle, -1)

            images = self.backend.numpy.where(is_rectangle, fill_values, images)
            return images

        input_shape = self.backend.shape(images)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        permutation_order = self.backend.random.shuffle(
            self.backend.numpy.arange(0, batch_size), seed=self.seed
        )
        lambda_sample = self.backend.random.beta(
            (batch_size,),
            self.alpha,
            self.alpha,
        )

        ratio = self.backend.numpy.sqrt(1 - lambda_sample)
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

        bounding_box_area = cut_height * cut_width
        lambda_sample = 1.0 - bounding_box_area / (image_height * image_width)
        lambda_sample = self.backend.cast(
            lambda_sample, dtype=self.compute_dtype
        )

        images = fill_rectangle(
            images,
            random_center_width,
            random_center_height,
            cut_width,
            cut_height,
            self.backend.numpy.take(images, permutation_order, axis=0),
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
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
