from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
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
            channel_axis = 1
        else:
            image_height = images_shape[-3]
            image_width = images_shape[-2]
            channel_axis = -1

        seed = seed or self._get_seed_generator(self.backend._backend)

        r_x = self.backend.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_width,
            dtype=self.compute_dtype,
            seed=seed,
        )
        r_y = self.backend.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_height,
            dtype=self.compute_dtype,
            seed=seed,
        )

        mix_weight = self.backend.random.beta(
            (batch_size,), self.alpha, self.alpha, seed=seed
        )

        batch_masks, mix_weight = self.get_batch_mask(
            channel_axis,
            image_height,
            image_width,
            images_shape,
            r_x,
            r_y,
            mix_weight,
        )

        permutation_order = self.backend.random.shuffle(
            self.backend.numpy.arange(0, batch_size, dtype="int32"),
            seed=seed,
        )

        return {
            "permutation_order": permutation_order,
            "batch_masks": batch_masks,
            "mix_weight": mix_weight,
        }

    def get_batch_mask(
        self,
        channel_axis,
        image_height,
        image_width,
        images_shape,
        r_x,
        r_y,
        mix_weight,
    ):
        ratio = 0.5 * self.backend.numpy.sqrt(1.0 - mix_weight)
        r_w_half = self.backend.cast(
            ratio * image_width, dtype=self.compute_dtype
        )
        r_h_half = self.backend.cast(
            ratio * image_height, dtype=self.compute_dtype
        )
        x0 = self.backend.numpy.clip(r_x - r_w_half, 0, image_width)
        y0 = self.backend.numpy.clip(r_y - r_h_half, 0, image_height)
        x1 = self.backend.numpy.clip(r_x + r_w_half, 0, image_width)
        y1 = self.backend.numpy.clip(r_y + r_h_half, 0, image_height)
        grid_y, grid_x = self.backend.numpy.meshgrid(
            self.backend.numpy.arange(image_height, dtype=self.compute_dtype),
            self.backend.numpy.arange(image_width, dtype=self.compute_dtype),
            indexing="ij",
        )

        if self.data_format == "channels_last":
            grid_y = self.backend.cast(
                grid_y[None, :, :, None], dtype=self.compute_dtype
            )
            grid_x = self.backend.cast(
                grid_x[None, :, :, None], dtype=self.compute_dtype
            )
        else:
            grid_y = self.backend.cast(
                grid_y[None, None, :, :], dtype=self.compute_dtype
            )
            grid_x = self.backend.cast(
                grid_x[None, None, :, :], dtype=self.compute_dtype
            )

        x0 = x0[:, None, None, None]
        y0 = y0[:, None, None, None]
        x1 = x1[:, None, None, None]
        y1 = y1[:, None, None, None]
        batch_masks = (
            (grid_x >= x0) & (grid_x < x1) & (grid_y >= y0) & (grid_y < y1)
        )
        batch_masks = self.backend.numpy.repeat(
            batch_masks, images_shape[channel_axis], axis=channel_axis
        )
        mix_weight = 1.0 - (x1 - x0) * (y1 - y0) / (image_width * image_height)
        return batch_masks, mix_weight

    def transform_images(self, images, transformation=None, training=True):
        if training and transformation is not None:
            images = self.backend.cast(images, self.compute_dtype)

            permutation_order = transformation["permutation_order"]
            batch_masks = transformation["batch_masks"]

            images = self.backend.numpy.where(
                batch_masks,
                self.backend.numpy.take(images, permutation_order, axis=0),
                images,
            )
        images = self.backend.cast(images, self.compute_dtype)
        return images

    def transform_labels(self, labels, transformation, training=True):
        if training and transformation is not None:
            permutation_order = transformation["permutation_order"]
            mix_weight = transformation["mix_weight"]

            cutout_labels = self.backend.numpy.take(
                labels, permutation_order, axis=0
            )
            mix_weight = self.backend.numpy.reshape(mix_weight, [-1, 1])
            labels = mix_weight * labels + (1.0 - mix_weight) * cutout_labels

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
