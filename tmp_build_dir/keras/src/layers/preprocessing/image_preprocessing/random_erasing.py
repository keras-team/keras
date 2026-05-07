from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator


@keras_export("keras.layers.RandomErasing")
class RandomErasing(BaseImagePreprocessingLayer):
    """Random Erasing data augmentation technique.

    Random Erasing is a data augmentation method where random patches of
    an image are erased (replaced by a constant value or noise)
    during training to improve generalization.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    References:
       - [Random Erasing paper](https://arxiv.org/abs/1708.04896).

    Args:
        factor: A single float or a tuple of two floats.
            `factor` controls the probability of applying the transformation.
            - `factor=0.0` ensures no erasing is applied.
            - `factor=1.0` means erasing is always applied.
            - If a tuple `(min, max)` is provided, a probability value
              is sampled between `min` and `max` for each image.
            - If a single float is provided, a probability is sampled
              between `0.0` and the given float.
            Default is 1.0.
        scale: A tuple of two floats representing the aspect ratio range of
            the erased patch. This defines the width-to-height ratio of
            the patch to be erased. It can help control the rw shape of
            the erased region. Default is (0.02, 0.33).
        fill_value: A value to fill the erased region with. This can be set to
            a constant value or `None` to sample a random value
            from a normal distribution. Default is `None`.
        value_range: the range of values the incoming images will have.
            Represented as a two-number tuple written `[low, high]`. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        seed: Integer. Used to create a random seed.
    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)

    def __init__(
        self,
        factor=1.0,
        scale=(0.02, 0.33),
        fill_value=None,
        value_range=(0, 255),
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self.scale = self._set_factor_by_name(scale, "scale")
        self.fill_value = fill_value
        self.value_range = value_range
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

    def _set_factor_by_name(self, factor, name):
        error_msg = (
            f"The `{name}` argument should be a number "
            "(or a list of two numbers) "
            "in the range "
            f"[{self._FACTOR_BOUNDS[0]}, {self._FACTOR_BOUNDS[1]}]. "
            f"Received: factor={factor}"
        )
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(error_msg)
            if (
                factor[0] > self._FACTOR_BOUNDS[1]
                or factor[1] < self._FACTOR_BOUNDS[0]
            ):
                raise ValueError(error_msg)
            lower, upper = sorted(factor)
        elif isinstance(factor, (int, float)):
            if (
                factor < self._FACTOR_BOUNDS[0]
                or factor > self._FACTOR_BOUNDS[1]
            ):
                raise ValueError(error_msg)
            factor = abs(factor)
            lower, upper = [max(-factor, self._FACTOR_BOUNDS[0]), factor]
        else:
            raise ValueError(error_msg)
        return lower, upper

    def _compute_crop_bounds(self, batch_size, image_length, crop_ratio, seed):
        crop_length = self.backend.cast(
            crop_ratio * image_length, dtype=self.compute_dtype
        )

        start_pos = self.backend.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=1,
            dtype=self.compute_dtype,
            seed=seed,
        ) * (image_length - crop_length)

        end_pos = start_pos + crop_length

        return start_pos, end_pos

    def _generate_batch_mask(self, images_shape, box_corners):
        def _generate_grid_xy(image_height, image_width):
            grid_y, grid_x = self.backend.numpy.meshgrid(
                self.backend.numpy.arange(
                    image_height, dtype=self.compute_dtype
                ),
                self.backend.numpy.arange(
                    image_width, dtype=self.compute_dtype
                ),
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
            return grid_x, grid_y

        image_height, image_width = (
            images_shape[self.height_axis],
            images_shape[self.width_axis],
        )
        grid_x, grid_y = _generate_grid_xy(image_height, image_width)

        x0, x1, y0, y1 = box_corners

        x0 = x0[:, None, None, None]
        y0 = y0[:, None, None, None]
        x1 = x1[:, None, None, None]
        y1 = y1[:, None, None, None]

        batch_masks = (
            (grid_x >= x0) & (grid_x < x1) & (grid_y >= y0) & (grid_y < y1)
        )
        batch_masks = self.backend.numpy.repeat(
            batch_masks, images_shape[self.channel_axis], axis=self.channel_axis
        )

        return batch_masks

    def _get_fill_value(self, images, images_shape, seed):
        fill_value = self.fill_value
        if fill_value is None:
            fill_value = (
                self.backend.random.normal(
                    images_shape,
                    dtype=self.compute_dtype,
                    seed=seed,
                )
                * self.value_range[1]
            )
        else:
            error_msg = (
                "The `fill_value` argument should be a number "
                "(or a list of three numbers) "
            )
            if isinstance(fill_value, (tuple, list)):
                if len(fill_value) != 3:
                    raise ValueError(error_msg)
                fill_value = self.backend.numpy.full_like(
                    images, fill_value, dtype=self.compute_dtype
                )
            elif isinstance(fill_value, (int, float)):
                fill_value = (
                    self.backend.numpy.ones(
                        images_shape, dtype=self.compute_dtype
                    )
                    * fill_value
                )
            else:
                raise ValueError(error_msg)
        fill_value = self.backend.numpy.clip(
            fill_value, self.value_range[0], self.value_range[1]
        )
        return fill_value

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data

        images_shape = self.backend.shape(images)
        rank = len(images_shape)
        if rank == 3:
            batch_size = 1
        elif rank == 4:
            batch_size = images_shape[0]
        else:
            raise ValueError(
                "Expected the input image to be rank 3 or 4. Received "
                f"inputs.shape={images_shape}"
            )

        image_height = images_shape[self.height_axis]
        image_width = images_shape[self.width_axis]

        seed = seed or self._get_seed_generator(self.backend._backend)

        mix_weight = self.backend.random.uniform(
            shape=(batch_size, 2),
            minval=self.scale[0],
            maxval=self.scale[1],
            dtype=self.compute_dtype,
            seed=seed,
        )

        mix_weight = self.backend.numpy.sqrt(mix_weight)

        x0, x1 = self._compute_crop_bounds(
            batch_size, image_width, mix_weight[:, 0], seed
        )
        y0, y1 = self._compute_crop_bounds(
            batch_size, image_height, mix_weight[:, 1], seed
        )

        batch_masks = self._generate_batch_mask(
            images_shape,
            (x0, x1, y0, y1),
        )

        erase_probability = self.backend.random.uniform(
            shape=(batch_size,),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=seed,
        )

        random_threshold = self.backend.random.uniform(
            shape=(batch_size,),
            minval=0.0,
            maxval=1.0,
            seed=seed,
        )
        apply_erasing = random_threshold < erase_probability

        fill_value = self._get_fill_value(images, images_shape, seed)

        return {
            "apply_erasing": apply_erasing,
            "batch_masks": batch_masks,
            "fill_value": fill_value,
        }

    def transform_images(self, images, transformation=None, training=True):
        if training:
            images = self.backend.cast(images, self.compute_dtype)
            batch_masks = transformation["batch_masks"]
            apply_erasing = transformation["apply_erasing"]
            fill_value = transformation["fill_value"]

            erased_images = self.backend.numpy.where(
                batch_masks,
                fill_value,
                images,
            )

            images = self.backend.numpy.where(
                apply_erasing[:, None, None, None],
                erased_images,
                images,
            )

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
            "scale": self.scale,
            "fill_value": self.fill_value,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
