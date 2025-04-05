from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.Equalization")
class Equalization(BaseImagePreprocessingLayer):
    """Preprocessing layer for histogram equalization on image channels.

    Histogram equalization is a technique to adjust image intensities to
    enhance contrast by effectively spreading out the most frequent
    intensity values. This layer applies equalization on a channel-wise
    basis, which can improve the visibility of details in images.

    This layer works with both grayscale and color images, performing
    equalization independently on each color channel. At inference time,
    the equalization is consistently applied.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        value_range: Optional list/tuple of 2 floats specifying the lower
            and upper limits of the input data values. Defaults to `[0, 255]`.
            If the input image has been scaled, use the appropriate range
            (e.g., `[0.0, 1.0]`). The equalization will be scaled to this
            range, and output values will be clipped accordingly.
        bins: Integer specifying the number of histogram bins to use for
            equalization. Defaults to 256, which is suitable for 8-bit images.
            Larger values can provide more granular intensity redistribution.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format,
        or `(..., channels, height, width)`, in `"channels_first"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`,
        or `(..., channels, target_height, target_width)`,
        in `"channels_first"` format.

    Example:

    ```python
    # Create an equalization layer for standard 8-bit images
    equalizer = keras.layers.Equalization()

    # An image with uneven intensity distribution
    image = [...] # your input image

    # Apply histogram equalization
    equalized_image = equalizer(image)

    # For images with custom value range
    custom_equalizer = keras.layers.Equalization(
        value_range=[0.0, 1.0],  # for normalized images
        bins=128  # fewer bins for more subtle equalization
    )
    custom_equalized = custom_equalizer(normalized_image)
    ```
    """

    def __init__(
        self, value_range=(0, 255), bins=256, data_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.bins = bins
        self._set_value_range(value_range)
        self.data_format = backend.standardize_data_format(data_format)

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        if len(value_range) != 2:
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        self.value_range = sorted(value_range)

    def _custom_histogram_fixed_width(self, values, value_range, nbins):
        values = self.backend.cast(values, "float32")
        value_min, value_max = value_range
        value_min = self.backend.cast(value_min, "float32")
        value_max = self.backend.cast(value_max, "float32")

        scaled = (values - value_min) * (nbins - 1) / (value_max - value_min)
        indices = self.backend.cast(scaled, "int32")
        indices = self.backend.numpy.clip(indices, 0, nbins - 1)
        flat_indices = self.backend.numpy.reshape(indices, [-1])

        if backend.backend() == "jax":
            # for JAX bincount is never jittable because of output shape
            histogram = self.backend.numpy.zeros(nbins, dtype="int32")
            for i in range(nbins):
                matches = self.backend.cast(
                    self.backend.numpy.equal(flat_indices, i), "int32"
                )
                bin_count = self.backend.numpy.sum(matches)
                one_hot = self.backend.cast(
                    self.backend.numpy.arange(nbins) == i, "int32"
                )
                histogram = histogram + (bin_count * one_hot)
            return histogram
        else:
            # TensorFlow/PyTorch/NumPy implementation using bincount
            return self.backend.numpy.bincount(
                flat_indices,
                minlength=nbins,
            )

    def _scale_values(self, values, source_range, target_range):
        source_min, source_max = source_range
        target_min, target_max = target_range
        scale = (target_max - target_min) / (source_max - source_min)
        offset = target_min - source_min * scale
        return values * scale + offset

    def _equalize_channel(self, channel, value_range):
        if value_range != (0, 255):
            channel = self._scale_values(channel, value_range, (0, 255))

        hist = self._custom_histogram_fixed_width(
            channel, value_range=(0, 255), nbins=self.bins
        )

        nonzero_bins = self.backend.numpy.count_nonzero(hist)
        equalized = self.backend.numpy.where(
            nonzero_bins <= 1, channel, self._apply_equalization(channel, hist)
        )

        if value_range != (0, 255):
            equalized = self._scale_values(equalized, (0, 255), value_range)

        return equalized

    def _apply_equalization(self, channel, hist):
        cdf = self.backend.numpy.cumsum(hist)

        if self.backend.name == "jax":
            mask = cdf > 0
            first_nonzero_idx = self.backend.numpy.argmax(mask)
            cdf_min = self.backend.numpy.take(cdf, first_nonzero_idx)
        else:
            cdf_min = self.backend.numpy.take(
                cdf, self.backend.numpy.nonzero(cdf)[0][0]
            )

        denominator = cdf[-1] - cdf_min
        denominator = self.backend.numpy.where(
            denominator == 0,
            self.backend.numpy.ones_like(1, dtype=denominator.dtype),
            denominator,
        )

        lookup_table = ((cdf - cdf_min) * 255) / denominator
        lookup_table = self.backend.numpy.clip(
            self.backend.numpy.round(lookup_table), 0, 255
        )

        scaled_channel = (channel / 255.0) * (self.bins - 1)
        indices = self.backend.cast(
            self.backend.numpy.clip(scaled_channel, 0, self.bins - 1), "int32"
        )
        return self.backend.numpy.take(lookup_table, indices)

    def transform_images(self, images, transformation, training=True):
        if training:
            images = self.backend.cast(images, self.compute_dtype)

            if self.data_format == "channels_first":
                channels = []
                for i in range(self.backend.core.shape(images)[-3]):
                    channel = images[..., i, :, :]
                    equalized = self._equalize_channel(
                        channel, self.value_range
                    )
                    channels.append(equalized)
                equalized_images = self.backend.numpy.stack(channels, axis=-3)
            else:
                channels = []
                for i in range(self.backend.core.shape(images)[-1]):
                    channel = images[..., i]
                    equalized = self._equalize_channel(
                        channel, self.value_range
                    )
                    channels.append(equalized)
                equalized_images = self.backend.numpy.stack(channels, axis=-1)

            return self.backend.cast(equalized_images, self.compute_dtype)
        return images

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_spec(self, inputs, **kwargs):
        return inputs

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        return bounding_boxes

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks

    def get_config(self):
        config = super().get_config()
        config.update({"bins": self.bins, "value_range": self.value_range})
        return config
