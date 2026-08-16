from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.ContrastLimitedAdaptiveHistogramEqualization")
class ContrastLimitedAdaptiveHistogramEqualization(BaseImagePreprocessingLayer):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) layer.

    CLAHE is a variant of Adaptive Histogram Equalization (AHE) which takes care
    of over-amplification of the contrast. It operates on small regions in the
    image, called tiles, rather than the entire image. The neighboring tiles are
    then combined using bilinear interpolation to remove the artificial
    boundaries. This algorithm can be applied to improve the contrast of an
    image.

    **Note:** This layer computes histograms using `self.backend.nn.one_hot`,
    which can be highly memory-intensive. For large batch sizes or
    high-resolution images, it may lead to high memory consumption or
    out-of-memory errors.

    Args:
        value_range: Optional list/tuple of 2 floats specifying the lower
            and upper limits of the input data values. Defaults to `(0, 255)`.
        clip_limit: Float. Limits the noise amplification in near-constant
            regions. Defaults to 4.0.
        tile_grid_size: Tuple of 2 integers `(height, width)`.
            Specifies the number of tiles to divide the image into.
            Defaults to `(8, 8)`.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format,
        or `(..., channels, height, width)`, in `"channels_first"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,
        or `(..., channels, height, width)`,
        in `"channels_first"` format.

    Example:

    ```python
    import keras
    import numpy as np

    # Create a CLAHE layer with default parameters
    clahe = keras.layers.ContrastLimitedAdaptiveHistogramEqualization()

    # Apply CLAHE to an image
    # image values should be in the range[0, 255] by default
    input_image = np.random.randint(0, 256, (1, 256, 256, 3))
    output_image = clahe(input_image)

    # For normalized images[0, 1]
    clahe_normalized=keras.layers.ContrastLimitedAdaptiveHistogramEqualization(
        value_range=(0.0, 1.0)
    )
    norm_image = np.random.rand(1, 256, 256, 3)
    output_norm = clahe_normalized(norm_image)
    ```
    """

    def __init__(
        self,
        value_range=(0, 255),
        clip_limit=4.0,
        tile_grid_size=(8, 8),
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = value_range
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.data_format = backend.standardize_data_format(data_format)

    def transform_images(self, images, transformation=None, training=True):
        if self.data_format == "channels_first":
            if len(images.shape) == 4:
                images = self.backend.numpy.transpose(images, (0, 2, 3, 1))
            else:
                images = self.backend.numpy.transpose(images, (1, 2, 0))

        original_dtype = images.dtype

        images = self._transform_value_range(
            images, self.value_range, (0, 255), dtype="float32"
        )

        images = self._clahe(images)

        images = self._transform_value_range(
            images, (0, 255), self.value_range, dtype="float32"
        )
        images = self.backend.cast(images, original_dtype)

        if self.data_format == "channels_first":
            if len(images.shape) == 4:
                images = self.backend.numpy.transpose(images, (0, 3, 1, 2))
            else:
                images = self.backend.numpy.transpose(images, (2, 0, 1))

        return images

    def _clahe(self, images):
        unbatched = False
        if len(images.shape) == 3:
            images = self.backend.numpy.expand_dims(images, axis=0)
            unbatched = True

        shape = self.backend.core.shape(images)
        batch_size = (
            images.shape[0] if images.shape[0] is not None else shape[0]
        )
        height = images.shape[1] if images.shape[1] is not None else shape[1]
        width = images.shape[2] if images.shape[2] is not None else shape[2]
        channels = images.shape[3] if images.shape[3] is not None else shape[3]

        grid_h, grid_w = self.tile_grid_size

        tile_h = (height + grid_h - 1) // grid_h
        tile_w = (width + grid_w - 1) // grid_w

        pad_h = (tile_h * grid_h) - height
        pad_w = (tile_w * grid_w) - width

        if (
            isinstance(pad_h, int)
            and isinstance(pad_w, int)
            and pad_h == 0
            and pad_w == 0
        ):
            padded_images = images
        else:
            images_nchw = self.backend.numpy.transpose(images, (0, 3, 1, 2))

            images_3d = self.backend.numpy.reshape(
                images_nchw, (-1, height, width)
            )
            padded_3d = self.backend.numpy.pad(
                images_3d, [[0, 0], [0, pad_h], [0, pad_w]], mode="symmetric"
            )
            padded_nchw = self.backend.numpy.reshape(
                padded_3d, (-1, channels, height + pad_h, width + pad_w)
            )
            padded_images = self.backend.numpy.transpose(
                padded_nchw, (0, 2, 3, 1)
            )

        # Compute Histograms per tile
        tiled = self.backend.numpy.reshape(
            padded_images,
            (batch_size, grid_h, tile_h, grid_w, tile_w, channels),
        )
        tiled = self.backend.numpy.transpose(tiled, (0, 1, 3, 5, 2, 4))

        tiled_flat = self.backend.numpy.reshape(
            tiled, (batch_size, grid_h, grid_w, channels, tile_h * tile_w)
        )

        tiled_int = self.backend.cast(tiled_flat, "int32")
        tiled_int = self.backend.numpy.clip(tiled_int, 0, 255)

        # Compute histograms via one_hot and sum
        hists = self.backend.numpy.sum(
            self.backend.nn.one_hot(tiled_int, 256), axis=-2
        )

        # Clip and redistribute
        if self.clip_limit > 0:
            limit = self.clip_limit * (tile_h * tile_w) / 256.0
            limit = self.backend.cast(limit, hists.dtype)

            clipped = self.backend.numpy.clip(hists, 0, limit)

            excess = self.backend.numpy.sum(
                hists - clipped, axis=-1, keepdims=True
            )
            redist = excess / 256.0
            hists = clipped + redist

        # Compute CDF
        cdf = self.backend.numpy.cumsum(hists, axis=-1)
        cdf_min = self.backend.numpy.min(cdf, axis=-1, keepdims=True)

        numerator = (cdf - cdf_min) * 255.0
        denominator = self.backend.cast(tile_h * tile_w, cdf.dtype) - cdf_min

        denominator = self.backend.numpy.where(
            denominator == 0,
            self.backend.numpy.ones_like(denominator),
            denominator,
        )
        cdf_norm = numerator / denominator
        cdf_norm = self.backend.numpy.clip(cdf_norm, 0, 255)

        # Interpolation

        top = cdf_norm[:, 0:1, :, :, :]
        bottom = cdf_norm[:, -1:, :, :, :]
        cdf_padded = self.backend.numpy.concatenate(
            [top, cdf_norm, bottom], axis=1
        )

        left = cdf_padded[:, :, 0:1, :, :]
        right = cdf_padded[:, :, -1:, :, :]
        cdf_padded = self.backend.numpy.concatenate(
            [left, cdf_padded, right], axis=2
        )

        H_padded = tile_h * grid_h
        W_padded = tile_w * grid_w

        y_range = self.backend.numpy.arange(H_padded, dtype="float32")
        x_range = self.backend.numpy.arange(W_padded, dtype="float32")

        y_grid = (y_range - (tile_h / 2.0)) / tile_h
        x_grid = (x_range - (tile_w / 2.0)) / tile_w

        y_grid = y_grid + 1.0
        x_grid = x_grid + 1.0

        y0 = self.backend.numpy.floor(y_grid)
        x0 = self.backend.numpy.floor(x_grid)
        y1 = y0 + 1.0
        x1 = x0 + 1.0

        wy = y_grid - y0
        wx = x_grid - x0

        y0 = self.backend.numpy.clip(y0, 0, grid_h + 1)
        y1 = self.backend.numpy.clip(y1, 0, grid_h + 1)
        x0 = self.backend.numpy.clip(x0, 0, grid_w + 1)
        x1 = self.backend.numpy.clip(x1, 0, grid_w + 1)

        y0 = self.backend.cast(y0, "int32")
        y1 = self.backend.cast(y1, "int32")
        x0 = self.backend.cast(x0, "int32")
        x1 = self.backend.cast(x1, "int32")

        stride_c = 256
        stride_x = stride_c * channels
        stride_y = stride_x * (grid_w + 2)
        stride_b = stride_y * (grid_h + 2)

        cdf_flat = self.backend.numpy.reshape(cdf_padded, (-1,))

        pixels = self.backend.cast(
            self.backend.numpy.clip(padded_images, 0, 255), "int32"
        )

        b_idx = self.backend.numpy.arange(batch_size, dtype="int32")[
            :, None, None, None
        ]

        c_idx = self.backend.numpy.arange(channels, dtype="int32")[
            None, None, None, :
        ]

        y0_e = y0[None, :, None, None]
        y1_e = y1[None, :, None, None]

        x0_e = x0[None, None, :, None]
        x1_e = x1[None, None, :, None]

        wy_e = wy[None, :, None, None]
        wx_e = wx[None, None, :, None]

        base_idx = b_idx * stride_b + c_idx * stride_c + pixels

        idx_nw = base_idx + y0_e * stride_y + x0_e * stride_x
        val_nw = self.backend.numpy.take(cdf_flat, idx_nw)

        idx_ne = base_idx + y0_e * stride_y + x1_e * stride_x
        val_ne = self.backend.numpy.take(cdf_flat, idx_ne)

        idx_sw = base_idx + y1_e * stride_y + x0_e * stride_x
        val_sw = self.backend.numpy.take(cdf_flat, idx_sw)

        idx_se = base_idx + y1_e * stride_y + x1_e * stride_x
        val_se = self.backend.numpy.take(cdf_flat, idx_se)

        top_interp = val_nw * (1.0 - wx_e) + val_ne * wx_e
        bot_interp = val_sw * (1.0 - wx_e) + val_se * wx_e
        result = top_interp * (1.0 - wy_e) + bot_interp * wy_e

        result = result[:, :height, :width, :]

        if unbatched:
            result = self.backend.numpy.squeeze(result, axis=0)

        return result

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "clip_limit": self.clip_limit,
                "tile_grid_size": self.tile_grid_size,
                "data_format": self.data_format,
            }
        )
        return config
