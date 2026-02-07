from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.CLAHE")
class CLAHE(BaseImagePreprocessingLayer):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) layer.

    CLAHE is a variant of Adaptive Histogram Equalization (AHE) which takes care
    of over-amplification of the contrast. It operates on small regions in the
    image, called tiles, rather than the entire image. The neighboring tiles are
    then combined using bilinear interpolation to remove the artificial
    boundaries. This algorithm can be applied to improve the contrast of an
    image.

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
        `(..., target_height, target_width, channels)`,
        or `(..., channels, target_height, target_width)`,
        in `"channels_first"` format.

    Example:

    ```python
    # Create a CLAHE layer with default parameters
    clahe = keras.layers.CLAHE()

    # Apply CLAHE to an image
    # image values should be in the range [0, 255] by default
    input_image = np.random.randint(0, 256, (1, 256, 256, 3))
    output_image = clahe(input_image)

    # For normalized images [0, 1]
    clahe_normalized = keras.layers.CLAHE(value_range=(0.0, 1.0))
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
            if len(ops.shape(images)) == 4:
                images = ops.transpose(images, (0, 2, 3, 1))
            else:
                images = ops.transpose(images, (1, 2, 0))

        original_dtype = images.dtype

        images = self._transform_value_range(
            images, self.value_range, (0, 255), dtype="float32"
        )

        images = self._clahe(images)

        images = self._transform_value_range(
            images, (0, 255), self.value_range, dtype="float32"
        )
        images = ops.cast(images, original_dtype)

        if self.data_format == "channels_first":
            if len(ops.shape(images)) == 4:
                images = ops.transpose(images, (0, 3, 1, 2))
            else:
                images = ops.transpose(images, (2, 0, 1))

        return images

    def _clahe(self, images):
        unbatched = False
        if len(ops.shape(images)) == 3:
            images = ops.expand_dims(images, axis=0)
            unbatched = True

        batch_size = ops.shape(images)[0]
        height = ops.shape(images)[1]
        width = ops.shape(images)[2]
        channels = ops.shape(images)[3]

        grid_h, grid_w = self.tile_grid_size

        tile_h = (height + grid_h - 1) // grid_h
        tile_w = (width + grid_w - 1) // grid_w

        pad_h = (tile_h * grid_h) - height
        pad_w = (tile_w * grid_w) - width

        padded_images = ops.pad(
            images, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]], mode="reflect"
        )

        # Compute Histograms per tile

        tiled = ops.reshape(
            padded_images,
            (batch_size, grid_h, tile_h, grid_w, tile_w, channels),
        )
        tiled = ops.transpose(tiled, (0, 1, 3, 5, 2, 4))

        tiled_flat = ops.reshape(
            tiled, (batch_size, grid_h, grid_w, channels, tile_h * tile_w)
        )

        tiled_int = ops.cast(tiled_flat, "int32")
        tiled_int = ops.clip(tiled_int, 0, 255)

        # Compute histograms via one_hot and sum
        hists = ops.sum(ops.one_hot(tiled_int, 256), axis=-2)

        # Clip and redistribute
        if self.clip_limit > 0:
            limit = self.clip_limit * (tile_h * tile_w) / 256.0
            limit = ops.cast(limit, hists.dtype)

            clipped = ops.clip(hists, 0, limit)

            excess = ops.sum(hists - clipped, axis=-1, keepdims=True)
            redist = excess / 256.0
            hists = clipped + redist

        # Compute CDF
        cdf = ops.cumsum(hists, axis=-1)
        cdf_min = ops.min(cdf, axis=-1, keepdims=True)

        numerator = (cdf - cdf_min) * 255.0
        denominator = ops.cast(tile_h * tile_w, cdf.dtype) - cdf_min

        denominator = ops.where(
            denominator == 0, ops.ones_like(denominator), denominator
        )
        cdf_norm = numerator / denominator
        cdf_norm = ops.clip(cdf_norm, 0, 255)

        # Interpolation

        cdf_padded = ops.pad(
            cdf_norm,
            [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
            mode="symmetric",
        )

        H_padded = tile_h * grid_h
        W_padded = tile_w * grid_w

        y_range = ops.arange(H_padded, dtype="float32")
        x_range = ops.arange(W_padded, dtype="float32")

        y_grid = (y_range - (tile_h / 2.0)) / tile_h
        x_grid = (x_range - (tile_w / 2.0)) / tile_w

        y_grid = y_grid + 1.0
        x_grid = x_grid + 1.0

        y0 = ops.floor(y_grid)
        x0 = ops.floor(x_grid)
        y1 = y0 + 1.0
        x1 = x0 + 1.0

        wy = y_grid - y0
        wx = x_grid - x0

        y0 = ops.clip(y0, 0, grid_h + 1)
        y1 = ops.clip(y1, 0, grid_h + 1)
        x0 = ops.clip(x0, 0, grid_w + 1)
        x1 = ops.clip(x1, 0, grid_w + 1)

        y0 = ops.cast(y0, "int32")
        y1 = ops.cast(y1, "int32")
        x0 = ops.cast(x0, "int32")
        x1 = ops.cast(x1, "int32")

        stride_c = 256
        stride_x = stride_c * channels
        stride_y = stride_x * (grid_w + 2)
        stride_b = stride_y * (grid_h + 2)

        cdf_flat = ops.reshape(cdf_padded, (-1,))

        pixels = ops.cast(ops.clip(padded_images, 0, 255), "int32")

        b_idx = ops.arange(batch_size, dtype="int32")[:, None, None, None]

        c_idx = ops.arange(channels, dtype="int32")[None, None, None, :]

        y0_e = y0[None, :, None, None]
        y1_e = y1[None, :, None, None]

        x0_e = x0[None, None, :, None]
        x1_e = x1[None, None, :, None]

        wy_e = wy[None, :, None, None]
        wx_e = wx[None, None, :, None]

        base_idx = b_idx * stride_b + c_idx * stride_c + pixels

        idx_nw = base_idx + y0_e * stride_y + x0_e * stride_x
        val_nw = ops.take(cdf_flat, idx_nw)

        idx_ne = base_idx + y0_e * stride_y + x1_e * stride_x
        val_ne = ops.take(cdf_flat, idx_ne)

        idx_sw = base_idx + y1_e * stride_y + x0_e * stride_x
        val_sw = ops.take(cdf_flat, idx_sw)

        idx_se = base_idx + y1_e * stride_y + x1_e * stride_x
        val_se = ops.take(cdf_flat, idx_se)

        top = val_nw * (1.0 - wx_e) + val_ne * wx_e
        bot = val_sw * (1.0 - wx_e) + val_se * wx_e
        result = top * (1.0 - wy_e) + bot * wy_e

        result = result[:, :height, :width, :]

        if unbatched:
            result = ops.squeeze(result, axis=0)

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
