from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomResizedCrop")
class RandomResizedCrop(BaseImagePreprocessingLayer):
    """Randomly crops and resizes images to a target size.

    This layer implements the random resized cropping data augmentation
    strategy commonly used in training vision models. During training, it
    randomly samples a crop area and aspect ratio, extracts the corresponding
    region from the input images, and resizes it to a fixed target size.
    This combines the effects of random cropping, random zooming, and resizing
    into a single efficient operation.

    During inference (`training=False`), the layer applies a deterministic
    center crop that preserves the target aspect ratio, followed by resizing
    to `(height, width)`. This ensures consistent and reproducible behavior
    at inference time.

    Args:
        height: Integer. Target height of the output images.
        width: Integer. Target width of the output images.
        scale: Tuple of two floats `(min_scale, max_scale)`. Specifies the
            range for the random area of the crop as a fraction of the input
            image area. Default is `(0.08, 1.0)`.
        ratio: Tuple of two floats `(min_ratio, max_ratio)`. Specifies the
            range for the random aspect ratio of the crop
            (`width / height`). Values are sampled in log-space.
            Default is `(0.75, 1.33)`.
        interpolation: String. Interpolation mode used for resizing.
            Defaults to `"bilinear"`.
        seed: Optional integer. Random seed for reproducibility.
        data_format: Optional string, either `"channels_last"` or
            `"channels_first"`. Defaults to the global Keras image data
            format.
        name: Optional string. Name of the layer.

    Input shape:
        3D tensor `(height, width, channels)` or
        4D tensor `(batch_size, height, width, channels)` if
        `data_format="channels_last"`.
        If `data_format="channels_first"`, the channels dimension is
        expected at axis 1.

    Output shape:
        Same rank as the input, with spatial dimensions replaced by
        `(height, width)`.

    Example:
        >>> import keras, numpy as np
        >>> layer = keras.layers.RandomResizedCrop(224, 224)
        >>> images = np.random.random((8, 256, 256, 3)).astype("float32")
        >>> augmented = layer(images, training=True)
        >>> augmented.shape
        (8, 224, 224, 3)
    """

    def __init__(
        self,
        height,
        width,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.33),
        interpolation="bilinear",
        seed=None,
        data_format=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, data_format=data_format, **kwargs)

        self.height = int(height)
        self.width = int(width)
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)
        self.interpolation = interpolation

        self.seed = (
            seed if seed is not None else backend.random.make_default_seed()
        )
        self.generator = SeedGenerator(self.seed)

        self.data_format = backend.standardize_data_format(self.data_format)
        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
        else:
            self.height_axis = -3
            self.width_axis = -2

    def get_random_transformation(self, data, training=True, seed=None):
        if isinstance(data, dict):
            images = data.get("images", None)
            shape = backend.shape(images)
        else:
            shape = backend.shape(data)

        input_h = ops.cast(shape[self.height_axis], "float32")
        input_w = ops.cast(shape[self.width_axis], "float32")

        if training:
            if seed is None:
                seed = self._get_seed_generator(self.backend._backend)
            h, w, ch, cw = self._random_crop_params(input_h, input_w, seed)
        else:
            h, w, ch, cw = self._center_crop_params(input_h, input_w)

        return (
            ops.cast(h, "int32"),
            ops.cast(w, "int32"),
            ops.cast(ch, "int32"),
            ops.cast(cw, "int32"),
        )

    def _random_crop_params(self, input_h, input_w, seed):
        scale_min, scale_max = self.scale
        ratio_min, ratio_max = self.ratio

        area = input_h * input_w
        target_area = (
            backend.random.uniform((), scale_min, scale_max, seed=seed) * area
        )

        log_ratio_min = ops.log(ops.convert_to_tensor(ratio_min, "float32"))
        log_ratio_max = ops.log(ops.convert_to_tensor(ratio_max, "float32"))
        aspect_ratio = ops.exp(
            backend.random.uniform((), log_ratio_min, log_ratio_max, seed=seed)
        )

        crop_h = ops.sqrt(target_area / aspect_ratio)
        crop_w = ops.sqrt(target_area * aspect_ratio)

        one = ops.convert_to_tensor(1.0, "float32")
        crop_h = ops.clip(crop_h, one, input_h)
        crop_w = ops.clip(crop_w, one, input_w)

        max_h = ops.maximum(input_h - crop_h, 0.0)
        max_w = ops.maximum(input_w - crop_w, 0.0)

        h_start = backend.random.uniform((), 0.0, 1.0, seed=seed) * max_h
        w_start = backend.random.uniform((), 0.0, 1.0, seed=seed) * max_w

        return h_start, w_start, crop_h, crop_w

    def _center_crop_params(self, input_h, input_w):
        target_aspect = ops.cast(self.width, "float32") / ops.cast(
            self.height, "float32"
        )
        input_aspect = input_w / input_h

        crop_h = ops.where(
            input_aspect > target_aspect,
            input_h,
            input_w / target_aspect,
        )
        crop_w = ops.where(
            input_aspect > target_aspect,
            input_h * target_aspect,
            input_w,
        )

        h_start = (input_h - crop_h) / 2.0
        w_start = (input_w - crop_w) / 2.0

        return h_start, w_start, crop_h, crop_w

    def _slice_images(self, x, h, w, ch, cw):
        if self.data_format == "channels_first":
            return x[:, :, h : h + ch, w : w + cw]
        return x[:, h : h + ch, w : w + cw, :]

    def transform_images(self, images, transformation=None, training=True):
        if transformation is None:
            transformation = self.get_random_transformation(images, training)
        h, w, ch, cw = transformation
        images = self._slice_images(images, h, w, ch, cw)
        return backend.image.resize(
            images,
            size=(self.height, self.width),
            interpolation=self.interpolation,
            data_format=self.data_format,
        )

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        rank = len(input_shape)

        if self.data_format == "channels_last":
            if rank == 4:
                input_shape[1] = self.height
                input_shape[2] = self.width
            elif rank == 3:
                input_shape[0] = self.height
                input_shape[1] = self.width
        else:  # channels_first
            if rank == 4:
                input_shape[2] = self.height
                input_shape[3] = self.width
            elif rank == 3:
                input_shape[1] = self.height
                input_shape[2] = self.width

        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "scale": self.scale,
                "ratio": self.ratio,
                "interpolation": self.interpolation,
                "seed": self.seed,
                "data_format": self.data_format,
            }
        )
        return config
