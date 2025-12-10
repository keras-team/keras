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

    This layer:
      1. Samples a random relative area from `scale`.
      2. Samples a random aspect ratio from `ratio`.
      3. Derives a crop window (height, width) from these values.
      4. Crops the image and resizes the crop to `(height, width)`.

    Args:
        height: Integer. Target height of the output image.
        width: Integer. Target width of the output image.
        scale: Tuple of two floats `(min_scale, max_scale)`. The
            sampled relative area (crop_area / image_area) will lie
            in this range. Default `(0.08, 1.0)`.
        ratio: Tuple of two floats `(min_ratio, max_ratio)`. Aspect
            ratio (width / height) of the crop is sampled from this
            interval in log-space. Default `(0.75, 1.33)`.
        interpolation: String. Interpolation mode used in the resize
            step, e.g. `"bilinear"`. Default `"bilinear"`.
        seed: Optional integer. Random seed.
        data_format: Optional string, `"channels_last"` or
            `"channels_first"`. Follows global image data format by
            default.
        name: Optional string name.
        **kwargs: Additional layer keyword arguments.

    Notes:
        * On inference (`training=False`), the layer performs a
          deterministic center crop that preserves the target
          aspect ratio, followed by resize to `(height, width)`.
        * On the OpenVINO backend, `backend.image.resize` is not
          implemented. In this case, the layer raises a
          `NotImplementedError` at runtime.
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

        self.supports_masking = False
        self.supports_jit = False
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def get_random_transformation(self, data, training=True, seed=None):
        """Returns a crop transformation `(h_start, w_start, crop_h, crop_w)`.

        The same crop parameters are applied to all images in a batch,
        which matches the behavior of other preprocessing layers.
        """
        if isinstance(data, dict):
            images = data.get("images", None)
            input_shape = backend.shape(images)
        else:
            input_shape = backend.shape(data)

        input_height = ops.cast(input_shape[self.height_axis], "float32")
        input_width = ops.cast(input_shape[self.width_axis], "float32")

        if training:
            h_start, w_start, crop_h, crop_w = self._get_random_crop_params(
                input_height, input_width, seed
            )
        else:
            h_start, w_start, crop_h, crop_w = self._get_center_crop_params(
                input_height, input_width
            )

        return (
            ops.cast(h_start, "int32"),
            ops.cast(w_start, "int32"),
            ops.cast(crop_h, "int32"),
            ops.cast(crop_w, "int32"),
        )

    def _get_random_crop_params(self, input_height, input_width, seed):
        """Compute random crop parameters using `scale` and `ratio`."""
        if seed is None:
            seed = self.generator

        scale_min = float(self.scale[0])
        scale_max = float(self.scale[1])
        scale_factor = backend.random.uniform(
            (), scale_min, scale_max, seed=seed
        )

        ratio_min = float(self.ratio[0])
        ratio_max = float(self.ratio[1])
        log_ratio_min = ops.log(
            ops.convert_to_tensor(ratio_min, dtype="float32")
        )
        log_ratio_max = ops.log(
            ops.convert_to_tensor(ratio_max, dtype="float32")
        )
        log_ratio = backend.random.uniform(
            (), log_ratio_min, log_ratio_max, seed=seed
        )
        aspect_ratio = ops.exp(log_ratio)

        area = input_height * input_width
        target_area = scale_factor * area

        crop_h = ops.sqrt(target_area / aspect_ratio)
        crop_w = ops.sqrt(target_area * aspect_ratio)

        one = ops.convert_to_tensor(1.0, dtype="float32")
        crop_h = ops.clip(crop_h, one, input_height)
        crop_w = ops.clip(crop_w, one, input_width)

        max_h_start = ops.maximum(input_height - crop_h, 0.0)
        max_w_start = ops.maximum(input_width - crop_w, 0.0)
        rand_h = backend.random.uniform((), 0.0, 1.0, seed=seed)
        rand_w = backend.random.uniform((), 0.0, 1.0, seed=seed)
        h_start = rand_h * max_h_start
        w_start = rand_w * max_w_start

        return h_start, w_start, crop_h, crop_w

    def _get_center_crop_params(self, input_height, input_width):
        """Center crop that preserves target aspect ratio."""
        target_aspect = ops.cast(self.width, "float32") / ops.cast(
            self.height, "float32"
        )

        input_aspect = input_width / input_height

        crop_h = ops.where(
            input_aspect > target_aspect,
            input_height,
            input_width / target_aspect,
        )
        crop_w = ops.where(
            input_aspect > target_aspect,
            input_height * target_aspect,
            input_width,
        )

        h_start = (input_height - crop_h) / 2.0
        w_start = (input_width - crop_w) / 2.0

        return h_start, w_start, crop_h, crop_w

    def _slice_images(self, tensor, h_start, w_start, crop_h, crop_w):
        """Slice [B, H, W, C] or [B, C, H, W] efficiently on all backends."""
        h_start = ops.cast(h_start, "int32")
        w_start = ops.cast(w_start, "int32")
        crop_h = ops.cast(crop_h, "int32")
        crop_w = ops.cast(crop_w, "int32")

        if self.data_format == "channels_first":
            return tensor[
                :, :, h_start : h_start + crop_h, w_start : w_start + crop_w
            ]
        else:
            return tensor[
                :, h_start : h_start + crop_h, w_start : w_start + crop_w, :
            ]

    def _resize_images(self, images):
        """Resize images to `(height, width)` using backend API.

        For OpenVINO, this raises NotImplementedError because the
        backend does not yet provide `backend.image.resize`.
        """
        if backend.backend() == "openvino":
            raise NotImplementedError(
                "`RandomResizedCrop` is not yet supported on the "
                "OpenVINO backend because `backend.image.resize` is "
                "not implemented there. Please use `RandomCrop` or "
                "switch to a different backend until resize support "
                "is added."
            )

        return backend.image.resize(
            images,
            size=(self.height, self.width),
            interpolation=self.interpolation,
            antialias=False,
            crop_to_aspect_ratio=False,
            pad_to_aspect_ratio=False,
            fill_mode="constant",
            fill_value=0.0,
            data_format=self.data_format,
        )

    def transform_images(self, images, transformation=None, training=True):
        """Apply random resized crop to a batch of images."""
        if transformation is None:
            transformation = self.get_random_transformation(
                images, training=training
            )
        h_start, w_start, crop_h, crop_w = transformation

        images = self._slice_images(images, h_start, w_start, crop_h, crop_w)
        images = self._resize_images(images)
        return images

    def transform_segmentation_masks(
        self, masks, transformation, training=True
    ):
        """Apply the same crop + resize to segmentation masks."""
        h_start, w_start, crop_h, crop_w = transformation
        masks = self._slice_images(masks, h_start, w_start, crop_h, crop_w)

        if backend.backend() == "openvino":
            raise NotImplementedError(
                "Segmentation mask resizing for `RandomResizedCrop` is "
                "not yet supported on the OpenVINO backend."
            )

        return backend.image.resize(
            masks,
            size=(self.height, self.width),
            interpolation="nearest",
            antialias=False,
            crop_to_aspect_ratio=False,
            pad_to_aspect_ratio=False,
            fill_mode="constant",
            fill_value=0.0,
            data_format=self.data_format,
        )

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(self, bboxes, transformation, training=True):
        """Transform bounding boxes to match the crop."""
        if not isinstance(bboxes, dict) or "boxes" not in bboxes:
            return bboxes

        h_start, w_start, crop_h, crop_w = [
            ops.cast(t, "float32") for t in transformation
        ]

        boxes = ops.cast(bboxes["boxes"], "float32")
        x_min, y_min, x_max, y_max = ops.unstack(boxes, axis=-1)

        source = bboxes.get("images", boxes)
        input_shape = backend.shape(source)
        input_h = ops.cast(input_shape[self.height_axis], "float32")
        input_w = ops.cast(input_shape[self.width_axis], "float32")

        x_min_px = x_min * input_w
        y_min_px = y_min * input_h
        x_max_px = x_max * input_w
        y_max_px = y_max * input_h

        x_min_crop = ops.maximum(x_min_px - w_start, 0.0)
        y_min_crop = ops.maximum(y_min_px - h_start, 0.0)
        x_max_crop = ops.minimum(x_max_px - w_start, crop_w)
        y_max_crop = ops.minimum(y_max_px - h_start, crop_h)

        x_min_crop = x_min_crop / crop_w
        y_min_crop = y_min_crop / crop_h
        x_max_crop = x_max_crop / crop_w
        y_max_crop = y_max_crop / crop_h

        x_min_crop = ops.clip(x_min_crop, 0.0, 1.0)
        y_min_crop = ops.clip(y_min_crop, 0.0, 1.0)
        x_max_crop = ops.clip(x_max_crop, 0.0, 1.0)
        y_max_crop = ops.clip(y_max_crop, 0.0, 1.0)

        result = dict(bboxes)
        result["boxes"] = ops.cast(
            ops.stack(
                [x_min_crop, y_min_crop, x_max_crop, y_max_crop], axis=-1
            ),
            boxes.dtype,
        )
        return result

    def compute_output_shape(self, input_shape, *args, **kwargs):
        output_shape = list(input_shape)
        output_shape[self.height_axis] = self.height
        output_shape[self.width_axis] = self.width
        return tuple(output_shape)

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
