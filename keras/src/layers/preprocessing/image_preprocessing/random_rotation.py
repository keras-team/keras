from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    base_image_preprocessing_transform_example,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes import (
    converters,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomRotation")
class RandomRotation(BaseImagePreprocessingLayer):
    """A preprocessing layer that randomly rotates images during training.

    This layer applies a random rotation to each image, filling areas outside
    the image boundaries according to `fill_mode`.

    By default, random rotations are applied only during training.
    At inference time, the layer returns the inputs unchanged. To force
    augmentation at inference time, pass `training=True` when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    can be integer or floating-point. The output is always floating-point.

    **Note:** This layer is safe to use inside `tf.data` and `grain` input
    pipelines. When used in a `tf.data` pipeline, the layer correctly handles
    symbolic tensors across supported backends.

    ## Structured inputs

    This layer supports combined augmentation of images and associated data
    by passing a dictionary with one or more of the following keys:
    - `"images"` (required): Input images tensor with shape
      `(..., height, width, channels)` when `data_format="channels_last"`, or
      `(..., channels, height, width)` when `data_format="channels_first"`.
    - `"segmentation_masks"` (optional): Segmentation masks with the same
      spatial dimensions as `"images"`. Masks always use `"nearest"`
      interpolation to preserve discrete label values.
    - `"bounding_boxes"` (optional): A dictionary with `"boxes"` and `"labels"`
      keys representing bounding boxes associated with `"images"`.
      When provided, `bounding_box_format` must also be specified.
    - `"labels"` (optional): Classification labels. Passed through unchanged.

    All entries are transformed using the same randomly sampled rotation,
    ensuring that images, masks, and bounding boxes remain spatially aligned.

    ## Crop mode

    When `fill_mode="crop"`, the layer applies an angle-dependent zoom
    during the rotation affine transform to remove border artifacts without
    explicit cropping or resizing.

    This preserves output shape while avoiding fill regions. For batched
    inputs, the zoom uses the maximal scale across the batch for uniform
    processing. This mode is particularly useful for dense prediction tasks
    such as segmentation.

    When using structured inputs, the zoomed rotation is applied to images
    and segmentation masks. Bounding boxes are rotated and clipped
    (zoom intentionally not applied).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)` when `data_format="channels_last"`, or
        `(..., channels, height, width)` when `data_format="channels_first"`.

    Output shape:
        Same as the input shape.

    Args:
        factor: A float or a tuple of two floats representing a fraction of a
            full rotation (360 degrees). If a single float is provided, the
            rotation angle is sampled uniformly from
            `[-factor * 360, factor * 360]`. If a tuple `(lower, upper)` is
            provided, the angle is sampled from
            `[lower * 360, upper * 360]`.
        fill_mode: Points outside the input boundaries are filled according to
            the given mode (one of
            `{"constant", "reflect", "wrap", "nearest", "crop"}`).
            - `"reflect"`: Reflects values at the edge.
            - `"constant"`: Fills with the constant value `fill_value`.
            - `"wrap"`: Wraps around to the opposite edge.
            - `"nearest"`: Extends the nearest edge value.
            - `"crop"`: Rotates with an angle-dependent zoom to remove border
              artifacts without explicit cropping or resizing.
        interpolation: Interpolation mode. Supported values are `"nearest"` and
            `"bilinear"`. Segmentation masks always use `"nearest"`
            interpolation.
        seed: Optional integer used to create a deterministic random seed.
        fill_value: Float value used when `fill_mode="constant"`.
        data_format: One of `"channels_last"` or `"channels_first"`. Defaults to
            the global Keras image data format.
        bounding_box_format: String specifying the format of bounding boxes when
            `"bounding_boxes"` are provided (e.g., `"xyxy"`, `"xywh"`). Required
            when using bounding box inputs.

    Example:

    {{base_image_preprocessing_transform_example}}
    """

    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest", "crop")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        data_format=None,
        bounding_box_format=None,
        **kwargs,
    ):
        super().__init__(
            factor=factor,
            data_format=data_format,
            bounding_box_format=bounding_box_format,
            **kwargs,
        )
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.supports_jit = False

        if self.fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected one of "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if self.interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected one of "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

    def _get_hw(self, images):
        shape = self.backend.shape(images)
        if self.data_format == "channels_last":
            return shape[-3], shape[-2]
        return shape[-2], shape[-1]

    def _ensure_batched(self, x):
        if len(self.backend.shape(x)) == 3:
            return self.backend.expand_dims(x, axis=0), True
        return x, False

    def _maybe_unbatch(self, x, was_unbatched):
        if was_unbatched:
            return self.backend.squeeze(x, axis=0)
        return x

    def _apply_affine(
        self, images, rotation_matrix, interpolation, fill_mode, fill_value
    ):
        return self.backend.image.affine_transform(
            images=images,
            transform=rotation_matrix,
            interpolation=interpolation,
            fill_mode=fill_mode,
            fill_value=fill_value,
            data_format=self.data_format,
        )

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if not training:
            return images

        images, was_unbatched = self._ensure_batched(images)

        out = self._apply_affine(
            images=images,
            rotation_matrix=transformation["rotation_matrix"],
            interpolation=self.interpolation,
            fill_mode=transformation.get("fill_mode", self.fill_mode),
            fill_value=transformation.get("fill_value", self.fill_value),
        )
        return self._maybe_unbatch(out, was_unbatched)

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        segmentation_masks = self.backend.cast(
            segmentation_masks, self.compute_dtype
        )
        if not training:
            return segmentation_masks

        segmentation_masks, was_unbatched = self._ensure_batched(
            segmentation_masks
        )

        out = self._apply_affine(
            images=segmentation_masks,
            rotation_matrix=transformation["rotation_matrix"],
            interpolation="nearest",
            fill_mode=transformation.get("fill_mode", self.fill_mode),
            fill_value=transformation.get("fill_value", 0.0),
        )
        return self._maybe_unbatch(out, was_unbatched)

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        if not training:
            return bounding_boxes

        height = transformation["image_height"]
        width = transformation["image_width"]

        bounding_boxes = converters.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=height,
            width=width,
        )

        angle = transformation["angle"]
        zeros = self.backend.numpy.zeros_like(angle)
        ones = self.backend.numpy.ones_like(angle)

        boxes = bounding_boxes["boxes"]
        boxes = converters.affine_transform(
            boxes=boxes,
            angle=angle,
            translate_x=zeros,
            translate_y=zeros,
            scale=ones,
            shear_x=zeros,
            shear_y=zeros,
            height=height,
            width=width,
        )
        bounding_boxes["boxes"] = boxes

        bounding_boxes = converters.clip_to_image_size(
            bounding_boxes,
            height=height,
            width=width,
            bounding_box_format="xyxy",
        )

        bounding_boxes = converters.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            height=height,
            width=width,
        )
        return bounding_boxes

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        images = data["images"] if isinstance(data, dict) else data
        shape = self.backend.shape(images)
        ndim = len(shape)

        if ndim == 4:
            batch_size = shape[0]
            if self.data_format == "channels_last":
                image_height = shape[1]
                image_width = shape[2]
            else:
                image_height = shape[2]
                image_width = shape[3]
        else:
            batch_size = 1
            if self.data_format == "channels_last":
                image_height = shape[0]
                image_width = shape[1]
            else:
                image_height = shape[1]
                image_width = shape[2]

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        angle = self.backend.random.uniform(
            shape=(batch_size,),
            minval=self.factor[0] * 360.0,
            maxval=self.factor[1] * 360.0,
            seed=seed or self.generator,
        )

        scale = self.backend.numpy.ones([batch_size])
        fill_mode = self.fill_mode
        fill_value = self.fill_value

        if self.fill_mode == "crop":
            scale = self._get_rotation_scale(
                self.backend.cast(image_height, "float32"),
                self.backend.cast(image_width, "float32"),
                angle,
            )
            fill_mode = "constant"
            fill_value = 0.0

        rotation_matrix = self._compute_affine_matrix(
            center_x=0.5,
            center_y=0.5,
            angle=angle,
            translate_x=self.backend.numpy.zeros([batch_size]),
            translate_y=self.backend.numpy.zeros([batch_size]),
            scale=scale,
            shear_x=self.backend.numpy.zeros([batch_size]),
            shear_y=self.backend.numpy.zeros([batch_size]),
            height=image_height,
            width=image_width,
        )

        if ndim == 3:
            rotation_matrix = self.backend.numpy.squeeze(
                rotation_matrix, axis=0
            )

        return {
            "angle": angle,
            "rotation_matrix": rotation_matrix,
            "image_height": self.backend.cast(image_height, "int32"),
            "image_width": self.backend.cast(image_width, "int32"),
            "batch_size": self.backend.cast(batch_size, "int32"),
            "fill_mode": fill_mode,
            "fill_value": fill_value,
        }

    def _get_rotation_scale(self, height, width, angles):
        """Compute angle-dependent scale < 1 so rotated image fits
        without fill."""
        angles_rad = self.backend.numpy.deg2rad(angles)
        sin_a = self.backend.numpy.abs(self.backend.numpy.sin(angles_rad))
        cos_a = self.backend.numpy.abs(self.backend.numpy.cos(angles_rad))

        denom_w = width * cos_a + height * sin_a
        denom_h = width * sin_a + height * cos_a

        scale_w = width / denom_w
        scale_h = height / denom_h
        scale = self.backend.numpy.minimum(scale_w, scale_h)

        min_scale = self.backend.numpy.min(scale)

        return min_scale

    def get_config(self):
        config = {
            "factor": self.factor,
            "fill_mode": self.fill_mode,
            "interpolation": self.interpolation,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}


RandomRotation.__doc__ = RandomRotation.__doc__.replace(
    "{{base_image_preprocessing_transform_example}}",
    base_image_preprocessing_transform_example.replace(
        "{LayerName}", "RandomRotation"
    ),
)
