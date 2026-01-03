from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
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

    **Structured inputs**

    This layer supports synchronized augmentation of images and associated data
    by passing a dictionary with one or more of the following keys:

    - `"images"` (required): Input images tensor with shape
    `(..., height, width, channels)` when `data_format="channels_last"`, or
    `(..., channels, height, width)` when `data_format="channels_first"`.
    - `"segmentation_masks"` (optional): Segmentation masks with the same
    spatial dimensions as `"images"`. Masks always use `"nearest"`
    interpolation to preserve discrete label values.
    - `"bounding_boxes"` (optional): A dictionary with `"boxes"` and `"labels"`
    keys representing bounding boxes associated with `"images"`. When provided,
    `bounding_box_format` must also be specified.
    - `"labels"` (optional): Classification labels. Passed through unchanged.

    All entries are transformed using the same randomly sampled rotation,
    ensuring that images, masks, and bounding boxes remain spatially aligned.

    **Crop mode**

    When `fill_mode="crop"`, the layer:

    1. Pads the input to a size large enough to fully contain the rotated image.
    2. Applies the rotation on the padded canvas.
    3. Center-crops the result back to the original spatial dimensions.

    This removes padded or filled border artifacts introduced by rotation while
    guaranteeing that the output shape matches the input shape. This mode is
    particularly useful for dense prediction tasks such as segmentation.

    When using structured inputs, cropping is applied to images and segmentation
    masks only. Bounding boxes are rotated and clipped to image boundaries, but
    are not additionally cropped.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)` when `data_format="channels_last"`, or
        `(..., channels, height, width)` when `data_format="channels_first"`.

    Output shape:
        Same as the input shape.

    Examples:

    **Basic usage with crop mode:**

    ```python
    import numpy as np
    from keras import layers

    images = np.random.randint(0, 256, (2, 224, 224, 3), dtype="uint8")
    layer = layers.RandomRotation(factor=0.2, fill_mode="crop")
    outputs = layer(images, training=True)
    ```

    **Synchronized rotation for images and segmentation masks:**

    ```python
    import numpy as np
    from keras import layers

    images = np.random.randint(0, 256, (2, 224, 224, 3), dtype="uint8")
    masks = np.random.randint(0, 5, (2, 224, 224, 1), dtype="uint8")

    data = {
        "images": images,
        "segmentation_masks": masks,
    }

    layer = layers.RandomRotation(factor=0.2, fill_mode="crop", seed=42)
    result = layer(data, training=True)
    # result["images"] and result["segmentation_masks"] are rotated identically.
    ```

    **Rotating images with bounding boxes:**

    ```python
    import numpy as np
    from keras import layers

    images = np.random.randint(0, 256, (2, 224, 224, 3), dtype="uint8")
    boxes = {
        "boxes": np.array(
            [
                [[10, 10, 50, 50], [60, 60, 100, 100]],
                [[20, 20, 80, 80], [120, 120, 180, 180]],
            ],
            dtype="float32",
        ),
        "labels": np.array([[1, 2], [3, 4]], dtype="int32"),
    }

    data = {
        "images": images,
        "bounding_boxes": boxes,
    }

    layer = layers.RandomRotation(
        factor=0.15,
        fill_mode="crop",
        bounding_box_format="xyxy",
        seed=42,
    )
    result = layer(data, training=True)
    ```

    **Comparing fill modes:**

    ```python
    import numpy as np
    from keras import layers

    image = np.random.randint(0, 256, (224, 224, 3), dtype="uint8")

    # fill_mode="constant" may leave visible border artifacts
    layer_constant = layers.RandomRotation(
        factor=0.3,
        fill_mode="constant",
        fill_value=0,
    )
    output_constant = layer_constant(image, training=True)

    # fill_mode="crop" removes border artifacts
    layer_crop = layers.RandomRotation(factor=0.3, fill_mode="crop")
    output_crop = layer_crop(image, training=True)
    ```

    Args:
        factor: A float or a tuple of two floats representing a fraction of a
            full rotation (360 degrees). If a single float is provided, the
            rotation angle is sampled uniformly from
            `[-factor * 360, factor * 360]`. If a tuple `(lower, upper)` is
            provided, the angle is sampled from `[lower * 360, upper * 360]`.
        fill_mode: Points outside the input boundaries are filled according to
            the given mode (one of
            `{"constant", "reflect", "wrap", "nearest", "crop"}`).
            - `"reflect"`: Reflects values at the edge.
            - `"constant"`: Fills with the constant value `fill_value`.
            - `"wrap"`: Wraps around to the opposite edge.
            - `"nearest"`: Extends the nearest edge value.
            - `"crop"`: Pads, rotates, then center-crops back to the original
              size. Removes border artifacts introduced by rotation.
        interpolation: Interpolation mode. Supported values are `"nearest"` and
            `"bilinear"`. Segmentation masks always use `"nearest"`
            interpolation.
        seed: Optional integer used to create a deterministic random seed.
        fill_value: Float value used when `fill_mode="constant"`.
        data_format: One of `"channels_last"` or `"channels_first"`. Defaults to
            the global Keras image data format.
        bounding_box_format: String, the format of bounding boxes when
            `"bounding_boxes"` are provided (e.g., `"xyxy"`, `"xywh"`). Required
            when using bounding box inputs.
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
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if self.interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

    def _get_hw(self, images):
        shape = ops.shape(images)
        if self.data_format == "channels_last":
            return shape[-3], shape[-2]
        return shape[-2], shape[-1]

    def _ensure_batched(self, x):
        # Supports rank-3 inputs by temporarily adding a batch dimension.
        if len(ops.shape(x)) == 3:
            return ops.expand_dims(x, axis=0), True
        return x, False

    def _maybe_unbatch(self, x, was_unbatched):
        if was_unbatched:
            return ops.squeeze(x, axis=0)
        return x

    def _pad_for_crop(self, images, height, width):
        """Pad so rotation can be center-cropped without border artifacts."""
        # side = ceil(sqrt(H^2 + W^2))
        h = ops.cast(height, "float32")
        w = ops.cast(width, "float32")
        side = ops.cast(ops.ceil(ops.sqrt(h * h + w * w)), "int32")

        pad_total_h = side - height
        pad_total_w = side - width
        pad_top = pad_total_h // 2
        pad_bottom = pad_total_h - pad_top
        pad_left = pad_total_w // 2
        pad_right = pad_total_w - pad_left

        if self.data_format == "channels_last":
            paddings = [
                [0, 0],
                [pad_top, pad_bottom],
                [pad_left, pad_right],
                [0, 0],
            ]
        else:
            paddings = [
                [0, 0],
                [0, 0],
                [pad_top, pad_bottom],
                [pad_left, pad_right],
            ]

        # Values are intended to be removed by the final center crop.
        return ops.pad(images, paddings, constant_values=0)

    def _center_crop_to(self, images, target_h, target_w):
        h, w = self._get_hw(images)
        offset_h = ops.cast((h - target_h) // 2, "int32")
        offset_w = ops.cast((w - target_w) // 2, "int32")

        shape = ops.shape(images)
        if self.data_format == "channels_last":
            batch = shape[0]
            channels = shape[3]
            return ops.slice(
                images,
                start_indices=[0, offset_h, offset_w, 0],
                shape=[batch, target_h, target_w, channels],
            )
        else:
            batch = shape[0]
            channels = shape[1]
            return ops.slice(
                images,
                start_indices=[0, 0, offset_h, offset_w],
                shape=[batch, channels, target_h, target_w],
            )

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

    def _apply_rotation_with_crop(self, images, transformation, interpolation):
        orig_h = transformation["image_height"]
        orig_w = transformation["image_width"]

        padded = self._pad_for_crop(images, orig_h, orig_w)
        padded_h, padded_w = self._get_hw(padded)

        padded_rotation_matrix = self._compute_affine_matrix(
            center_x=0.5,
            center_y=0.5,
            angle=transformation["angle"],
            translate_x=ops.numpy.zeros([transformation["batch_size"]]),
            translate_y=ops.numpy.zeros([transformation["batch_size"]]),
            scale=ops.numpy.ones([transformation["batch_size"]]),
            shear_x=ops.numpy.zeros([transformation["batch_size"]]),
            shear_y=ops.numpy.zeros([transformation["batch_size"]]),
            height=padded_h,
            width=padded_w,
        )

        rotated = self._apply_affine(
            images=padded,
            rotation_matrix=padded_rotation_matrix,
            interpolation=interpolation,
            fill_mode="constant",
            fill_value=0.0,
        )
        return self._center_crop_to(rotated, orig_h, orig_w)

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if not training:
            return images

        images, was_unbatched = self._ensure_batched(images)

        if self.fill_mode != "crop":
            out = self._apply_affine(
                images=images,
                rotation_matrix=transformation["rotation_matrix"],
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )
            return self._maybe_unbatch(out, was_unbatched)

        out = self._apply_rotation_with_crop(
            images, transformation, self.interpolation
        )
        return self._maybe_unbatch(out, was_unbatched)

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        if not training:
            return bounding_boxes

        ops = self.backend
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
        zeros = ops.numpy.zeros_like(angle)
        ones = ops.numpy.ones_like(angle)

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

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        # Masks are discrete: always use nearest interpolation.
        segmentation_masks = self.backend.cast(
            segmentation_masks, self.compute_dtype
        )
        if not training:
            return segmentation_masks

        segmentation_masks, was_unbatched = self._ensure_batched(
            segmentation_masks
        )

        if self.fill_mode != "crop":
            # For masks, fill with 0 to avoid introducing non-class values.
            out = self._apply_affine(
                images=segmentation_masks,
                rotation_matrix=transformation["rotation_matrix"],
                interpolation="nearest",
                fill_mode=self.fill_mode,
                fill_value=0.0,
            )
            return self._maybe_unbatch(out, was_unbatched)

        out = self._apply_rotation_with_crop(
            segmentation_masks, transformation, "nearest"
        )
        return self._maybe_unbatch(out, was_unbatched)

    def get_random_transformation(self, data, training=True, seed=None):
        ops = self.backend
        if not training:
            return None

        images = data["images"] if isinstance(data, dict) else data
        shape = ops.core.shape(images)
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
            seed = self._get_seed_generator(ops._backend)

        angle = ops.random.uniform(
            shape=(batch_size,),
            minval=self.factor[0] * 360.0,
            maxval=self.factor[1] * 360.0,
            seed=seed,
        )

        rotation_matrix = self._compute_affine_matrix(
            center_x=0.5,
            center_y=0.5,
            angle=angle,
            translate_x=ops.numpy.zeros([batch_size]),
            translate_y=ops.numpy.zeros([batch_size]),
            scale=ops.numpy.ones([batch_size]),
            shear_x=ops.numpy.zeros([batch_size]),
            shear_y=ops.numpy.zeros([batch_size]),
            height=image_height,
            width=image_width,
        )

        if ndim == 3:
            rotation_matrix = ops.numpy.squeeze(rotation_matrix, axis=0)

        return {
            "angle": angle,
            "rotation_matrix": rotation_matrix,
            "image_height": ops.core.cast(image_height, "int32"),
            "image_width": ops.core.cast(image_width, "int32"),
            "batch_size": ops.core.cast(batch_size, "int32"),
        }

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "data_format": self.data_format,
            "bounding_box_format": self.bounding_box_format,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
