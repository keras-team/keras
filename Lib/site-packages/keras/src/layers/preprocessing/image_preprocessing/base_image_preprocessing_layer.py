import math

from keras.src.backend import config as backend_config
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.validation import (  # noqa: E501
    densify_bounding_boxes,
)
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer


class BaseImagePreprocessingLayer(TFDataLayer):
    _USE_BASE_FACTOR = True
    _FACTOR_BOUNDS = (-1, 1)

    def __init__(
        self, factor=None, bounding_box_format=None, data_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.data_format = backend_config.standardize_data_format(data_format)
        if self._USE_BASE_FACTOR:
            factor = factor or 0.0
            self._set_factor(factor)
        elif factor is not None:
            raise ValueError(
                f"Layer {self.__class__.__name__} does not take "
                f"a `factor` argument. Received: factor={factor}"
            )

    def _set_factor(self, factor):
        error_msg = (
            "The `factor` argument should be a number "
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
        self.factor = lower, upper

    def get_random_transformation(self, data, training=True, seed=None):
        return None

    def transform_images(self, images, transformation, training=True):
        raise NotImplementedError()

    def transform_labels(self, labels, transformation, training=True):
        raise NotImplementedError()

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
        raise NotImplementedError()

    def transform_single_image(self, image, transformation, training=True):
        images = self.backend.numpy.expand_dims(image, axis=0)
        outputs = self.transform_images(
            images, transformation=transformation, training=training
        )
        return self.backend.numpy.squeeze(outputs, axis=0)

    def transform_single_label(self, label, transformation, training=True):
        labels = self.backend.numpy.expand_dims(label, axis=0)
        outputs = self.transform_labels(
            labels, transformation=transformation, training=training
        )
        return self.backend.numpy.squeeze(outputs, axis=0)

    def transform_single_bounding_box(
        self,
        bounding_box,
        transformation,
        training=True,
    ):
        bounding_boxes = self._format_single_input_bounding_box(bounding_box)
        outputs = self.transform_bounding_boxes(
            bounding_boxes,
            transformation=transformation,
            training=training,
        )
        bounding_box = self._format_single_output_bounding_box(outputs)
        return bounding_box

    def transform_single_segmentation_mask(
        self, segmentation_mask, transformation, training=True
    ):
        segmentation_masks = self.backend.numpy.expand_dims(
            segmentation_mask, axis=0
        )
        outputs = self.transform_segmentation_masks(
            segmentation_masks, transformation=transformation, training=training
        )
        return self.backend.numpy.squeeze(outputs, axis=0)

    def _is_batched(self, maybe_image_batch):
        shape = self.backend.core.shape(maybe_image_batch)
        if len(shape) == 3:
            return False
        if len(shape) == 4:
            return True
        raise ValueError(
            "Expected image tensor to have rank 3 (single image) "
            f"or 4 (batch of images). Received: data.shape={shape}"
        )

    def call(self, data, training=True):
        transformation = self.get_random_transformation(data, training=training)
        if isinstance(data, dict):
            is_batched = self._is_batched(data["images"])
            if is_batched:
                data["images"] = self.transform_images(
                    self.backend.convert_to_tensor(data["images"]),
                    transformation=transformation,
                    training=training,
                )
            else:
                data["images"] = self.transform_single_image(
                    self.backend.convert_to_tensor(data["images"]),
                    transformation=transformation,
                    training=training,
                )
            if "bounding_boxes" in data:
                if not self.bounding_box_format:
                    raise ValueError(
                        "You passed an input with a 'bounding_boxes' key, "
                        "but you didn't specify a bounding box format. "
                        "Pass a `bounding_box_format` argument to your "
                        f"{self.__class__.__name__} layer, e.g. "
                        "`bounding_box_format='xyxy'`."
                    )
                bounding_boxes = densify_bounding_boxes(
                    data["bounding_boxes"],
                    is_batched=is_batched,
                    backend=self.backend,
                )

                if is_batched:
                    data["bounding_boxes"] = self.transform_bounding_boxes(
                        bounding_boxes,
                        transformation=transformation,
                        training=training,
                    )
                else:
                    data["bounding_boxes"] = self.transform_single_bounding_box(
                        bounding_boxes,
                        transformation=transformation,
                        training=training,
                    )
            if "labels" in data:
                if is_batched:
                    data["labels"] = self.transform_labels(
                        self.backend.convert_to_tensor(data["labels"]),
                        transformation=transformation,
                        training=training,
                    )
                else:
                    data["labels"] = self.transform_single_label(
                        self.backend.convert_to_tensor(data["labels"]),
                        transformation=transformation,
                        training=training,
                    )
            if "segmentation_masks" in data:
                if is_batched:
                    data["segmentation_masks"] = (
                        self.transform_segmentation_masks(
                            data["segmentation_masks"],
                            transformation=transformation,
                            training=training,
                        )
                    )
                else:
                    data["segmentation_masks"] = (
                        self.transform_single_segmentation_mask(
                            data["segmentation_masks"],
                            transformation=transformation,
                            training=training,
                        )
                    )
            return data

        # `data` is just images.
        if self._is_batched(data):
            return self.transform_images(
                self.backend.convert_to_tensor(data),
                transformation=transformation,
                training=training,
            )
        return self.transform_single_image(
            self.backend.convert_to_tensor(data),
            transformation=transformation,
            training=training,
        )

    def _format_single_input_bounding_box(self, bounding_box):
        for key in bounding_box:
            if key == "labels":
                bounding_box[key] = self.backend.numpy.expand_dims(
                    bounding_box[key], axis=0
                )
            if key == "boxes":
                bounding_box[key] = self.backend.numpy.expand_dims(
                    bounding_box[key], axis=0
                )

        return bounding_box

    def _format_single_output_bounding_box(self, bounding_boxes):
        for key in bounding_boxes:
            if key == "labels":
                bounding_boxes[key] = self.backend.numpy.squeeze(
                    bounding_boxes[key], axis=0
                )
            if key == "boxes":
                bounding_boxes[key] = self.backend.numpy.squeeze(
                    bounding_boxes[key], axis=0
                )

        return bounding_boxes

    def get_config(self):
        config = super().get_config()
        if self.bounding_box_format is not None:
            config.update(
                {
                    "bounding_box_format": self.bounding_box_format,
                }
            )
        return config

    def _transform_value_range(
        self, images, original_range, target_range, dtype="float32"
    ):
        """Convert input values from `original_range` to `target_range`.

        This function is intended to be used in preprocessing layers that
        rely upon color values. This allows us to assume internally that
        the input tensor is always in the range `(0, 255)`.

        Args:
            images: the set of images to transform to the target range.
            original_range: the value range to transform from.
            target_range: the value range to transform to.
            dtype: the dtype to compute the conversion with,
                defaults to "float32".

        Returns:
            a new Tensor with values in the target range.

        Example:

        ```python
        original_range = [0, 1]
        target_range = [0, 255]
        images = layer.preprocessing.transform_value_range(
            images,
            original_range,
            target_range
        )
        images = ops.minimum(images + 10, 255)
        images = layer.preprocessing.transform_value_range(
            images,
            target_range,
            original_range
        )
        ```
        """
        if (
            original_range[0] == target_range[0]
            and original_range[1] == target_range[1]
        ):
            return images

        images = self.backend.cast(images, dtype=dtype)
        original_min_value, original_max_value = self._unwrap_value_range(
            original_range, dtype=dtype
        )
        target_min_value, target_max_value = self._unwrap_value_range(
            target_range, dtype=dtype
        )

        # images in the [0, 1] scale
        images = (images - original_min_value) / (
            original_max_value - original_min_value
        )

        scale_factor = target_max_value - target_min_value
        return (images * scale_factor) + target_min_value

    def _unwrap_value_range(self, value_range, dtype="float32"):
        min_value, max_value = value_range
        min_value = self.backend.cast(min_value, dtype=dtype)
        max_value = self.backend.cast(max_value, dtype=dtype)
        return min_value, max_value

    def _compute_affine_matrix(
        self,
        center_x,
        center_y,
        angle,
        translate_x,
        translate_y,
        scale,
        shear_x,
        shear_y,
        height,
        width,
    ):
        """
        #       Scaling          Shear           Rotation
        #     [sx  0   0]    [1   shx  0]   [cos(θ)  -sin(θ)  0]
        # M = [0   sy  0] *  [shy  1   0] * [sin(θ)   cos(θ)  0]
        #     [0   0   1]    [0    0   1]   [0        0       1]

        # a0 = sx * (cos(θ) + shx * sin(θ))
        # a1 = sx * (-sin(θ) + shx * cos(θ))
        # a2 = tx + cx - cx * a0 - cy * a1
        # b0 = sy * (shy * cos(θ) + sin(θ))
        # b1 = sy * (shy * -sin(θ) + cos(θ))
        # b2 = ty + cy - cx * b0 - cy * b1
        """
        ops = self.backend

        degree_to_radian_factor = ops.convert_to_tensor(math.pi / 180.0)

        angle = angle * degree_to_radian_factor
        shear_x = shear_x * degree_to_radian_factor
        shear_y = shear_y * degree_to_radian_factor

        batch_size = ops.shape(angle)[0]
        dtype = angle.dtype
        width = ops.cast(width, dtype)
        height = ops.cast(height, dtype)
        cx = center_x * (width - 1)
        cy = center_y * (height - 1)

        cos_theta = ops.numpy.cos(angle)
        sin_theta = ops.numpy.sin(angle)
        shear_x = ops.numpy.tan(shear_x)
        shear_y = ops.numpy.tan(shear_y)

        a0 = scale * (cos_theta + shear_x * sin_theta)
        a1 = scale * (-sin_theta + shear_x * cos_theta)
        a2 = translate_x + cx - cx * a0 - cy * a1
        b0 = scale * (shear_y * cos_theta + sin_theta)
        b1 = scale * (shear_y * -sin_theta + cos_theta)
        b2 = translate_y + cy - cx * b0 - cy * b1
        affine_matrix = ops.numpy.concatenate(
            [
                a0[:, None],
                a1[:, None],
                a2[:, None],
                b0[:, None],
                b1[:, None],
                b2[:, None],
                ops.numpy.zeros((batch_size, 2)),
            ],
            axis=1,
        )

        return affine_matrix
