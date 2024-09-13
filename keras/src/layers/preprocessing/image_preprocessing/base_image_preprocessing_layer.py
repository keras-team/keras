from keras.src.backend import config
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.validation import (  # noqa: E501
    densify_bounding_boxes,
)
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer


class BaseImagePreprocessingLayer(TFDataLayer):

    _FACTOR_BOUNDS = (-1, 1)

    def __init__(
        self, factor=0.0, bounding_box_format=None, data_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.data_format = data_format or config.image_data_format()
        self._set_factor(factor)

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
            if factor < 0 or factor > self._FACTOR_BOUNDS[1]:
                raise ValueError(error_msg)
            factor = abs(factor)
            lower, upper = [max(-factor, self._FACTOR_BOUNDS[0]), factor]
        else:
            raise ValueError(error_msg)
        self.factor = lower, upper

    def get_random_transformation(self, data, seed=None):
        raise NotImplementedError()

    def augment_images(self, images, transformation, training=True):
        raise NotImplementedError()

    def augment_labels(self, labels, transformation, training=True):
        raise NotImplementedError()

    def augment_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        raise NotImplementedError()

    def augment_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        raise NotImplementedError()

    def augment_single_image(self, image, transformation, training=True):
        images = self.backend.numpy.expand_dims(image, axis=0)
        outputs = self.augment_images(
            images, transformation=transformation, training=training
        )
        return self.backend.numpy.squeeze(outputs, axis=0)

    def augment_single_label(self, label, transformation, training=True):
        labels = self.backend.numpy.expand_dims(label, axis=0)
        outputs = self.augment_labels(
            labels, transformation=transformation, training=training
        )
        return self.backend.numpy.squeeze(outputs, axis=0)

    def augment_single_bounding_box(
        self, bounding_box, transformation, training=True
    ):
        bounding_boxes = self.backend.numpy.expand_dims(bounding_box, axis=0)
        outputs = self.augment_bounding_boxes(
            bounding_boxes, transformation=transformation, training=training
        )
        return self.backend.numpy.squeeze(outputs, axis=0)

    def augment_single_segmentation_mask(
        self, segmentation_mask, transformation, training=True
    ):
        segmentation_masks = self.backend.numpy.expand_dims(
            segmentation_mask, axis=0
        )
        outputs = self.augment_segmentation_masks(
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
                data["images"] = self.augment_images(
                    self.backend.convert_to_tensor(data["images"]),
                    transformation=transformation,
                    training=training,
                )
            else:
                data["images"] = self.augment_single_image(
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
                    data["bounding_boxes"], backend=self.backend
                )
                if is_batched:
                    data["bounding_boxes"] = self.augment_bounding_boxes(
                        bounding_boxes,
                        transformation=transformation,
                        training=training,
                    )
                else:
                    data["bounding_boxes"] = self.augment_single_bounding_box(
                        bounding_boxes,
                        transformation=transformation,
                        training=training,
                    )
            if "labels" in data:
                if is_batched:
                    data["labels"] = self.augment_labels(
                        self.backend.convert_to_tensor(data["labels"]),
                        transformation=transformation,
                        training=training,
                    )
                else:
                    data["labels"] = self.augment_single_label(
                        self.backend.convert_to_tensor(data["labels"]),
                        transformation=transformation,
                        training=training,
                    )
            if "segmentation_masks" in data:
                if is_batched:
                    data["segmentation_masks"] = (
                        self.augment_segmentation_masks(
                            data["segmentation_masks"],
                            transformation=transformation,
                            training=training,
                        )
                    )
                else:
                    data["segmentation_masks"] = (
                        self.augment_single_segmentation_mask(
                            data["segmentation_masks"],
                            transformation=transformation,
                            training=training,
                        )
                    )
            return data

        # `data` is just images.
        if self._is_batched(data):
            return self.augment_images(
                self.backend.convert_to_tensor(data),
                transformation=transformation,
                training=training,
            )
        return self.augment_single_image(
            self.backend.convert_to_tensor(data),
            transformation=transformation,
            training=training,
        )

    def get_config(self):
        config = super().get_config()
        if self.bounding_box_format is not None:
            config.update(
                {
                    "bounding_box_format": self.bounding_box_format,
                }
            )
        return config
