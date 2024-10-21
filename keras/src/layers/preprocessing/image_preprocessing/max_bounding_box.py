from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.MaxBoundingBox")
class MaxBoundingBox(BaseImagePreprocessingLayer):
    """Ensure the maximum number of bounding boxes.

    Args:
        max_number: Desired output number of bounding boxs.
        padding_value: The padding value of the `boxes` and `classes` in
            `bounding_boxes`. Defaults to `-1`.
    """

    def __init__(self, max_number, fill_value=-1, **kwargs):
        super().__init__(**kwargs)
        self.max_number = int(max_number)
        self.fill_value = int(fill_value)

    def transform_images(self, images, transformation=None, training=True):
        return images

    def transform_labels(self, labels, transformation=None, training=True):
        return labels

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        ops = self.backend
        boxes = bounding_boxes["boxes"]
        classes = bounding_boxes["classes"]
        boxes_shape = ops.shape(boxes)
        batch_size = boxes_shape[0]
        num_boxes = boxes_shape[1]

        # Get pad size
        pad_size = ops.numpy.maximum(
            ops.numpy.subtract(self.max_number, num_boxes), 0
        )
        boxes = boxes[:, : self.max_number, ...]
        boxes = ops.numpy.pad(
            boxes,
            [[0, 0], [0, pad_size], [0, 0]],
            constant_values=self.fill_value,
        )
        classes = classes[:, : self.max_number]
        classes = ops.numpy.pad(
            classes, [[0, 0], [0, pad_size]], constant_values=self.fill_value
        )

        # Ensure shape
        boxes = ops.numpy.reshape(boxes, [batch_size, self.max_number, 4])
        classes = ops.numpy.reshape(classes, [batch_size, self.max_number])

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes["classes"] = classes
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation=None, training=True
    ):
        return self.transform_images(segmentation_masks)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict) and "bounding_boxes" in input_shape:
            input_keys = set(input_shape["bounding_boxes"].keys())
            extra_keys = input_keys - set(("boxes", "classes"))
            if extra_keys:
                raise KeyError(
                    "There are unsupported keys in `bounding_boxes`: "
                    f"{list(extra_keys)}. "
                    "Only `boxes` and `classes` are supported."
                )

            boxes_shape = list(input_shape["bounding_boxes"]["boxes"])
            boxes_shape[1] = self.max_number
            classes_shape = list(input_shape["bounding_boxes"]["classes"])
            classes_shape[1] = self.max_number
            input_shape["bounding_boxes"]["boxes"] = boxes_shape
            input_shape["bounding_boxes"]["classes"] = classes_shape
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"max_number": self.max_number})
        return config
