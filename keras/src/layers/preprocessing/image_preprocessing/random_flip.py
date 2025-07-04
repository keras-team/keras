from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    clip_to_image_size,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.converters import (  # noqa: E501
    convert_format,
)
from keras.src.random.seed_generator import SeedGenerator
from keras.src.utils import backend_utils

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


@keras_export("keras.layers.RandomFlip")
class RandomFlip(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly flips images during training.

    This layer will flip the images horizontally and or vertically based on the
    `mode` attribute. During inference time, the output will be identical to
    input. Call the layer with `training=True` to flip the input.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        mode: String indicating which flip mode to use. Can be `"horizontal"`,
            `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a
            left-right flip and `"vertical"` is a top-bottom flip. Defaults to
            `"horizontal_and_vertical"`
        seed: Integer. Used to create a random seed.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    _USE_BASE_FACTOR = False

    def __init__(
        self,
        mode=HORIZONTAL_AND_VERTICAL,
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.mode = mode
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        shape = self.backend.core.shape(images)
        if len(shape) == 3:
            flips_shape = (1, 1, 1)
        else:
            flips_shape = (shape[0], 1, 1, 1)

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        flips = self.backend.numpy.less_equal(
            self.backend.random.uniform(shape=flips_shape, seed=seed), 0.5
        )
        return {"flips": flips, "input_shape": shape}

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training:
            return self._flip_inputs(images, transformation)
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        def _flip_boxes_horizontal(boxes):
            x1, x2, x3, x4 = self.backend.numpy.split(boxes, 4, axis=-1)
            outputs = self.backend.numpy.concatenate(
                [1 - x3, x2, 1 - x1, x4], axis=-1
            )
            return outputs

        def _flip_boxes_vertical(boxes):
            x1, x2, x3, x4 = self.backend.numpy.split(boxes, 4, axis=-1)
            outputs = self.backend.numpy.concatenate(
                [x1, 1 - x4, x3, 1 - x2], axis=-1
            )
            return outputs

        def _transform_xyxy(boxes, box_flips):
            bboxes = boxes["boxes"]
            if self.mode in {HORIZONTAL, HORIZONTAL_AND_VERTICAL}:
                bboxes = self.backend.numpy.where(
                    box_flips,
                    _flip_boxes_horizontal(bboxes),
                    bboxes,
                )
            if self.mode in {VERTICAL, HORIZONTAL_AND_VERTICAL}:
                bboxes = self.backend.numpy.where(
                    box_flips,
                    _flip_boxes_vertical(bboxes),
                    bboxes,
                )
            return bboxes

        if training:
            if backend_utils.in_tf_graph():
                self.backend.set_backend("tensorflow")

            flips = self.backend.numpy.squeeze(transformation["flips"], axis=-1)

            if self.data_format == "channels_first":
                height_axis = -2
                width_axis = -1
            else:
                height_axis = -3
                width_axis = -2

            input_height, input_width = (
                transformation["input_shape"][height_axis],
                transformation["input_shape"][width_axis],
            )

            bounding_boxes = convert_format(
                bounding_boxes,
                source=self.bounding_box_format,
                target="rel_xyxy",
                height=input_height,
                width=input_width,
            )

            bounding_boxes["boxes"] = _transform_xyxy(bounding_boxes, flips)

            bounding_boxes = clip_to_image_size(
                bounding_boxes=bounding_boxes,
                height=input_height,
                width=input_width,
                bounding_box_format="xyxy",
            )

            bounding_boxes = convert_format(
                bounding_boxes,
                source="rel_xyxy",
                target=self.bounding_box_format,
                height=input_height,
                width=input_width,
            )

            self.backend.reset()

        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def _flip_inputs(self, inputs, transformation):
        if transformation is None:
            return inputs

        flips = transformation["flips"]
        inputs_shape = self.backend.shape(inputs)
        unbatched = len(inputs_shape) == 3
        if unbatched:
            inputs = self.backend.numpy.expand_dims(inputs, axis=0)

        flipped_outputs = inputs
        if self.data_format == "channels_last":
            horizontal_axis = -2
            vertical_axis = -3
        else:
            horizontal_axis = -1
            vertical_axis = -2

        if self.mode == HORIZONTAL or self.mode == HORIZONTAL_AND_VERTICAL:
            flipped_outputs = self.backend.numpy.where(
                flips,
                self.backend.numpy.flip(flipped_outputs, axis=horizontal_axis),
                flipped_outputs,
            )
        if self.mode == VERTICAL or self.mode == HORIZONTAL_AND_VERTICAL:
            flipped_outputs = self.backend.numpy.where(
                flips,
                self.backend.numpy.flip(flipped_outputs, axis=vertical_axis),
                flipped_outputs,
            )
        if unbatched:
            flipped_outputs = self.backend.numpy.squeeze(
                flipped_outputs, axis=0
            )
        return flipped_outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seed": self.seed,
                "mode": self.mode,
                "data_format": self.data_format,
            }
        )
        return config
