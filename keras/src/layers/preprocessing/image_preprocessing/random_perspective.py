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


@keras_export("keras.layers.RandomPerspective")
class RandomPerspective(BaseImagePreprocessingLayer):
    """A preprocessing layer that applies random perspective transformations.

    This layer distorts the perspective of input images by shifting their
    corner points, simulating a 3D-like transformation. The amount of distortion
    is controlled by the `factor` and `scale` parameters.

    Args:
        factor: A float or a tuple of two floats.
            Represents the probability of applying the perspective
            transformation to each image in the batch.
            - `factor=0.0` ensures no transformation is applied.
            - `factor=1.0` means the transformation is always applied.
            - If a tuple `(min, max)` is provided, a probability is randomly
              sampled between `min` and `max` for each image.
            - If a single float is given, the probability is sampled between
              `0.0` and the provided float.
            Default is 1.0.
        scale: A float defining the relative amount of perspective shift.
            Determines how much the image corners are displaced, affecting
            the intensity of the perspective effect.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        fill_value: a float represents the value to be filled outside the
            boundaries when `fill_mode="constant"`.
        seed: Integer. Used to create a random seed.

    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
            self,
            factor=1.0,
            scale=0.3,
            interpolation="bilinear",
            fill_value=0.0,
            seed=None,
            data_format=None,
            **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self.scale = scale
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.supports_jit = False

        if scale < 0.0 or scale > 1.0:
            raise ValueError(
                "The `scale` argument should be a number "
                "in the range "
                f"[0,1]. "
                f"Received: scale={scale}"
            )

        if interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
            self.channel_axis = -3
        else:
            self.height_axis = -3
            self.width_axis = -2
            self.channel_axis = -1

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data

        images_shape = self.backend.shape(images)
        unbatched = len(images_shape) == 3
        if unbatched:
            batch_size = 1
        else:
            batch_size = images_shape[0]

        seed = seed or self._get_seed_generator(self.backend._backend)

        transformation_probability = self.backend.random.uniform(
            shape=(batch_size,),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=seed,
        )

        random_threshold = self.backend.random.uniform(
            shape=(batch_size,),
            minval=0.0,
            maxval=1.0,
            seed=seed,
        )
        apply_perspective = random_threshold < transformation_probability

        perspective_factor = self.backend.random.uniform(
            shape=(batch_size, 4, 2),
            minval=-0.5 * self.scale,
            maxval=0.5 * self.scale,
        )

        return {
            "apply_perspective": apply_perspective,
            "perspective_factor": perspective_factor,
            "input_shape": images_shape,
        }

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training and transformation is not None:
            apply_perspective = transformation["apply_perspective"]
            perspective_images = self._perspective_inputs(
                images, transformation
            )

            images = self.backend.numpy.where(
                apply_perspective[:, None, None, None],
                perspective_images,
                images,
            )
        return images

    def get_matrix_by_points(self, startpoints, endpoints):
        batch_size = self.backend.shape(startpoints)[0]

        endpoints = self.backend.numpy.tile(endpoints, [batch_size, 1, 1])

        x1, y1 = startpoints[:, 0, 0], startpoints[:, 0, 1]
        x2, y2 = startpoints[:, 1, 0], startpoints[:, 1, 1]
        x3, y3 = startpoints[:, 2, 0], startpoints[:, 2, 1]
        x4, y4 = startpoints[:, 3, 0], startpoints[:, 3, 1]

        x1p, y1p = endpoints[:, 0, 0], endpoints[:, 0, 1]
        x2p, y2p = endpoints[:, 1, 0], endpoints[:, 1, 1]
        x3p, y3p = endpoints[:, 2, 0], endpoints[:, 2, 1]
        x4p, y4p = endpoints[:, 3, 0], endpoints[:, 3, 1]

        A = self.backend.numpy.stack(
            [
                self.backend.numpy.stack(
                    [
                        x1,
                        y1,
                        self.backend.numpy.ones_like(x1),
                        self.backend.numpy.zeros_like(x1),
                        self.backend.numpy.zeros_like(x1),
                        self.backend.numpy.zeros_like(x1),
                        -x1p * x1,
                        -x1p * y1,
                    ],
                    axis=-1,
                ),
                self.backend.numpy.stack(
                    [
                        self.backend.numpy.zeros_like(x1),
                        self.backend.numpy.zeros_like(x1),
                        self.backend.numpy.zeros_like(x1),
                        x1,
                        y1,
                        self.backend.numpy.ones_like(x1),
                        -y1p * x1,
                        -y1p * y1,
                    ],
                    axis=-1,
                ),
                self.backend.numpy.stack(
                    [
                        x2,
                        y2,
                        self.backend.numpy.ones_like(x2),
                        self.backend.numpy.zeros_like(x2),
                        self.backend.numpy.zeros_like(x2),
                        self.backend.numpy.zeros_like(x2),
                        -x2p * x2,
                        -x2p * y2,
                    ],
                    axis=-1,
                ),
                self.backend.numpy.stack(
                    [
                        self.backend.numpy.zeros_like(x2),
                        self.backend.numpy.zeros_like(x2),
                        self.backend.numpy.zeros_like(x2),
                        x2,
                        y2,
                        self.backend.numpy.ones_like(x2),
                        -y2p * x2,
                        -y2p * y2,
                    ],
                    axis=-1,
                ),
                self.backend.numpy.stack(
                    [
                        x3,
                        y3,
                        self.backend.numpy.ones_like(x3),
                        self.backend.numpy.zeros_like(x3),
                        self.backend.numpy.zeros_like(x3),
                        self.backend.numpy.zeros_like(x3),
                        -x3p * x3,
                        -x3p * y3,
                    ],
                    axis=-1,
                ),
                self.backend.numpy.stack(
                    [
                        self.backend.numpy.zeros_like(x3),
                        self.backend.numpy.zeros_like(x3),
                        self.backend.numpy.zeros_like(x3),
                        x3,
                        y3,
                        self.backend.numpy.ones_like(x3),
                        -y3p * x3,
                        -y3p * y3,
                    ],
                    axis=-1,
                ),
                self.backend.numpy.stack(
                    [
                        x4,
                        y4,
                        self.backend.numpy.ones_like(x4),
                        self.backend.numpy.zeros_like(x4),
                        self.backend.numpy.zeros_like(x4),
                        self.backend.numpy.zeros_like(x4),
                        -x4p * x4,
                        -x4p * y4,
                    ],
                    axis=-1,
                ),
                self.backend.numpy.stack(
                    [
                        self.backend.numpy.zeros_like(x4),
                        self.backend.numpy.zeros_like(x4),
                        self.backend.numpy.zeros_like(x4),
                        x4,
                        y4,
                        self.backend.numpy.ones_like(x4),
                        -y4p * x4,
                        -y4p * y4,
                    ],
                    axis=-1,
                ),
            ],
            axis=1,
        )

        B = self.backend.numpy.stack(
            [x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p], axis=-1
        )
        B = self.backend.numpy.expand_dims(B, axis=-1)

        h = self.backend.linalg.solve(A, B)
        h = self.backend.numpy.reshape(h, [-1, 8])

        return h

    def _perspective_inputs(self, inputs, transformation):
        if transformation is None:
            return inputs

        inputs_shape = self.backend.shape(inputs)
        unbatched = len(inputs_shape) == 3
        if unbatched:
            inputs = self.backend.numpy.expand_dims(inputs, axis=0)

        outputs = self.backend.image.affine_transform(
            inputs,
            transform=self._get_perspective_matrix(transformation),
            interpolation=self.interpolation,
            fill_mode="constant",
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

        if unbatched:
            outputs = self.backend.numpy.squeeze(outputs, axis=0)
        return outputs

    def _get_perspective_matrix(self, transformation):
        perspective_factor = transformation["perspective_factor"]
        input_shape = transformation["input_shape"]

        height, width = (
            input_shape[self.height_axis],
            input_shape[self.width_axis],
        )
        start_points = self.backend.convert_to_tensor(
            [
                [
                    [0.0, 0.0],
                    [width - 1, 0.0],
                    [0.0, height - 1],
                    [width - 1, height - 1],
                ]
            ],
            dtype=self.compute_dtype,
        )

        end_points = start_points + start_points * perspective_factor
        return self.get_matrix_by_points(end_points, start_points)

    def _get_transformed_coordinates(self, x, y, transform):
        a0, a1, a2, b0, b1, b2, c0, c1 = self.backend.numpy.split(
            transform, 8, axis=-1
        )

        x_transformed = (a1 * (y - b2) - b1 * (x - a2)) / (a1 * b0 - a0 * b1)
        y_transformed = (b0 * (x - a2) - a0 * (y - b2)) / (a1 * b0 - a0 * b1)

        return x_transformed, y_transformed

    def transform_bounding_boxes(
            self,
            bounding_boxes,
            transformation,
            training=True,
    ):
        if training:
            if backend_utils.in_tf_graph():
                self.backend.set_backend("tensorflow")

        input_height, input_width = (
            transformation["input_shape"][self.height_axis],
            transformation["input_shape"][self.width_axis],
        )

        bounding_boxes = convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=input_height,
            width=input_width,
        )

        boxes = bounding_boxes["boxes"]

        x0, y0, x1, y1 = self.backend.numpy.split(boxes, 4, axis=-1)

        transform = self._get_perspective_matrix(transformation)
        transform = self.backend.numpy.expand_dims(transform, axis=1)
        transform = self.backend.cast(transform, dtype=self.compute_dtype)

        x_1, y_1 = self._get_transformed_coordinates(x0, y0, transform)
        x_2, y_2 = self._get_transformed_coordinates(x1, y1, transform)
        x_3, y_3 = self._get_transformed_coordinates(x0, y1, transform)
        x_4, y_4 = self._get_transformed_coordinates(x1, y0, transform)

        xs = self.backend.numpy.concatenate([x_1, x_2, x_3, x_4], axis=-1)
        ys = self.backend.numpy.concatenate([y_1, y_2, y_3, y_4], axis=-1)

        min_x = self.backend.numpy.min(xs, axis=-1)
        max_x = self.backend.numpy.max(xs, axis=-1)
        min_y = self.backend.numpy.min(ys, axis=-1)
        max_y = self.backend.numpy.max(ys, axis=-1)

        min_x = self.backend.numpy.expand_dims(min_x, axis=-1)
        max_x = self.backend.numpy.expand_dims(max_x, axis=-1)
        min_y = self.backend.numpy.expand_dims(min_y, axis=-1)
        max_y = self.backend.numpy.expand_dims(max_y, axis=-1)

        boxes = self.backend.numpy.concatenate(
            [min_x, min_y, max_x, max_y], axis=-1
        )

        apply_perspective = transformation["apply_perspective"]

        bounding_boxes["boxes"] = self.backend.numpy.where(
            apply_perspective[:, None, None],
            boxes,
            bounding_boxes["boxes"],
        )

        bounding_boxes = clip_to_image_size(
            bounding_boxes=bounding_boxes,
            height=input_height,
            width=input_width,
            bounding_box_format="xyxy",
        )

        return bounding_boxes

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_segmentation_masks(
            self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "factor": self.factor,
            "scale": self.scale,
            "interpolation": self.interpolation,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        return {**base_config, **config}
