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
        scale=1.0,
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
        height, width = (
            images.shape[self.height_axis],
            images.shape[self.width_axis],
        )

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
            seed=seed,
            dtype=self.compute_dtype,
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

        start_points = self.backend.numpy.repeat(
            start_points, batch_size, axis=0
        )
        end_points = start_points + start_points * perspective_factor

        return {
            "apply_perspective": apply_perspective,
            "start_points": start_points,
            "end_points": end_points,
            "input_shape": images_shape,
        }

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training and transformation is not None:
            images = self._perspective_inputs(images, transformation)
            images = self.backend.cast(images, self.compute_dtype)
        return images

    def _perspective_inputs(self, inputs, transformation):
        if transformation is None:
            return inputs

        inputs_shape = self.backend.shape(inputs)
        unbatched = len(inputs_shape) == 3
        if unbatched:
            inputs = self.backend.numpy.expand_dims(inputs, axis=0)

        start_points = transformation["start_points"]
        end_points = transformation["end_points"]

        outputs = self.backend.image.perspective_transform(
            inputs,
            start_points,
            end_points,
            interpolation=self.interpolation,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

        apply_perspective = transformation["apply_perspective"]
        outputs = self.backend.numpy.where(
            apply_perspective[:, None, None, None],
            outputs,
            inputs,
        )

        if unbatched:
            outputs = self.backend.numpy.squeeze(outputs, axis=0)
        return outputs

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        if training and transformation is not None:
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

            start_points = transformation["start_points"]
            end_points = transformation["end_points"]
            transform = self.backend.image.compute_homography_matrix(
                start_points, end_points
            )
            transform = self.backend.numpy.expand_dims(transform, axis=1)
            transform = self.backend.cast(transform, dtype=self.compute_dtype)

            corners = [
                self._get_transformed_coordinates(x, y, transform)
                for x, y in [(x0, y0), (x1, y1), (x0, y1), (x1, y0)]
            ]
            x_corners, y_corners = zip(*corners)

            xs = self.backend.numpy.stack(x_corners, axis=-1)
            ys = self.backend.numpy.stack(y_corners, axis=-1)

            min_x, max_x = (
                self.backend.numpy.min(xs, axis=-1),
                self.backend.numpy.max(xs, axis=-1),
            )
            min_y, max_y = (
                self.backend.numpy.min(ys, axis=-1),
                self.backend.numpy.max(ys, axis=-1),
            )

            min_x = self.backend.numpy.expand_dims(min_x, axis=-1)
            max_x = self.backend.numpy.expand_dims(max_x, axis=-1)
            min_y = self.backend.numpy.expand_dims(min_y, axis=-1)
            max_y = self.backend.numpy.expand_dims(max_y, axis=-1)

            boxes = self.backend.numpy.concatenate(
                [min_x, min_y, max_x, max_y], axis=-1
            )

            apply_perspective = self.backend.core.convert_to_tensor(
                transformation["apply_perspective"], dtype=boxes.dtype
            )

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

            self.backend.reset()

        return bounding_boxes

    def _get_transformed_coordinates(
        self, x_coords, y_coords, transformation_matrix
    ):
        backend = self.backend

        batch_size = backend.shape(transformation_matrix)[0]

        homogeneous_transform = backend.numpy.concatenate(
            [transformation_matrix, backend.numpy.ones((batch_size, 1, 1))],
            axis=-1,
        )
        homogeneous_transform = backend.numpy.reshape(
            homogeneous_transform, (batch_size, 3, 3)
        )

        inverse_transform = backend.linalg.inv(homogeneous_transform)

        ones_column = backend.numpy.ones_like(x_coords)
        homogeneous_coords = backend.numpy.concatenate(
            [x_coords, y_coords, ones_column], axis=-1
        )

        homogeneous_coords = backend.numpy.moveaxis(homogeneous_coords, -1, -2)
        transformed_coords = backend.numpy.matmul(
            inverse_transform, homogeneous_coords
        )
        transformed_coords = backend.numpy.moveaxis(transformed_coords, -1, -2)

        x_transformed = transformed_coords[..., 0] / transformed_coords[..., 2]
        y_transformed = transformed_coords[..., 1] / transformed_coords[..., 2]

        return x_transformed, y_transformed

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
