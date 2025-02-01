from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


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
        interpolation: Interpolation mode. Supported values: `nearest`,
            `bilinear`.
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

        if interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

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

        horizontal_shift = self.backend.random.uniform(
            minval=-self.scale,
            maxval=self.scale,
            shape=[batch_size, 1],
            seed=seed,
            dtype=self.compute_dtype,
        )
        vertical_shift = self.backend.random.uniform(
            minval=-self.scale,
            maxval=self.scale,
            shape=[batch_size, 1],
            seed=seed,
            dtype=self.compute_dtype,
        )

        perspective_factor = self.backend.cast(
            self.backend.numpy.concatenate(
                [horizontal_shift, vertical_shift], axis=1
            ),
            dtype=self.compute_dtype,
        )
        return {
            "apply_perspective": apply_perspective,
            "perspective_factor": perspective_factor,
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

    def _perspective_inputs(self, inputs, transformation):
        if transformation is None:
            return inputs

        inputs_shape = self.backend.shape(inputs)
        unbatched = len(inputs_shape) == 3
        if unbatched:
            inputs = self.backend.numpy.expand_dims(inputs, axis=0)

        perspective_factor = transformation["perspective_factor"]
        outputs = self.backend.image.affine_transform(
            inputs,
            transform=self._get_perspective_matrix(perspective_factor),
            interpolation=self.interpolation,
            fill_mode="constant",
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

        if unbatched:
            outputs = self.backend.numpy.squeeze(outputs, axis=0)
        return outputs

    def _get_perspective_matrix(self, perspectives):
        num_perspectives = self.backend.shape(perspectives)[0]
        return self.backend.numpy.concatenate(
            [
                self.backend.numpy.ones(
                    (num_perspectives, 1), dtype=self.compute_dtype
                )
                + perspectives[:, :1],
                perspectives[:, :1],
                self.backend.numpy.zeros((num_perspectives, 1)),
                perspectives[:, 1:],
                self.backend.numpy.ones(
                    (num_perspectives, 1), dtype=self.compute_dtype
                )
                + perspectives[:, 1:],
                self.backend.numpy.zeros((num_perspectives, 1)),
                self.backend.numpy.zeros((num_perspectives, 2)),
            ],
            axis=1,
        )

    def transform_labels(self, labels, transformation, training=True):
        return labels

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
