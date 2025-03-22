from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator
from keras.src.utils import backend_utils


@keras_export("keras.layers.RandomApply")
class RandomApply(BaseImagePreprocessingLayer):
    """Apply randomly a list of transformations with a given probability.

    Args:
        transforms: A list of transformation operations to randomly apply.
        factor: A float or a tuple of two floats specifying the probability of
            applying the transformations.
            - `factor=0.0` ensures no transformations are applied.
            - `factor=1.0` means transformations are always applied.
            - A tuple `(min, max)` results in a probability value sampled
              uniformly between `min` and `max` for each image.
            - A single float value specifies a probability sampled between
              `0.0` and the given float.
            Default is `1.0`.
        seed: Integer. Used to create a random seed.

    """

    NOT_SUPPORTED_TRANSFORMATIONS = [
        "CenterCrop",
        "RandomCrop",
        "Resizing",
    ]

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)

    def __init__(
        self,
        transforms,
        factor=1.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transform_layers = transforms

        for transform_layer in transforms:
            if (
                transform_layer.__class__.__name__
                in self.NOT_SUPPORTED_TRANSFORMATIONS
            ):
                raise NotImplementedError(
                    f"The transformation '{transform_layer.__class__.__name__}' is not supported by this implementation. "
                    f"Supported transformations do not include: {', '.join(self.NOT_SUPPORTED_TRANSFORMATIONS)}."
                )

        self._set_factor(factor)
        self.seed = seed
        self.generator = SeedGenerator(seed)

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if backend_utils.in_tf_graph():
            self.backend.set_backend("tensorflow")
            for transform_layer in self.transform_layers:
                transform_layer.backend.set_backend("tensorflow")

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data

        images_shape = self.backend.shape(images)
        rank = len(images_shape)
        if rank == 3:
            batch_size = 1
        elif rank == 4:
            batch_size = images_shape[0]
        else:
            raise ValueError(
                "Expected the input image to be rank 3 or 4. Received "
                f"inputs.shape={images_shape}"
            )

        seed = seed or self._get_seed_generator(self.backend._backend)

        apply_probability = self.backend.random.uniform(
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
        apply_transform = random_threshold < apply_probability

        transformations = {
            "apply_transform": apply_transform,
            "transform_values": {},
        }

        for transform_layer in self.transform_layers:
            name = transform_layer.__class__.__name__
            transformations["transform_values"][name] = (
                transform_layer.get_random_transformation(
                    images, training=training
                )
            )
        return transformations

    def build(self, input_shape):
        for transform_layer in self.transform_layers:
            transform_layer.build(input_shape)

    def transform_images(self, images, transformation, training=True):
        if training:
            images = self.backend.cast(images, self.compute_dtype)
            transform_values = transformation["transform_values"]
            apply_transform = transformation["apply_transform"]
            apply_transform = (
                apply_transform[:, None, None]
                if len(images.shape) == 3
                else apply_transform[:, None, None, None]
            )

            transformed_images = images
            for transform_layer in self.transform_layers:
                name = transform_layer.__class__.__name__
                transform = transform_values[name]
                transformed_images = transform_layer.transform_images(
                    transformed_images, transform, training=training
                )

            images = self.backend.numpy.where(
                apply_transform,
                transformed_images,
                images,
            )

        images = self.backend.cast(images, self.compute_dtype)
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        if training:
            transformed_bounding_boxes = {
                "boxes": self.backend.numpy.copy(bounding_boxes["boxes"]),
                "labels": bounding_boxes["labels"],
            }
            apply_transform = transformation["apply_transform"]
            transform_values = transformation["transform_values"]
            for transform_layer in self.transform_layers:
                name = transform_layer.__class__.__name__
                transform = transform_values[name]
                transformed_bounding_boxes = (
                    transform_layer.transform_bounding_boxes(
                        transformed_bounding_boxes, transform, training=training
                    )
                )

            boxes = bounding_boxes["boxes"]
            transformed_boxes = transformed_bounding_boxes["boxes"]

            boxes = self.backend.numpy.where(
                apply_transform[:, None, None],
                transformed_boxes,
                boxes,
            )

            bounding_boxes["boxes"] = boxes

        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "transforms": self.transform_layers,
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
