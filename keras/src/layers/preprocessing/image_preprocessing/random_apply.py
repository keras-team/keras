from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomApply")
class RandomApply(BaseImagePreprocessingLayer):
    """Preprocessing layer to randomly apply a specified layer during training.

    This layer randomly applies a given transformation layer to inputs based on
    the `rate` parameter. It is useful for stochastic data augmentation to
    improve model robustness. At inference time, the output is identical to
    the input. Call the layer with `training=True` to enable random application.

    Args:
        layer: A `keras.Layer` to apply. The layer must not modify input shape.
        rate: Float between 0.0 and 1.0, representing the probability of
            applying the layer. Defaults to 0.5.
        batchwise: Boolean. If `True`, the decision to apply the layer is made
            for the entire batch. If `False`, it is made independently for each
            input. Defaults to `False`.
        seed: Optional integer to ensure reproducibility.

    Inputs: A tensor or dictionary of tensors. The input type must be compatible
        with the wrapped layer.

    Output: A tensor or dictionary of tensors matching the input structure, with
        the transformation layer randomly applied according to a specified rate.
    """

    def __init__(
        self,
        layer,
        rate=0.5,
        batchwise=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not (0 <= rate <= 1.0):
            raise ValueError(
                f"rate must be in range [0, 1]. Received rate: {rate}"
            )
        self._layer = layer
        self._rate = rate
        self.batchwise = batchwise
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.built = True

    def _should_augment(self, batch_size=None):
        if batch_size is None:
            return (
                self.backend.random.uniform(
                    shape=(),
                    seed=self._get_seed_generator(self.backend._backend),
                )
                > 1.0 - self._rate
            )
        else:
            return (
                self.backend.random.uniform(
                    shape=(batch_size,),
                    seed=self._get_seed_generator(self.backend._backend),
                )
                > 1.0 - self._rate
            )

    def _batch_augment(self, inputs):
        if self.batchwise:
            if self._should_augment():
                return self._layer(inputs)
            return inputs

        batch_size = ops.shape(inputs)[0]
        should_augment = self._should_augment(batch_size)
        should_augment = ops.reshape(should_augment, (-1, 1, 1, 1))

        augmented = self._layer(inputs)
        return self.backend.numpy.where(should_augment, augmented, inputs)

    def transform_images(self, images, transformation, training=True):
        if not training or transformation is None:
            return images

        should_augment = transformation["should_augment"]
        should_augment = ops.reshape(should_augment, (-1, 1, 1, 1))

        if hasattr(self._layer, "get_random_transformation"):
            layer_transform = self._layer.get_random_transformation(
                images, training=training
            )
            augmented = self._layer.transform_images(
                images, layer_transform, training=training
            )
        else:
            augmented = self._layer(images)

        return self.backend.numpy.where(should_augment, augmented, images)

    def call(self, inputs, training=True):
        if not training:
            return inputs
        if isinstance(inputs, dict):
            result = {}
            for key, value in inputs.items():
                result[key] = self._batch_augment(value)
            return result

        return self._batch_augment(inputs)

    def transform_labels(self, labels, transformation, training=True):
        if not training or transformation is None:
            return labels

        should_augment = transformation["should_augment"]
        should_augment = ops.reshape(should_augment, (-1, 1))

        if hasattr(self._layer, "transform_labels"):
            layer_transform = self._layer.get_random_transformation(
                labels, training=training
            )
            augmented = self._layer.transform_labels(
                labels, layer_transform, training=training
            )
        else:
            augmented = self._layer(labels)

        return self.backend.numpy.where(should_augment, augmented, labels)

    def transform_bounding_boxes(self, bboxes, transformation, training=True):
        if not training or transformation is None:
            return bboxes

        should_augment = transformation["should_augment"]

        if hasattr(self._layer, "transform_bounding_boxes"):
            layer_transform = self._layer.get_random_transformation(
                bboxes, training=training
            )
            augmented = self._layer.transform_bounding_boxes(
                bboxes, layer_transform, training=training
            )
        else:
            augmented = self._layer(bboxes)

        return self.backend.numpy.where(should_augment, augmented, bboxes)

    def transform_segmentation_masks(
        self, masks, transformation, training=True
    ):
        if not training or transformation is None:
            return masks

        should_augment = transformation["should_augment"]

        if hasattr(self._layer, "transform_segmentation_masks"):
            layer_transform = self._layer.get_random_transformation(
                masks, training=training
            )
            augmented = self._layer.transform_segmentation_masks(
                masks, layer_transform, training=training
            )
        else:
            augmented = self._layer(masks)

        return self.backend.numpy.where(should_augment, augmented, masks)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self._rate,
                "layer": self._layer,
                "seed": self.seed,
                "batchwise": self.batchwise,
            }
        )
        return config
