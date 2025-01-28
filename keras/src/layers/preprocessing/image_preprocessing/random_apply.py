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

    Inputs: A tensor (rank 3 for single input, rank 4 for batch input). The
        input can have any dtype and range.

    Output: A tensor with the same shape and dtype as the input, with the
        transformation layer applied to selected inputs.
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

    def get_random_transformation(self, inputs=None, training=True):
        if not training:
            return None
        if inputs is None:
            return {"should_augment": False}

        if isinstance(inputs, dict):
            images = inputs["images"]
        else:
            images = inputs

        batch_size = ops.shape(images)[0]
        if self.batchwise:
            do_augment = self._should_augment()
            should_augment = ops.full((batch_size,), do_augment)
        else:
            should_augment = self._should_augment(batch_size)

        should_augment = ops.reshape(should_augment, (batch_size, 1))
        return {"should_augment": should_augment}

    def _batch_augment(self, inputs):
        if self.batchwise:
            if self._should_augment():
                return self._layer(inputs)
            else:
                return inputs

    def _augment(self, inputs):
        if self._should_augment():
            return self._layer(inputs)
        else:
            return inputs

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
        if isinstance(inputs, dict):
            data = {
                "images": self.transform_images(
                    inputs["images"],
                    self.get_random_transformation(inputs, training),
                    training=training,
                )
            }
            if "labels" in inputs:
                data["labels"] = self.transform_labels(
                    inputs["labels"],
                    self.get_random_transformation(inputs, training),
                    training=training,
                )
            return data
        else:
            return self.transform_images(
                inputs,
                self.get_random_transformation(inputs, training),
                training=training,
            )

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
                masks, training=True
            )
            augmented = self._layer.transform_segmentation_masks(
                masks, layer_transform, training=True
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
