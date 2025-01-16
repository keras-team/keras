from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomApply")
class RandomApply(BaseImagePreprocessingLayer):
    """A preprocessing layer that randomly applies a specified layer during
    training.

    This layer randomly applies a given transformation layer to inputs based on
    the `rate` parameter. It is useful for stochastic data augmentation to
    improve model robustness. At inference time, the output is identical to
    the input. Call the layer with `training=True` to enable random application.

    Args:
        layer: A `keras.Layer` to apply. The layer must not modify input shape.
        rate: Float between 0.0 and 1.0, representing the probability of
            applying the layer. Defaults to 0.5.
        batchwise: Boolean. If True, the decision to apply the layer is made for
            the entire batch. If False, it is made independently for each input.
            Defaults to False.
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
        auto_vectorize=False,
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
        self.auto_vectorize = auto_vectorize
        self.batchwise = batchwise
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.built = True

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        if isinstance(data, dict):
            inputs = data["images"]
        else:
            inputs = data

        input_shape = self.backend.shape(inputs)
        if self.batchwise:
            should_augment = self._rate > self.backend.random.uniform(
                shape=(), seed=seed
            )
        else:
            batch_size = input_shape[0]
            random_values = self.backend.random.uniform(
                shape=(batch_size,), seed=seed
            )
            should_augment = random_values < self._rate

            ndims = len(input_shape)
            ones = [1] * (ndims - 1)
            broadcast_shape = tuple([batch_size] + ones)
            should_augment = self.backend.numpy.reshape(
                should_augment, broadcast_shape
            )

        return {
            "should_augment": should_augment,
            "input_shape": input_shape,
        }

    def transform_images(self, images, transformation, training=True):
        if not training or transformation is None:
            return images

        should_augment = transformation["should_augment"]

        if hasattr(self._layer, "get_random_transformation"):
            layer_transform = self._layer.get_random_transformation(
                images, training=True
            )
            augmented = self._layer.transform_images(
                images, layer_transform, training=True
            )
        else:
            augmented = self._layer(images)

        return self.backend.numpy.where(should_augment, augmented, images)

    def call(self, inputs):
        if isinstance(inputs, dict):
            return {
                key: self._call_single(input_tensor)
                for key, input_tensor in inputs.items()
            }
        else:
            return self._call_single(inputs)

    def _call_single(self, inputs):
        transformation = self.get_random_transformation(inputs, training=True)
        return self.transform_images(inputs, transformation, training=True)

    @staticmethod
    def _num_zero_batches(images):
        """Count number of all-zero batches in the input."""
        num_batches = ops.shape(images)[0]
        flattened = ops.reshape(images, (num_batches, -1))
        any_nonzero = ops.any(ops.not_equal(flattened, 0), axis=1)
        num_non_zero_batches = ops.sum(ops.cast(any_nonzero, dtype="int32"))
        return num_batches - num_non_zero_batches

    def transform_labels(self, labels, transformation, training=True):
        if not training or transformation is None:
            return labels
        return self.transform_images(labels, transformation, training)

    def transform_bounding_boxes(self, bboxes, transformation, training=True):
        if not training or transformation is None:
            return bboxes
        return self.transform_images(bboxes, transformation, training)

    def transform_segmentation_masks(
        self, masks, transformation, training=True
    ):
        if not training or transformation is None:
            return masks
        return self.transform_images(masks, transformation, training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self._rate,
                "layer": self._layer,
                "seed": self.seed,
                "batchwise": self.batchwise,
                "auto_vectorize": self.auto_vectorize,
            }
        )
        return config
