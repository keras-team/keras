from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomChoice")
class RandomChoice(BaseImagePreprocessingLayer):
    """A preprocessing layer that randomly selects and applies one layer from a
    list of layers.

    Useful for creating randomized data augmentation pipelines. During
    training, for each input (or batch of inputs), it randomly selects one layer
    from the provided list and applies it to the input. This allows for diverse
    augmentations to be applied dynamically.

    Args:
        layers: A list of `keras.Layers` instances. Each layer should subclass
            `BaseImagePreprocessingLayer`. During augmentation, one layer is
            randomly selected and applied to the input.
        auto_vectorize: Boolean, whether to use vectorized operations to apply
            augmentations. This can greatly improve performance but requires
            that all layers in layers support vectorization. Defaults to False.
        batchwise: Boolean, whether to apply the same randomly selected layer to
            the entire batch of inputs. When True, the entire batch is passed to
            a single layer. When False, each input in the batch is processed by
            an independently selected layer. Defaults to False.
        seed: Integer to seed random number generator for reproducibility.
            Defaults to `None`.

    Call Arguments:
        inputs: Single image tensor (rank 3), batch of image tensors (rank 4),
            or a dictionary of tensors. The input is augmented by one randomly
            selected layer from the `layers` list.

    Returns:
        Augmented inputs, with the same shape and structure as the input.
    """

    def __init__(
        self,
        layers,
        auto_vectorize=False,
        batchwise=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = layers
        self.auto_vectorize = auto_vectorize
        self.batchwise = batchwise
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

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
            selected_op = ops.floor(
                self.backend.random.uniform(
                    shape=(),
                    minval=0,
                    maxval=len(self.layers),
                    dtype="float32",
                    seed=seed,
                )
            )
        else:
            batch_size = input_shape[0]
            selected_op = ops.floor(
                self.backend.random.uniform(
                    shape=(batch_size,),
                    minval=0,
                    maxval=len(self.layers),
                    dtype="float32",
                    seed=seed,
                )
            )

            ndims = len(input_shape)
            ones = [1] * (ndims - 1)
            broadcast_shape = tuple([batch_size] + ones)
            selected_op = self.backend.numpy.reshape(
                selected_op, broadcast_shape
            )

        return {
            "selected_op": selected_op,
            "input_shape": input_shape,
        }

    def transform_images(self, images, transformation, training=True):
        if not training or transformation is None:
            return images

        images = self.backend.cast(images, self.compute_dtype)
        selected_op = transformation["selected_op"]
        output = images

        for i, layer in enumerate(self.layers):
            condition = ops.equal(selected_op, float(i))
            if hasattr(layer, "get_random_transformation"):
                layer_transform = layer.get_random_transformation(
                    images,
                    training=True,
                    seed=self._get_seed_generator(self.backend._backend),
                )
                augmented = layer.transform_images(
                    images, layer_transform, training=True
                )
            else:
                augmented = layer(images)

            output = self.backend.numpy.where(condition, augmented, output)

        return output

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

    def transform_labels(self, labels, transformation=None, training=True):
        if not training:
            return labels
        return self.call(labels)

    def transform_bounding_boxes(
        self, bboxes, transformation=None, training=True
    ):
        if not training:
            return bboxes
        return self.call(bboxes)

    def transform_segmentation_masks(
        self, masks, transformation=None, training=True
    ):
        if not training:
            return masks
        return self.call(masks)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layers": self.layers,
                "auto_vectorize": self.auto_vectorize,
                "batchwise": self.batchwise,
                "seed": self.seed,
            }
        )
        return config
