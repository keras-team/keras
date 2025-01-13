# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras.src import ops
import keras.src.random as random
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (
    BaseImagePreprocessingLayer,
)
import keras.src.backend as K


@keras_export("keras.layers.RandomChoice")
class RandomChoice(BaseImagePreprocessingLayer):
    """A preprocessing layer that randomly selects and applies one layer from a list of layers.

    This layer is useful for creating randomized data augmentation pipelines. During training,
    for each input (or batch of inputs), it randomly selects one layer from the provided list
    and applies it to the input. This allows for diverse augmentations to be applied dynamically.

    **Example:**
    ```python
    # Construct a list of augmentation layers
    layers = [
        keras_cv.layers.RandomFlip("horizontal"),
        keras_cv.layers.RandomRotation(factor=0.2),
        keras_cv.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ]

    # Create the RandomChoice pipeline
    pipeline = keras_cv.layers.RandomChoice(layers=layers, batchwise=True)

    # Apply the pipeline to a batch of images
    augmented_images = pipeline(images)
    ```

    **Args:**
        layers: A list of `keras.Layers` instances. Each layer should subclass
            `BaseImagePreprocessingLayer`. During augmentation, one layer will be
            randomly selected and applied to the input.
        auto_vectorize: Boolean, whether to use vectorized operations to apply the
            augmentations. This can significantly improve performance but requires
            that all layers in `layers` support vectorization. Defaults to `False`.
        batchwise: Boolean, whether to apply the same randomly selected layer to
            the entire batch of inputs. When `True`, the entire batch is passed to
            a single layer. When `False`, each input in the batch is processed by
            an independently selected layer. Defaults to `False`.
        seed: Integer, used to seed the random number generator for reproducibility.
            Defaults to `None`.

    **Call Arguments:**
        inputs: A single image tensor (rank 3), a batch of image tensors (rank 4),
            or a dictionary of tensors. The input will be augmented by one randomly
            selected layer from the `layers` list.

    **Returns:**
        Augmented inputs, with the same shape and structure as the input.

    **Notes:**
        - When `batchwise=True`, the same layer is applied to all inputs in the batch.
        - When `batchwise=False`, each input in the batch is processed by an independently
          selected layer, which can lead to more diverse augmentations.
        - All layers in the `layers` list must support the same input shape and dtype.

    **Example with Batchwise Augmentation:**
    ```python
    layers = [
        keras_cv.layers.RandomFlip("horizontal"),
        keras_cv.layers.RandomRotation(factor=0.2),
    ]
    pipeline = keras_cv.layers.RandomChoice(layers=layers, batchwise=True)
    augmented_images = pipeline(images)  # Same layer applied to the entire batch
    ```

    **Example with Per-Image Augmentation:**
    ```python
    layers = [
        keras_cv.layers.RandomFlip("horizontal"),
        keras_cv.layers.RandomRotation(factor=0.2),
    ]
    pipeline = keras_cv.layers.RandomChoice(layers=layers, batchwise=False)
    augmented_images = pipeline(images)  # Each image processed by a random layer
    ```
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
        if K.backend() == "jax":
            self.seed_generator = random.SeedGenerator(seed)

    def _augment(self, inputs, seed=None):
        if K.backend() == "jax":
            selected_op = ops.floor(random.uniform(
                shape=(),
                minval=0,
                maxval=len(self.layers),
                dtype="float32",
                seed=seed
            ))
        else:
            selected_op = ops.floor(random.uniform(
                shape=(),
                minval=0,
                maxval=len(self.layers),
                dtype="float32",
                seed=self.seed
            ))
        output = inputs
        for i, layer in enumerate(self.layers):
            condition = ops.equal(selected_op, float(i))
            output = ops.cond(
                condition,
                lambda l=layer: l(inputs),
                lambda: output
            )
        return output

    def call(self, inputs):
        if isinstance(inputs, dict):
            return {key: self._call_single(input_tensor) for
                    key, input_tensor in inputs.items()}
        else:
            return self._call_single(inputs)

    def _call_single(self, inputs):
        inputs_rank = len(inputs.shape)
        is_single_sample = ops.equal(inputs_rank, 3)
        is_batch = ops.equal(inputs_rank, 4)

        if K.backend() == "jax":
            seed = self.seed_generator.next()
        else:
            seed = None

        def augment_single():
            return self._augment(inputs, seed=seed)

        def augment_batch():
            if self.batchwise:
                return self._augment(inputs, seed=seed)
            else:
                batch_size = ops.shape(inputs)[0]
                augmented = []
                for i in range(batch_size):
                    if K.backend() == "jax":
                        seed_i = self.seed_generator.next()
                        augmented.append(self._augment(inputs[i], seed=seed_i))
                    else:
                        augmented.append(self._augment(inputs[i]))
                return ops.stack(augmented)

        return ops.cond(is_batch, augment_batch, augment_single)

    def transform_images(self, images, transformation=None, training=True):
        if not training:
            return images
        return self.call(images)

    def transform_labels(self, labels, transformation=None, training=True):
        if not training:
            return labels
        return self.call(labels)

    def transform_bounding_boxes(self, bboxes, transformation=None, training=True):
        if not training:
            return bboxes
        return self.call(bboxes)

    def transform_segmentation_masks(self, masks, transformation=None, training=True):
        if not training:
            return masks
        return self.call(masks)

    def get_config(self):
        config = super().get_config()
        config.update({
            "layers": self.layers,
            "auto_vectorize": self.auto_vectorize,
            "batchwise": self.batchwise,
            "seed": self.seed,
        })
        return config