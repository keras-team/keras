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
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
import keras.src.backend as K


@keras_export("keras.layers.RandomApply")
class RandomApply(BaseImagePreprocessingLayer):
    """A preprocessing layer that randomly applies a provided layer to elements in a batch.

    This layer is useful for applying data augmentations or transformations with a specified
    probability. During training, each input (or batch of inputs) has a chance to be transformed
    by the provided layer, controlled by the `rate` parameter. This allows for stochastic
    application of augmentations, which can improve model robustness.

    **Example:**
    ```python
    # Create a layer that zeroes out all pixels in an image
    zero_out = keras.layers.Lambda(lambda x: 0 * x)

    # Create a batch of random 2x2 images
    images = tf.random.uniform(shape=(5, 2, 2, 1))
    print(images[..., 0])
    # <tf.Tensor: shape=(5, 2, 2), dtype=float32, numpy=
    # array([[[0.82, 0.41],
    #         [0.39, 0.32]],
    #
    #        [[0.35, 0.73],
    #         [0.56, 0.98]],
    #
    #        [[0.55, 0.13],
    #         [0.29, 0.51]],
    #
    #        [[0.39, 0.81],
    #         [0.60, 0.11]],
    #
    #        [[0.52, 0.13],
    #         [0.29, 0.25]]], dtype=float32)>

    # Apply the layer with 50% probability
    random_apply = RandomApply(layer=zero_out, rate=0.5, seed=1234)
    outputs = random_apply(images)
    print(outputs[..., 0])
    # <tf.Tensor: shape=(5, 2, 2), dtype=float32, numpy=
    # array([[[0.00, 0.00],
    #         [0.00, 0.00]],
    #
    #        [[0.35, 0.73],
    #         [0.56, 0.98]],
    #
    #        [[0.55, 0.13],
    #         [0.29, 0.51]],
    #
    #        [[0.39, 0.81],
    #         [0.60, 0.11]],
    #
    #        [[0.00, 0.00],
    #         [0.00, 0.00]]], dtype=float32)>

    # Observe that the layer was applied to 2 out of 5 samples.
    ```

    **Args:**
        layer: A `keras.Layer` or `BaseImagePreprocessingLayer` instance. This layer will
            be applied to randomly selected inputs in the batch. The layer should not
            modify the shape of the input.
        rate: A float between 0 and 1, controlling the probability of applying the layer.
            - `1.0` means the layer is applied to all inputs.
            - `0.0` means the layer is never applied.
            Defaults to `0.5`.
        batchwise: A boolean, indicating whether the decision to apply the layer should
            be made for the entire batch at once (`True`) or for each input individually
            (`False`). When `True`, the layer is either applied to the entire batch or
            not at all. When `False`, the layer is applied independently to each input
            in the batch. Defaults to `False`.
        auto_vectorize: A boolean, indicating whether to use vectorized operations for
            batched inputs. This can improve performance but may not work with XLA.
            Defaults to `False`.
        seed: An integer, used to seed the random number generator for reproducibility.
            Defaults to `None`.

    **Call Arguments:**
        inputs: A single input tensor (rank 3), a batch of input tensors (rank 4),
            or a dictionary of tensors. The input will be transformed by the provided
            layer with probability `rate`.

    **Returns:**
        Transformed inputs, with the same shape and structure as the input.

    **Notes:**
        - When `batchwise=True`, the layer is applied to the entire batch or not at all,
          based on a single random decision.
        - When `batchwise=False`, the layer is applied independently to each input in
          the batch, allowing for more fine-grained control.
        - The provided `layer` should not modify the shape of the input, as this could
          lead to inconsistencies in the output.

    **Example with Batchwise Application:**
    ```python
    # Apply a layer to the entire batch with 50% probability
    random_apply = RandomApply(layer=zero_out, rate=0.5, batchwise=True, seed=1234)
    outputs = random_apply(images)  # Either all images are zeroed out or none are
    ```

    **Example with Per-Input Application:**
    ```python
    # Apply a layer to each input independently with 50% probability
    random_apply = RandomApply(layer=zero_out, rate=0.5, batchwise=False, seed=1234)
    outputs = random_apply(images)  # Each image is independently zeroed out or left unchanged
    ```
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
            raise ValueError(f"rate must be in range [0, 1]. Received rate: {rate}")
        self._layer = layer
        self._rate = rate
        self.auto_vectorize = auto_vectorize
        self.batchwise = batchwise
        self.seed = seed
        self.built = True
        if K.backend() == "jax":
            self.seed_generator = random.SeedGenerator(seed)

    def _get_should_augment(self, inputs, seed=None):
        input_shape = ops.shape(inputs)
        
        if self.batchwise:
            return self._rate > random.uniform(shape=(), seed=seed)
        
        batch_size = input_shape[0]
        random_values = random.uniform(shape=(batch_size,), seed=seed)
        should_augment = random_values < self._rate
        
        ndims = len(inputs.shape)
        ones = [1] * (ndims - 1)
        broadcast_shape = tuple([batch_size] + ones)
        
        return ops.reshape(should_augment, broadcast_shape)

    def _augment_single(self, inputs, seed=None):
        random_value = random.uniform(shape=(), seed=seed)
        should_augment = random_value < self._rate
        
        def apply_layer():
            return self._layer(inputs)
        
        def return_inputs():
            return inputs
        
        return ops.cond(should_augment, apply_layer, return_inputs)

    def _augment_batch(self, inputs, seed=None):
        should_augment = self._get_should_augment(inputs, seed=seed)
        augmented = self._layer(inputs)
        return ops.where(should_augment, augmented, inputs)

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
            seed = self.seed
        
        def augment_single():
            return self._augment_single(inputs, seed=seed)
        
        def augment_batch():
            return self._augment_batch(inputs, seed=seed)
        
        condition = ops.logical_or(is_single_sample, is_batch)
        return ops.cond(ops.all(condition), augment_batch, augment_single)

    @staticmethod
    def _num_zero_batches(images):
        """Count number of all-zero batches in the input."""
        num_batches = ops.shape(images)[0]
        flattened = ops.reshape(images, (num_batches, -1))
        any_nonzero = ops.any(ops.not_equal(flattened, 0), axis=1)
        num_non_zero_batches = ops.sum(ops.cast(any_nonzero, dtype="int32"))
        return num_batches - num_non_zero_batches

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
            "rate": self._rate,
            "layer": self._layer,
            "seed": self.seed,
            "batchwise": self.batchwise,
            "auto_vectorize": self.auto_vectorize,
        })
        return config