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

import tensorflow as tf
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.RandomChoice")
class RandomChoice(BaseImagePreprocessingLayer):
    """RandomChoice constructs a pipeline based on provided arguments.

    The implemented policy does the following: for each input provided in
    `call`(), the policy selects a random layer from the provided list of
    `layers`. It then calls the `layer()` on the inputs.

    Example:
    ```python
    # construct a list of layers
    layers = keras_cv.layers.RandAugment.get_standard_policy(
        value_range=(0, 255), magnitude=0.75, magnitude_stddev=0.3
    )
    layers = layers[:4]  # slice out some layers you don't want for whatever
                           reason
    layers = layers + [keras_cv.layers.GridMask()]

    # create the pipeline.
    pipeline = keras_cv.layers.RandomChoice(layers=layers)

    augmented_images = pipeline(images)
    ```

    Args:
        layers: a list of `keras.Layers`. These are randomly inputs during
            augmentation to augment the inputs passed in `call()`. The layers
            passed should subclass `BaseImagePreprocessingLayer`.
        auto_vectorize: whether to use `tf.vectorized_map` or `tf.map_fn` to
            apply the augmentations. This offers a significant performance
            boost, but can only be used if all the layers provided to the
            `layers` argument support auto vectorization.
        batchwise: Boolean, whether to pass entire batches to the
            underlying layer. When set to `True`, each batch is passed to a
            single layer, instead of each sample to an independent layer. This
            is useful when using `MixUp()`, `CutMix()`, `Mosaic()`, etc.
            Defaults to `False`.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        layers,
        auto_vectorize=False,
        batchwise=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.layers = layers
        self.auto_vectorize = auto_vectorize
        self.batchwise = batchwise
        self.seed = seed

    def _curry_call_layer(self, inputs, layer):
        def call_layer():
            return layer(inputs)

        return call_layer

    def _augment(self, inputs):
        selected_op = tf.random.uniform(
            (), minval=0, maxval=len(self.layers), dtype=tf.int32, seed=self.seed
        )
        branch_fns = [
            (i, self._curry_call_layer(inputs, layer))
            for (i, layer) in enumerate(self.layers)
        ]
        return tf.switch_case(
            branch_index=selected_op,
            branch_fns=branch_fns,
            default=lambda: inputs,
        )

    def call(self, inputs):
        if isinstance(inputs, dict):
            return {key: self._call_single(input_tensor) for 
                    key, input_tensor in inputs.items()}
        else:
            return self._call_single(inputs)

    def _call_single(self, inputs):
        inputs_rank = tf.rank(inputs)
        is_single_sample = tf.equal(inputs_rank, 3)
        is_batch = tf.equal(inputs_rank, 4)
        
        def augment_single():
            return self._augment(inputs)
        
        def augment_batch():
            if self.batchwise:
                return self._augment(inputs)
            else:
                return tf.map_fn(self._augment, inputs)
        
        condition = tf.logical_or(is_single_sample, is_batch)
        return tf.cond(tf.reduce_all(condition), augment_batch, augment_single)

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