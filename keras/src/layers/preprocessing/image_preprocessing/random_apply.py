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

from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
import tensorflow as tf

@keras_export("keras.layers.RandomApply")
class RandomApply(BaseImagePreprocessingLayer):
    """Apply provided layer to random elements in a batch.

    Args:
        layer: a keras `Layer` or `BaseImagePreprocessingLayer`. This layer will
            be applied to randomly chosen samples in a batch. Layer should not
            modify the size of provided inputs.
        rate: controls the frequency of applying the layer. 1.0 means all
            elements in a batch will be modified. 0.0 means no elements will be
            modified. Defaults to 0.5.
        batchwise: (Optional) bool, whether to pass entire batches to the
            underlying layer. When set to true, only a single random sample is
            drawn to determine if the batch should be passed to the underlying
            layer.
        auto_vectorize: bool, whether to use tf.vectorized_map or tf.map_fn for
            batched input. Setting this to True might give better performance
            but currently doesn't work with XLA. Defaults to False.
        seed: integer, controls random behaviour.

    Example:
    ```
    # Let's declare an example layer that will set all image pixels to zero.
    zero_out = keras.layers.Lambda(lambda x: {"images": 0 * x["images"]})

    # Create a small batch of random, single-channel, 2x2 images:
    images = tf.random.stateless_uniform(shape=(5, 2, 2, 1), seed=[0, 1])
    print(images[..., 0])
    # <tf.Tensor: shape=(5, 2, 2), dtype=float32, numpy=
    # array([[[0.08216608, 0.40928006],
    #         [0.39318466, 0.3162533 ]],
    #
    #        [[0.34717774, 0.73199546],
    #         [0.56369007, 0.9769211 ]],
    #
    #        [[0.55243933, 0.13101244],
    #         [0.2941643 , 0.5130266 ]],
    #
    #        [[0.38977218, 0.80855536],
    #         [0.6040567 , 0.10502195]],
    #
    #        [[0.51828027, 0.12730157],
    #         [0.288486  , 0.252975  ]]], dtype=float32)>

    # Apply the layer with 50% probability:
    random_apply = RandomApply(layer=zero_out, rate=0.5, seed=1234)
    outputs = random_apply(images)
    print(outputs[..., 0])
    # <tf.Tensor: shape=(5, 2, 2), dtype=float32, numpy=
    # array([[[0.        , 0.        ],
    #         [0.        , 0.        ]],
    #
    #        [[0.34717774, 0.73199546],
    #         [0.56369007, 0.9769211 ]],
    #
    #        [[0.55243933, 0.13101244],
    #         [0.2941643 , 0.5130266 ]],
    #
    #        [[0.38977218, 0.80855536],
    #         [0.6040567 , 0.10502195]],
    #
    #        [[0.        , 0.        ],
    #         [0.        , 0.        ]]], dtype=float32)>

    # We can observe that the layer has been randomly applied to 2 out of 5
    samples.
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
        super().__init__(seed=seed, **kwargs)
        if not (0 <= rate <= 1.0):
            raise ValueError(f"rate must be in range [0, 1]. Received rate: {rate}")
        self._layer = layer
        self._rate = rate
        self.auto_vectorize = auto_vectorize
        self.batchwise = batchwise
        self.seed = seed
        self.built = True

    def _get_should_augment(self, inputs):
        input_shape = tf.shape(inputs)
        
        if self.batchwise:
            return self._rate > tf.random.uniform(shape=(), seed=self.seed)
        
        batch_size = input_shape[0]
        random_values = tf.random.uniform(shape=(batch_size,), seed=self.seed)
        should_augment = random_values < self._rate
        
        ndims = tf.rank(inputs)
        broadcast_shape = tf.concat(
            [input_shape[:1], tf.ones(ndims - 1, dtype=tf.int32)], 
            axis=0
        )
        return tf.reshape(should_augment, broadcast_shape)

    def _augment_single(self, inputs):
        random_value = tf.random.uniform(shape=(), seed=self.seed)
        should_augment = random_value < self._rate
        
        def apply_layer():
            return self._layer(inputs)
        
        def return_inputs():
            return inputs
        
        return tf.cond(should_augment, apply_layer, return_inputs)

    def _augment_batch(self, inputs):
        should_augment = self._get_should_augment(inputs)
        augmented = self._layer(inputs)
        return tf.where(should_augment, augmented, inputs)

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
            return self._augment_single(inputs)
        
        def augment_batch():
            return self._augment_batch(inputs)
        
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
            "rate": self._rate,
            "layer": self._layer,
            "seed": self.seed,
            "batchwise": self.batchwise,
            "auto_vectorize": self.auto_vectorize,
        })
        return config