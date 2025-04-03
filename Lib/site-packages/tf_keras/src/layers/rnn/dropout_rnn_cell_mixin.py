# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mixin holding dropout fields for RNN cells."""


import tensorflow.compat.v2 as tf
from tensorflow.tools.docs import doc_controls

from tf_keras.src import backend


@doc_controls.do_not_generate_docs
class DropoutRNNCellMixin:
    """Object that hold dropout related fields for RNN Cell.

    This class is not a standalone RNN cell. It suppose to be used with a RNN
    cell by multiple inheritance. Any cell that mix with class should have
    following fields:
      dropout: a float number within range [0, 1). The ratio that the input
        tensor need to dropout.
      recurrent_dropout: a float number within range [0, 1). The ratio that the
        recurrent state weights need to dropout.
      _random_generator: A backend.RandomGenerator instance, which will be used
        to produce outputs based on the inputs and dropout rate.
    This object will create and cache created dropout masks, and reuse them for
    the incoming data, so that the same mask is used for every batch input.
    """

    def __init__(self, *args, **kwargs):
        self._create_non_trackable_mask_cache()
        super().__init__(*args, **kwargs)

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def _create_non_trackable_mask_cache(self):
        """Create the cache for dropout and recurrent dropout mask.

        Note that the following two masks will be used in "graph function" mode,
        e.g. these masks are symbolic tensors. In eager mode, the `eager_*_mask`
        tensors will be generated differently than in the "graph function" case,
        and they will be cached.

        Also note that in graph mode, we still cache those masks only because
        the RNN could be created with `unroll=True`. In that case, the
        `cell.call()` function will be invoked multiple times, and we want to
        ensure same mask is used every time.

        Also the caches are created without tracking. Since they are not
        pickleable by python when deepcopy, we don't want
        `layer._obj_reference_counts_dict` to track it by default.
        """
        self._dropout_mask_cache = backend.ContextValueCache(
            self._create_dropout_mask
        )
        self._recurrent_dropout_mask_cache = backend.ContextValueCache(
            self._create_recurrent_dropout_mask
        )

    def reset_dropout_mask(self):
        """Reset the cached dropout masks if any.

        This is important for the RNN layer to invoke this in it `call()` method
        so that the cached mask is cleared before calling the `cell.call()`. The
        mask should be cached across the timestep within the same batch, but
        shouldn't be cached between batches. Otherwise it will introduce
        unreasonable bias against certain index of data within the batch.
        """
        self._dropout_mask_cache.clear()

    def reset_recurrent_dropout_mask(self):
        """Reset the cached recurrent dropout masks if any.

        This is important for the RNN layer to invoke this in it call() method
        so that the cached mask is cleared before calling the cell.call(). The
        mask should be cached across the timestep within the same batch, but
        shouldn't be cached between batches. Otherwise it will introduce
        unreasonable bias against certain index of data within the batch.
        """
        self._recurrent_dropout_mask_cache.clear()

    def _create_dropout_mask(self, inputs, training, count=1):
        return _generate_dropout_mask(
            self._random_generator,
            tf.ones_like(inputs),
            self.dropout,
            training=training,
            count=count,
        )

    def _create_recurrent_dropout_mask(self, inputs, training, count=1):
        return _generate_dropout_mask(
            self._random_generator,
            tf.ones_like(inputs),
            self.recurrent_dropout,
            training=training,
            count=count,
        )

    def get_dropout_mask_for_cell(self, inputs, training, count=1):
        """Get the dropout mask for RNN cell's input.

        It will create mask based on context if there isn't any existing cached
        mask. If a new mask is generated, it will update the cache in the cell.

        Args:
          inputs: The input tensor whose shape will be used to generate dropout
            mask.
          training: Boolean tensor, whether its in training mode, dropout will
            be ignored in non-training mode.
          count: Int, how many dropout mask will be generated. It is useful for
            cell that has internal weights fused together.
        Returns:
          List of mask tensor, generated or cached mask based on context.
        """
        if self.dropout == 0:
            return None
        init_kwargs = dict(inputs=inputs, training=training, count=count)
        return self._dropout_mask_cache.setdefault(kwargs=init_kwargs)

    def get_recurrent_dropout_mask_for_cell(self, inputs, training, count=1):
        """Get the recurrent dropout mask for RNN cell.

        It will create mask based on context if there isn't any existing cached
        mask. If a new mask is generated, it will update the cache in the cell.

        Args:
          inputs: The input tensor whose shape will be used to generate dropout
            mask.
          training: Boolean tensor, whether its in training mode, dropout will
            be ignored in non-training mode.
          count: Int, how many dropout mask will be generated. It is useful for
            cell that has internal weights fused together.
        Returns:
          List of mask tensor, generated or cached mask based on context.
        """
        if self.recurrent_dropout == 0:
            return None
        init_kwargs = dict(inputs=inputs, training=training, count=count)
        return self._recurrent_dropout_mask_cache.setdefault(kwargs=init_kwargs)

    def __getstate__(self):
        # Used for deepcopy. The caching can't be pickled by python, since it
        # will contain tensor and graph.
        state = super().__getstate__()
        state.pop("_dropout_mask_cache", None)
        state.pop("_recurrent_dropout_mask_cache", None)
        return state

    def __setstate__(self, state):
        state["_dropout_mask_cache"] = backend.ContextValueCache(
            self._create_dropout_mask
        )
        state["_recurrent_dropout_mask_cache"] = backend.ContextValueCache(
            self._create_recurrent_dropout_mask
        )
        super().__setstate__(state)


def _generate_dropout_mask(generator, ones, rate, training=None, count=1):
    def dropped_inputs():
        return generator.dropout(ones, rate)

    if count > 1:
        return [
            backend.in_train_phase(dropped_inputs, ones, training=training)
            for _ in range(count)
        ]
    return backend.in_train_phase(dropped_inputs, ones, training=training)

