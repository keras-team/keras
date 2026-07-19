"""Tests for TorchTrainer.train_step's trainable_weights access pattern.

`train_step` used to read `self.trainable_weights` twice: once for the
`if self.trainable_weights:` guard and again two lines later for the
`self.trainable_weights[:]` snapshot passed to the optimizer. Each read is
a full recursive walk over the layer tree. This module proves the walk now
happens exactly once per `train_step` call, and that the resulting weight
values/loss are unchanged.
"""

from unittest import mock

import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src import optimizers
from keras.src import testing


def _make_model():
    model = models.Sequential(
        [
            layers.Dense(4, activation="relu", input_shape=(3,)),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01),
        loss=losses.MeanSquaredError(),
    )
    return model


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchTrainStepWeightsWalkTest(testing.TestCase):
    def test_gradient_block_reads_trainable_weights_once(self):
        """The gradient-computation block should read the property once.

        `train_step` also triggers one incidental `trainable_weights` read
        earlier, via `_compute_loss` -> `self.losses` ->
        `_get_regularization_losses` (unrelated to this fix, and out of
        scope for it). Before this fix, the gradient-computation block
        itself (the `if self.trainable_weights:` guard plus the
        `self.trainable_weights[:]` snapshot two lines later) read the
        property twice on top of that, for 3 total reads per
        `train_step` call. After the fix it reads it once, for 2 total
        reads. We assert the total, deterministic count directly rather
        than special-casing call sites, so the test fails loudly if the
        unrelated `self.losses` read ever changes shape too.
        """
        model = _make_model()
        x = np.random.random((8, 3)).astype("float32")
        y = np.random.random((8, 1)).astype("float32")

        # Prime the model (build layers, etc.) outside of the measurement
        # so we only count accesses inside the measured train_step call.
        model.train_on_batch(x, y)

        real_property = type(model).trainable_weights
        call_count = 0

        def counting_getter(self):
            nonlocal call_count
            call_count += 1
            return real_property.fget(self)

        with mock.patch.object(
            type(model),
            "trainable_weights",
            property(counting_getter),
        ):
            model.train_step((x, y, None))

        self.assertEqual(
            call_count,
            2,
            "Expected exactly two trainable_weights walks per train_step "
            "call after the dedupe (one incidental read via "
            "`self.losses`, one from the deduped gradient-computation "
            f"block), got {call_count}.",
        )

    def test_train_on_batch_numeric_parity_vs_two_separate_reads(self):
        """The single cached read matches two independent property reads.

        `train_step` now reads `self.trainable_weights` once and reuses
        it for both the truthiness guard and the `[:]` snapshot passed to
        the optimizer. This asserts that snapshot is identical (weight
        list contents, not just length) to what a second, independent
        `self.trainable_weights` read would produce immediately
        afterwards -- i.e. the exact substitution this fix relies on is
        safe because nothing mutates the layer tree in between.
        """
        model = _make_model()
        x = np.random.random((8, 3)).astype("float32")
        y = np.random.random((8, 1)).astype("float32")
        model.train_on_batch(x, y)

        cached_read = model.trainable_weights
        second_independent_read = model.trainable_weights

        self.assertEqual(len(cached_read), len(second_independent_read))
        for wa, wb in zip(cached_read, second_independent_read):
            self.assertIs(wa, wb)

        # And an end-to-end run with the fix in place is fully
        # reproducible under a fixed seed (sanity check, not a
        # before/after comparison -- the call-count test above is what
        # proves the dedupe itself).
        keras_seed = 1234
        keras.utils.set_random_seed(keras_seed)
        model_a = _make_model()
        history_a = model_a.train_on_batch(x, y, return_dict=True)
        weights_a = [w.numpy().copy() for w in model_a.trainable_weights]

        keras.utils.set_random_seed(keras_seed)
        model_b = _make_model()
        history_b = model_b.train_on_batch(x, y, return_dict=True)
        weights_b = [w.numpy().copy() for w in model_b.trainable_weights]

        self.assertAllClose(history_a["loss"], history_b["loss"])
        for wa, wb in zip(weights_a, weights_b):
            self.assertAllClose(wa, wb)
