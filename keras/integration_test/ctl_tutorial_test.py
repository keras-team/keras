# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests that Custom Training Loop docs match actual behavior.

The tutorial at https://www.tensorflow.org/tutorials/distribute/custom_training,
defined at
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb
makes several statements about

  * ways to reduce loss terms to the actual training loss, and
  * how they compare to the built-in behavior of Keras Model.fit().

This test verifies that these statements match the actual behavior,
under a variety of distribution strategies.
"""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.distribute import strategy_combinations


def make_compute_loss_fn(variant, loss_object, GLOBAL_BATCH_SIZE):
    """Returns the `compute_loss()` function as defined in the tutorial."""

    if variant == "basic":
        # The basic form of the loss function, shown verbatim in the tutorial.
        def compute_loss(labels, predictions, model_losses):
            per_example_loss = loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss)
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            return loss

    elif variant == "fixed_batch_size":
        # The variant that adds a fixed `global_batch_size=` arg
        # (described but not shown verbatim).
        def compute_loss(labels, predictions, model_losses):
            per_example_loss = loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE
            )
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            return loss

    elif variant == "balanced":
        # The variant that scales the loss to balance out varying batch sizes
        # (described but not shown verbatim).
        def compute_loss(labels, predictions, model_losses):
            per_example_loss = loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss)
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            observed_global_batch_size = (
                tf.distribute.get_strategy().num_replicas_in_sync
                * tf.shape(per_example_loss)[0]
            )
            loss *= tf.math.divide(
                tf.cast(observed_global_batch_size, tf.float32),
                tf.cast(GLOBAL_BATCH_SIZE, tf.float32),
            )
            return loss

    else:
        raise ValueError(f"Unknown {variant=}")

    return compute_loss


def create_dataset(global_batch_size):
    """Creates the dataset for ImpliedExampleWeightsTest.

    It contains two batches: the first has full size, the second just 1 element.
    The i-th element `(x,y)` has model input `x = onehot(i)` and label `y = 0`.
    """
    n = global_batch_size + 1
    ds = tf.data.Dataset.from_tensor_slices((tf.eye(n), tf.zeros([n, 1])))
    ds = ds.batch(global_batch_size)
    return ds


def create_model(n):
    """Creates the model for ImpliedExampleWeightsTest.

    The model has three trainable weights of interest, all initialized to 1.0:

      * "predicting/kernel:0" of shape [n, 1] maps a one-hot encoded input to
        the model output. When used with the MeanAbsoluteError loss, an input
        onehot(i) produces a gradient onehot(i) for this weight, subject to
        the training loop's loss reduction across examples.
      * "activity_regularized/kernel:0" of shape [n, 1] has an activity
        regularizer loss in the model so that input onehot(i) produces a
        gradient of 1/batch_size * onehot(i) for this weight.
      * "weight_regularized:0" of shape [1] has a weight regularizer loss in
        the model that produces a gradient of 1 for this weight, independent
        of batch size.
    """
    inputs = tf.keras.Input(shape=(n,), name="inputs")

    predicting = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer="ones", name="predicting"
    )
    activity_regularized = tf.keras.layers.Dense(
        1,
        use_bias=False,
        kernel_initializer="ones",
        activity_regularizer=tf.keras.regularizers.L1(l1=1.0),
        name="activity_regularized",
    )
    weight_regularized = tf.keras.layers.Dense(
        1,
        kernel_initializer="zeros",
        bias_initializer="ones",
        bias_regularizer=tf.keras.regularizers.L1(l1=1.0),
        name="weight_regularized",
    )

    # Make outputs = predicting(inputs), depending on the other Layers as well.
    add = tf.keras.layers.Add(name="add")
    multiply = tf.keras.layers.Multiply(name="multiply")
    outputs = add(
        [
            predicting(inputs),
            multiply(
                [np.array([[0.0]], np.float32), activity_regularized(inputs)]
            ),
            multiply(
                [np.array([[0.0]], np.float32), weight_regularized(inputs)]
            ),
        ]
    )

    model = tf.keras.Model(inputs, outputs)
    return model


def create_loss(**kwargs):
    """Returns the loss to be used with the model from create_model()."""
    return tf.keras.losses.MeanAbsoluteError(**kwargs)


def create_optimizer(learning_rate):
    """Returns the optimizer that applies gradients in the most obvious way."""
    return tf.keras.optimizers.SGD(learning_rate)


def get_expected_example_weights(
    ctl_variant, *, local_batch_size, num_replicas_in_sync
):
    """Returns the weights that examples have in the gradient updates seen."""

    global_batch_size = local_batch_size * num_replicas_in_sync
    n = global_batch_size + 1
    num_batches = 2

    expected = dict(
        # Examples in a full batch receive the expected gradient weight,
        # independent of the CTL variant.
        example_prediction_fullbatch=1.0,
        example_activity_fullbatch=1.0,
    )
    if ctl_variant == "basic":
        # In the basic variant of the CTL, when a batch of size 1 hits a
        # replica, the singleton example receives the weight that is
        # normally spread evenly across the local_batch_size.
        expected["example_prediction_singleton"] = local_batch_size
        expected["example_activity_singleton"] = local_batch_size
        # Weight regularization applies equally in each batch,
        # irrespective of its size.
        expected["total_weight_regularization"] = num_batches
    elif ctl_variant == "fixed_batch_size":
        # In the CTL variant that fixes GLOBAL_BATCH_SIZE for the reduction
        # of prediction losses, the weight of a singleton example is
        # reverted to normal for prediction, but activity and weight
        # regularization behaves as in the "basic" variant.
        expected["example_prediction_singleton"] = 1.0
        expected["example_activity_singleton"] = local_batch_size
        expected["total_weight_regularization"] = num_batches
    elif ctl_variant == "balanced":
        # The CTL variant that corrects both prediction and regularization
        # losses for the batch size achieves equal weights of examples
        # both for the prediction and for an activity regularizer
        expected["example_prediction_singleton"] = 1.0
        expected["example_activity_singleton"] = 1.0
        # Weight regularization, in sync with the other loss terms,
        # applies proportional to the number of examples.
        expected["total_weight_regularization"] = n / global_batch_size
    return expected


class MaybeStrategyScope:
    """Provides a context allowing no distribution strategy."""

    def __init__(self, strategy):
        self._strategy = strategy
        self._scope = None

    def __enter__(self):
        if self._strategy:
            self._scope = self._strategy.scope()
            self._scope.__enter__()

    def __exit__(self, exc_type, value, traceback):
        if self._strategy:
            self._scope.__exit__(exc_type, value, traceback)
            self._scope = None


class ImpliedExampleWeightsTest(tf.test.TestCase, parameterized.TestCase):
    """Tests weights of loss terms depending on batch size and training loop."""

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            strategy=strategy_combinations.all_strategies
            + strategy_combinations.multiworker_strategies
            + [None],
            ctl_variant=["basic", "fixed_batch_size", "balanced"],
        )
    )
    def test_ctl(self, strategy, ctl_variant):
        """Tests a variant of the CTL under a distribution strategy."""
        if strategy is None:
            num_replicas_in_sync = 1
        else:
            num_replicas_in_sync = strategy.num_replicas_in_sync

        local_batch_size = 2  # For a full batch; greater than 1.
        global_batch_size = local_batch_size * num_replicas_in_sync
        ds = create_dataset(global_batch_size)
        if strategy is not None:
            ds = strategy.experimental_distribute_dataset(ds)

        n = global_batch_size + 1
        learning_rate = 0.01
        with MaybeStrategyScope(strategy):
            model = create_model(n)
            loss_object = create_loss(reduction=tf.keras.losses.Reduction.NONE)
            compute_loss = make_compute_loss_fn(
                ctl_variant, loss_object, global_batch_size
            )
            optimizer = create_optimizer(learning_rate)

            def train_step(inputs):
                x, labels = inputs
                with tf.GradientTape() as tape:
                    predictions = model(x, training=True)
                    loss = compute_loss(labels, predictions, model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )
                return loss

            @tf.function
            def wrapped_train_step(inputs):
                if strategy is None:
                    return train_step(inputs)
                else:
                    per_replica_losses = strategy.run(
                        train_step, args=(inputs,)
                    )
                    return strategy.reduce(
                        tf.distribute.ReduceOp.SUM,
                        per_replica_losses,
                        axis=None,
                    )

            num_epochs = 1
            num_batches = 0
            for epoch in range(num_epochs):
                total_loss = 0.0
                for x in ds:
                    total_loss += wrapped_train_step(x)
                    num_batches += 1
                train_loss = total_loss / num_batches
                self.assertTrue(tf.math.is_finite(train_loss).numpy())

        self.assertEqual(num_batches, 2)

        expected = get_expected_example_weights(
            ctl_variant,
            local_batch_size=local_batch_size,
            num_replicas_in_sync=num_replicas_in_sync,
        )
        self.assert_implied_example_weights(
            model,
            **expected,
            rtol=1e-6 if strategy is None else 1e-4,
            learning_rate=learning_rate,
            global_batch_size=global_batch_size,
        )

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            strategy=strategy_combinations.all_strategies
            + strategy_combinations.multiworker_strategies
            + [None],
        )
    )
    def test_fit(self, strategy):
        """Tests Model.fit()."""
        if strategy is None:
            num_replicas_in_sync = 1
        else:
            num_replicas_in_sync = strategy.num_replicas_in_sync

        local_batch_size = 2  # For a full batch; greater than 1.
        global_batch_size = local_batch_size * num_replicas_in_sync
        ds = create_dataset(global_batch_size)

        n = global_batch_size + 1
        learning_rate = 0.01
        with MaybeStrategyScope(strategy):
            model = create_model(n)
            model.compile(
                optimizer=create_optimizer(learning_rate), loss=create_loss()
            )
        epochs = 1
        steps_per_epoch = 2
        model.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch)

        expected = get_expected_example_weights(
            ctl_variant="basic",  # The tutorial claims this consistency!
            local_batch_size=local_batch_size,
            num_replicas_in_sync=num_replicas_in_sync,
        )
        self.assert_implied_example_weights(
            model,
            **expected,
            rtol=1e-6 if strategy is None else 1e-4,
            learning_rate=learning_rate,
            global_batch_size=global_batch_size,
        )

    def assert_implied_example_weights(
        self,
        model,
        *,
        learning_rate,
        global_batch_size,
        rtol,
        example_prediction_fullbatch,
        example_prediction_singleton,
        example_activity_fullbatch,
        example_activity_singleton,
        total_weight_regularization,
    ):
        """Checks model.weights for the expected effects of training."""
        model_weights = {
            v.name: self._get_var_value(v).numpy()
            for v in model.trainable_variables
        }

        # The total weight received by each one-hot example in the prediction
        # loss is the change of its corresponding weight from the initial
        # value 1, adjusted for the expected averaging by global_batch_size and
        # scaling by SGD's learning_rate.
        predicting_kernel = model_weights["predicting/kernel:0"]
        example_prediction_weights = (
            (1.0 - predicting_kernel) / learning_rate * global_batch_size
        )
        # There was one full batch of examples, followed by a singleton.
        self.assertEqual(predicting_kernel.shape, (global_batch_size + 1, 1))
        # Check the examples in the full batch.
        actual_example_prediction_fullbatch = self.reduce_assert_equal(
            example_prediction_weights[:-1, 0]
        )
        self.assertAllClose(
            example_prediction_fullbatch,
            actual_example_prediction_fullbatch,
            rtol=rtol,
        )
        # Check the singleton example after the full batch.
        actual_example_prediction_singleton = example_prediction_weights[-1, 0]
        self.assertAllClose(
            example_prediction_singleton,
            actual_example_prediction_singleton,
            rtol=rtol,
        )

        # Analogous to predictions, check weights for acticity regularization.
        activity_regularized_kernel = model_weights[
            "activity_regularized/kernel:0"
        ]
        example_activity_weights = (
            (1.0 - activity_regularized_kernel)
            / learning_rate
            * global_batch_size
        )
        self.assertEqual(
            activity_regularized_kernel.shape, (global_batch_size + 1, 1)
        )
        actual_example_activity_fullbatch = self.reduce_assert_equal(
            example_activity_weights[:-1, 0]
        )
        self.assertAllClose(
            example_activity_fullbatch,
            actual_example_activity_fullbatch,
            rtol=rtol,
        )
        actual_example_activity_singleton = example_activity_weights[-1, 0]
        self.assertAllClose(
            example_activity_singleton,
            actual_example_activity_singleton,
            rtol=rtol,
        )

        # The total weight of weight regularization is the change of this
        # (otherwise unused) bias term from its initial value 1,
        # adjusted for the expected scaling by SGD's learning_rate.
        actual_total_weight_reguarization = (
            1.0 - model_weights["weight_regularized/bias:0"][0]
        ) / learning_rate
        self.assertAllClose(
            total_weight_regularization,
            actual_total_weight_reguarization,
            rtol=rtol,
        )

    def reduce_assert_equal(self, x):
        """Returns first element of x and asserts all others are equal."""
        result = x[0]
        for i, value in enumerate(x[1:]):
            self.assertAllEqual(result, value, msg=f"at position {i=}")
        return result

    def _get_var_value(self, var):
        """Returns the (unique) value of a (possibly distributed) Variable."""
        if hasattr(var, "values"):  # Distributed.
            result = self.reduce_assert_equal([v.value() for v in var.values])
        else:
            result = var.value()
        return result


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
