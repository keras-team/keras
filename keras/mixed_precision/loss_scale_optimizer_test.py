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
"""Tests for LossScaleOptimizer."""

import os
from unittest import mock

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import optimizers
from keras.mixed_precision import loss_scale_optimizer
from keras.mixed_precision import test_util as mp_test_util
from keras.optimizers import adam as adam_experimental
from keras.optimizers import optimizer as optimizer_experimental
from keras.optimizers import sgd as sgd_experimental
from keras.optimizers.legacy import adam
from keras.optimizers.legacy import gradient_descent
from keras.optimizers.legacy import optimizer_v2
from keras.optimizers.schedules import learning_rate_schedule
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

# isort: off
from tensorflow.python.framework import (
    test_util as tf_test_utils,
)
from tensorflow.python.platform import tf_logging

# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = tf.distribute.get_strategy


def create_mirrored_strategy():
    if tf.config.list_logical_devices("GPU"):
        return tf.distribute.MirroredStrategy(["cpu:0", "gpu:0"])
    else:
        return tf.distribute.MirroredStrategy(["cpu:0"])


STRATEGY_FNS = [default_strategy_fn, create_mirrored_strategy]


def create_sgd(base_optimizer_cls, *args, **kwargs):
    """Creates an SGD optimizer.

    Will return either the new experimental SGD optimizer subclassing from
    `optimizer_experimental.Optimizer` or the old SGD optimizer subclassing from
    `optimizer_v2.OptimizerV2`, depending on `base_optimizer_cls`.

    Args:
      base_optimizer_cls: What the superclass of the returned SGD optimizer will
        be. Either `optimizer_experimental.Optimizer` or
        `optimizer_v2.OptimizerV2`.
      *args: Arguments to pass to the SGD constructor
      **kwargs: Keyword arguments to pass to the SGD constructor.

    Returns:
      An SGD optimizer.
    """
    if base_optimizer_cls == optimizer_v2.OptimizerV2:
        return gradient_descent.SGD(*args, **kwargs)
    else:
        assert (
            base_optimizer_cls == optimizer_experimental.Optimizer
        ), f"Got invalid base_optimizer_cls: {base_optimizer_cls}"
        return sgd_experimental.SGD(*args, **kwargs)


# TODO(b/215568552): Remove this as the delegation is handled by metaclass.
def create_lso(
    inner_optimizer, dynamic=True, initial_scale=None, dynamic_growth_steps=None
):
    """Creates a LossScaleOptimizer.

    Creates either the new LossScaleOptimizerV3 subclassing from
    `optimizer_experimental.Optimizer` or the old LossScaleOptimizer subclassing
    from `optimizer_v2.OptimizerV2`, depending on the type of `inner_optimizer`.

    Args:
      inner_optimizer: The optimizer to wrap. Either an
        `optimizer_experimental.Optimizer` or an `optimizer_v2.OptimizerV2`.
      dynamic: Whether dynamic loss scaling is used.
      initial_scale: The initial loss scale.
      dynamic_growth_steps: How frequently to increase the dynamic loss scale.

    Returns:
      Returns a LossScaleOptimizerV3 or a LossScaleOptimizer, depending on the
      type of `inner_optimizer`.
    """
    return loss_scale_optimizer.BaseLossScaleOptimizer(
        inner_optimizer,
        dynamic=dynamic,
        initial_scale=initial_scale,
        dynamic_growth_steps=dynamic_growth_steps,
    )


def opt_and_strategy_and_mode_combinations():
    """Returns combinations for running with multiple optimizers and strategies.

    Returns:
      Combinations that run with both OptimizerV2 and the experimental
      optimizer; and with the default strategy and mirrored strategy; and in
      both graph and eager mode.
    """
    # For the experimental optimizer, don't use graph mode directly since it's
    # unsupported. Instead, run both without and with a tf.function, in order to
    # test both graph and eager mode.
    experimental_opt_combinations = test_combinations.combine(
        opt_cls=optimizer_experimental.Optimizer,
        strategy_fn=STRATEGY_FNS,
        mode="eager",
        use_tf_function=[False, True],
    )
    orig_opt_combinations = test_combinations.combine(
        opt_cls=optimizer_v2.OptimizerV2,
        strategy_fn=STRATEGY_FNS,
        mode=["graph", "eager"],
        use_tf_function=False,
    )
    return experimental_opt_combinations + orig_opt_combinations


def opt_combinations_only():
    """Returns two combinations for running with the two base optimizers."""
    experimental_opt_combinations = test_combinations.combine(
        mode="eager", opt_cls=optimizer_experimental.Optimizer
    )
    orig_opt_combination = test_combinations.combine(
        opt_cls=optimizer_v2.OptimizerV2
    )
    return experimental_opt_combinations + orig_opt_combination


@tf_test_utils.with_control_flow_v2
class LossScaleOptimizerTest(tf.test.TestCase, parameterized.TestCase):
    def _run_if_in_graph_mode(self, val):
        # Running only in graph mode is useful, because optimizers sometimes
        # return a value that, in Graph mode, is runnable with self.evaluate.
        # But in Eager mode, the optimizer already does the computations and the
        # return value cannot be run.
        if not tf.executing_eagerly():
            self.evaluate(val)

    def _eval_if_tensor(self, val):
        # Calls self.evaluate on val if val is a Tensor or Variable. This is
        # useful, since hyperparameters are tf.Variables on OptimizerV2 and are
        # Python floats on the experimental optimizer.
        return (
            self.evaluate(val)
            if isinstance(val, (tf.Tensor, tf.Variable))
            else val
        )

    def _run_fn_with_grad_check(self, strategy, var, opt, expected_grad):
        grad_check_fn = mp_test_util.create_identity_with_grad_check_fn(
            expected_grad
        )
        loss = lambda: grad_check_fn(var) / strategy.num_replicas_in_sync
        return lambda: opt.minimize(loss, var_list=[var])

    def testIsInstance(self):
        optimizer = create_lso(sgd_experimental.SGD())
        self.assertIsInstance(
            optimizer, loss_scale_optimizer.BaseLossScaleOptimizer
        )

        optimizer = create_lso(gradient_descent.SGD())
        self.assertIsInstance(
            optimizer, loss_scale_optimizer.BaseLossScaleOptimizer
        )

    @test_combinations.generate(opt_and_strategy_and_mode_combinations())
    def testFixedLossScaleAppliedToLossWithMinimize(
        self, opt_cls, strategy_fn, use_tf_function
    ):
        with strategy_fn().scope() as strategy:
            var = tf.Variable([5.0])
            opt = create_sgd(opt_cls, 2.0)
            loss_scale = 10.0
            opt = create_lso(opt, dynamic=False, initial_scale=loss_scale)
            self.assertEqual(self.evaluate(opt.loss_scale), loss_scale)
            self.assertIsInstance(opt.loss_scale, tf.Tensor)
            # We need num_replicas_in_sync to divide loss_scale, otherwise
            # loss_scale / strategy.num_replicas_in_sync will not be exact,
            # which could lead to assertion failures due to rounding issues.
            self.assertEqual(loss_scale % strategy.num_replicas_in_sync, 0)
            run_fn = self._run_fn_with_grad_check(
                strategy, var, opt, loss_scale / strategy.num_replicas_in_sync
            )
            if use_tf_function:
                run_fn = tf.function(run_fn)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            # The loss is the identity of the variable. Therefore the gradient
            # is 1, and so the variable will be init_val - grad * lr == 5 - 1 *
            # 2 == 3
            self.assertAllClose([3.0], self.evaluate(var))

    def testFixedLossScaleAppliedToLossWithGetGradients(self):
        with tf.Graph().as_default():
            var = tf.Variable([2.0])
            opt = gradient_descent.SGD(1.0)
            loss_scale = 10.0
            opt = loss_scale_optimizer.LossScaleOptimizer(
                opt, dynamic=False, initial_scale=loss_scale
            )
            grad_check_fn = mp_test_util.create_identity_with_grad_check_fn(
                loss_scale
            )
            loss = grad_check_fn(var)
            run_op = opt.get_gradients(loss, [var])
            self.evaluate(tf.compat.v1.global_variables_initializer())
            # This will cause an assertion to run, as
            # mp_test_util.create_identity_with_grad_check_fn added an assertion
            # op.
            self.evaluate(run_op)

    @test_combinations.generate(opt_combinations_only())
    def testDynamicAttrsWithFixedLossScale(self, opt_cls):
        opt = create_sgd(opt_cls)
        opt = create_lso(opt, dynamic=False, initial_scale=2.0)
        self.assertFalse(opt.dynamic)
        self.assertIsNone(opt.dynamic_counter)
        self.assertIsNone(opt.dynamic_growth_steps)

    @test_combinations.generate(opt_combinations_only())
    def testGetScaledLoss(self, opt_cls):
        opt = create_sgd(opt_cls)
        opt = create_lso(opt, dynamic=False, initial_scale=2.0)
        loss = tf.convert_to_tensor(5.0)
        self.assertEqual(10.0, self.evaluate(opt.get_scaled_loss(loss)))
        self.assertEqual(
            10.0, self.evaluate(opt.get_scaled_loss(lambda: loss)())
        )
        loss = tf.convert_to_tensor(5.0, dtype="float16")
        self.assertEqual(10.0, self.evaluate(opt.get_scaled_loss(loss)))
        self.assertEqual(
            10.0, self.evaluate(opt.get_scaled_loss(lambda: loss)())
        )

    @test_combinations.generate(opt_combinations_only())
    def testGetUnscaledGradients(self, opt_cls):
        opt = create_sgd(opt_cls)
        opt = create_lso(opt, dynamic=False, initial_scale=2)
        scaled_grads = [
            tf.convert_to_tensor(3.0),
            None,
            tf.convert_to_tensor(-4.0, dtype="float16"),
        ]
        grads = opt.get_unscaled_gradients(scaled_grads)
        grads = [self.evaluate(g) if g is not None else g for g in grads]
        self.assertEqual([1.5, None, -2.0], grads)

    @test_combinations.generate(opt_combinations_only())
    def testGetUnscaledSparseGradients(self, opt_cls):
        opt = create_sgd(opt_cls)
        opt = create_lso(opt, dynamic=False, initial_scale=2)
        sparse_scaled_grad = tf.IndexedSlices(
            tf.convert_to_tensor([[4.0, 2.0], [8.0, 5.0]]),
            tf.convert_to_tensor([1, 3], dtype="int32"),
            dense_shape=tf.convert_to_tensor([5, 2], dtype="int32"),
        )
        sparse_grad = opt.get_unscaled_gradients([sparse_scaled_grad])[0]
        self.assertIsInstance(sparse_grad, tf.IndexedSlices)
        self.assertAllEqual(
            [[2.0, 1.0], [4.0, 2.5]], self.evaluate(sparse_grad.values)
        )

    @test_combinations.generate(opt_and_strategy_and_mode_combinations())
    def testDynamicLossScale(self, opt_cls, strategy_fn, use_tf_function):
        strategy = strategy_fn()
        learning_rate = 2.0
        expected_gradient = tf.Variable(
            learning_rate / strategy.num_replicas_in_sync
        )
        with strategy.scope():
            var = tf.Variable([5.0])
            opt = create_sgd(opt_cls, learning_rate)
            opt = create_lso(opt, initial_scale=2, dynamic_growth_steps=1)
            self.assertEqual(opt.initial_scale, 2.0)
            self.assertIsInstance(opt.initial_scale, float)
            self.assertEqual(opt.dynamic_growth_steps, 1)
            self.assertIsInstance(opt.dynamic_growth_steps, int)

            self.assertEqual(
                opt.initial_scale % strategy.num_replicas_in_sync, 0
            )
            run_fn = self._run_fn_with_grad_check(
                strategy, var, opt, expected_gradient
            )
            if use_tf_function:
                run_fn = tf.function(run_fn)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            # The loss is the identity of the variable. Therefore the gradient
            # is 1, and so the variable will be init_val - grad * lr == 5 - 1 *
            # 2 == 3
            self.assertAllClose([3.0], self.evaluate(var))

            # Loss scale will be double, so the expected gradient is also
            # doubled.
            self.evaluate(
                expected_gradient.assign(
                    2 * learning_rate / strategy.num_replicas_in_sync
                )
            )
            run_op = strategy.experimental_run(run_fn)
            self._run_if_in_graph_mode(run_op)
            # As before, the 2 is subtracted from the variable, making it's new
            # value 1.
            self.assertAllClose([1.0], self.evaluate(var))

    @test_combinations.generate(opt_combinations_only())
    def testDynamicLossScaleDefaultValues(self, opt_cls):
        opt = create_sgd(opt_cls)
        opt = create_lso(opt)
        self.assertEqual(opt.initial_scale, 2**15)
        self.assertEqual(opt.dynamic_growth_steps, 2000)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertEqual(self.evaluate(opt.loss_scale), 2**15)

    @test_combinations.generate(opt_and_strategy_and_mode_combinations())
    def testClipping(self, opt_cls, strategy_fn, use_tf_function):
        strategy = strategy_fn()
        learning_rate = 2.0
        for clip_type in ("clipnorm", "global_clipnorm", "clipvalue"):
            with strategy.scope(), self.subTest(clip_type=clip_type):
                var = tf.Variable([5.0])
                opt = create_sgd(opt_cls, learning_rate, **{clip_type: 2.0})
                opt = create_lso(opt, initial_scale=2, dynamic_growth_steps=1)
                if isinstance(opt, loss_scale_optimizer.LossScaleOptimizer):
                    # Only OptimizerV2 exposes the clipping attributes
                    self.assertEqual(getattr(opt, clip_type), 2.0)
                self.assertEqual(
                    opt.initial_scale % strategy.num_replicas_in_sync, 0
                )

                loss = lambda: var * 4 / strategy.num_replicas_in_sync
                run_fn = lambda: opt.minimize(loss, var_list=[var])
                if use_tf_function:
                    run_fn = tf.function(run_fn)

                # Test running with clipped gradients
                run_op = strategy.experimental_run(run_fn)
                self.evaluate(tf.compat.v1.global_variables_initializer())
                self._run_if_in_graph_mode(run_op)
                # The gradient is 4 but is clipped to 2, so the variable will be
                # init_val - clipped_grad * lr == 5 - 2 * 2 == 1
                self.assertAllClose([1.0], self.evaluate(var))
                self.assertEqual(self.evaluate(opt.loss_scale), 4)

                if isinstance(opt, loss_scale_optimizer.LossScaleOptimizerV3):
                    # Only OptimizerV2 exposes the clipping attributes, so we
                    # cannot set them on the new optimizer
                    return
                # Test changing the clip amount and running again
                setattr(opt, clip_type, 3.0)
                run_op = strategy.experimental_run(run_fn)
                self._run_if_in_graph_mode(run_op)
                # The gradient is 4 but is clipped to 3, so the variable will be
                # prev_var - clipped_grad * lr == 1 - 3 * 2 == -5
                self.assertAllClose([-5.0], self.evaluate(var))
                self.assertEqual(self.evaluate(opt.loss_scale), 8)

                # Test Inf gradients are still skipped instead of being clipped
                loss = lambda: var * float("Inf")
                run_fn = lambda: opt.minimize(loss, var_list=[var])
                run_op = strategy.experimental_run(run_fn)
                self._run_if_in_graph_mode(run_op)
                self.assertAllClose(
                    [-5.0], self.evaluate(var)
                )  # Var does not change
                self.assertEqual(self.evaluate(opt.loss_scale), 4)

    @test_combinations.generate(opt_and_strategy_and_mode_combinations())
    def testDynamicUpdate(self, opt_cls, strategy_fn, use_tf_function):
        with strategy_fn().scope() as strategy:
            var = tf.Variable([1.0, 2.0])
            opt = create_sgd(opt_cls, 1.0)
            opt = create_lso(opt, initial_scale=2, dynamic_growth_steps=1)

            # Test optimizer with finite gradients
            loss = lambda: var * 2.0 / strategy.num_replicas_in_sync
            run_fn = lambda: opt.minimize(loss, var_list=[var])
            if use_tf_function:
                run_fn = tf.function(run_fn)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            # Gradient is 2, so variable will have 2 subtracted from it
            self.assertAllClose([-1.0, 0.0], self.evaluate(var))
            # Loss scale has doubled from 2 to 4
            self.assertEqual(4.0, self.evaluate(opt.loss_scale))

            # Test optimizer with NaN gradients
            loss = lambda: var * float("NaN")
            run_fn = lambda: opt.minimize(loss, var_list=[var])
            run_op = strategy.experimental_run(run_fn)
            self._run_if_in_graph_mode(run_op)
            # Variable should not change from before, due to NaN gradients.
            self.assertAllClose(self.evaluate(var), [-1.0, 0.0])
            # Loss scale should half due to NaN gradients.
            self.assertEqual(2.0, self.evaluate(opt.loss_scale))

    @test_combinations.generate(opt_and_strategy_and_mode_combinations())
    def testDynamicLossScaleWithFloat16Loss(
        self, opt_cls, strategy_fn, use_tf_function
    ):
        strategy = strategy_fn()
        learning_rate = 2.0
        with strategy.scope():
            var = tf.Variable([5.0])
            opt = create_sgd(opt_cls, learning_rate)
            opt = create_lso(opt, initial_scale=2, dynamic_growth_steps=1)

            def loss():
                return tf.cast(var / strategy.num_replicas_in_sync, "float16")

            run_fn = lambda: opt.minimize(loss, var_list=[var])
            if use_tf_function:
                run_fn = tf.function(run_fn)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            # The loss is the identity of the variable. Therefore the gradient
            # is 1, and so the variable will be init_val - grad * lr == 5 - 1 *
            # 2 == 3
            self.assertAllClose([3.0], self.evaluate(var))

    @test_combinations.generate(opt_and_strategy_and_mode_combinations())
    def testNanOnOneReplicaOnly(self, opt_cls, strategy_fn, use_tf_function):
        if strategy_fn == default_strategy_fn:
            self.skipTest("The test is only useful for non-default strategies")
        if not tf.test.is_gpu_available():
            self.skipTest("Test requires GPU")
        if (
            not tf.executing_eagerly()
            and not tf.compat.v1.control_flow_v2_enabled()
        ):
            self.skipTest(
                "b/181283011: GradientTape does not work properly with "
                "V1 control flow, and opt.minimize uses GradientTape"
            )
        with strategy_fn().scope() as strategy:
            var = tf.Variable([1.0, 2.0])
            opt = create_sgd(opt_cls, 1.0)
            opt = create_lso(opt, initial_scale=2, dynamic_growth_steps=2)

            def loss():
                rep_id = (
                    tf.distribute.get_replica_context().replica_id_in_sync_group
                )
                # The last element of last replica's gradient is NaN.
                return tf.cond(
                    tf.equal(rep_id, 0),
                    lambda: var * 2.0,
                    lambda: var * tf.constant([1.0, float("NaN")]),
                )

            run_fn = lambda: opt.minimize(loss, var_list=[var])
            if use_tf_function:
                run_fn = tf.function(run_fn)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            # Variable should not change from before, due to NaN gradients.
            self.assertAllClose(self.evaluate(var), [1.0, 2.0])
            # Loss scale should half due to NaN gradients.
            self.assertEqual(1.0, self.evaluate(opt.loss_scale))

    def testCustomAggregater(self):
        def gradient_aggregator(grads_and_vars):
            # Simulate an all-reduce where a replica has a NaN gradient by
            # setting the last gradient to NaN
            grads_and_vars = list(grads_and_vars)
            last_grad, last_var = grads_and_vars[-1]
            grads_and_vars[-1] = (last_grad * float("NaN"), last_var)
            return grads_and_vars

        var = tf.Variable([1.0, 2.0])
        opt = gradient_descent.SGD(1.0, gradient_aggregator=gradient_aggregator)
        opt = loss_scale_optimizer.LossScaleOptimizer(
            opt, initial_scale=2, dynamic_growth_steps=2
        )

        loss = lambda: var * 2
        run_op = opt.minimize(loss, var_list=[var])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self._run_if_in_graph_mode(run_op)
        # Variable should not change from before, due to NaN gradients.
        self.assertAllClose(self.evaluate(var), [1.0, 2.0])
        # Loss scale should half due to NaN gradients.
        self.assertEqual(1.0, self.evaluate(opt.loss_scale))

    @test_combinations.generate(opt_and_strategy_and_mode_combinations())
    def testDynamicLossScaleWithSlots(
        self, opt_cls, strategy_fn, use_tf_function
    ):
        strategy_obj = strategy_fn()
        if (
            isinstance(strategy_obj, tf.distribute.MirroredStrategy)
            and tf.compat.v1.control_flow_v2_enabled()
            and not tf.executing_eagerly()
        ):
            self.skipTest("b/138667997")
        with strategy_obj.scope() as strategy:
            var = tf.Variable([1.0, 2.0])
            # An SGD optimizer with momentum has slot variables.
            opt = create_sgd(opt_cls, 1.0, momentum=1.0)
            initial_scale = 2.0
            opt = create_lso(
                opt, initial_scale=initial_scale, dynamic_growth_steps=1
            )
            loss = lambda: var / strategy.num_replicas_in_sync
            run_fn = lambda: opt.minimize(loss, var_list=[var])
            if use_tf_function:
                run_fn = tf.function(run_fn)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            # The momentum accumulator starts at 0 and the gradient is 1. The
            # accumulator is incremented by the gradient, so it is now 1. Then
            # the variable is subtracted by the accumulator, so the variable is
            # subtracted by 1.
            self.assertAllClose([0.0, 1.0], self.evaluate(var))
            self.assertEqual(self.evaluate(opt.loss_scale), initial_scale * 2)

            run_op = strategy.experimental_run(run_fn)
            self._run_if_in_graph_mode(run_op)
            # The momentum accumulator was 1 before this step and the gradient
            # is 1. The accumulator is incremented by the gradient, so it is
            # now 2. Then the variable is subtracted by the accumulator, so the
            # variable is subtracted by 2.
            self.assertAllClose([-2.0, -1.0], self.evaluate(var))
            self.assertEqual(self.evaluate(opt.loss_scale), initial_scale * 4)

            if isinstance(opt, loss_scale_optimizer.LossScaleOptimizer):
                self.assertEqual(opt.get_slot_names(), ["momentum"])

    def testIterations(self):
        opt = gradient_descent.SGD(2.0)
        lso = loss_scale_optimizer.LossScaleOptimizer(
            opt, dynamic=False, initial_scale=10.0
        )
        lso.iterations = 7
        self.assertEqual(lso.iterations, 7)
        self.assertEqual(opt.iterations, 7)

    @test_combinations.generate(opt_and_strategy_and_mode_combinations())
    def testIterationsIncremented(self, opt_cls, strategy_fn, use_tf_function):
        with strategy_fn().scope() as strategy:
            # Test iterations is incremented in opt.minimize.
            opt = create_sgd(opt_cls, 1.0)
            opt = create_lso(opt)
            var = tf.Variable([5.0])
            loss = lambda: var * 2.0 / strategy.num_replicas_in_sync
            run_fn = lambda: opt.minimize(loss, [var])
            if use_tf_function:
                run_fn = tf.function(run_fn)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            self.assertEqual(
                self.evaluate(var), 3.0
            )  # Grad is 2, so var is 5 - 2
            self.assertEqual(self.evaluate(opt.iterations), 1)

            # Test iterations is incremented in opt.minimize even if gradients
            # aren't applied to variables due to NaN gradients.
            loss = lambda: var * float("NaN")
            run_fn = lambda: opt.minimize(loss, [var])
            if use_tf_function:
                run_fn = tf.function(run_fn)
            run_op = strategy.experimental_run(run_fn)
            self._run_if_in_graph_mode(run_op)
            self.assertEqual(self.evaluate(var), 3.0)
            self.assertEqual(self.evaluate(opt.iterations), 2)

    def testWeightMethods(self):
        with self.test_session():
            var = tf.Variable([1.0])
            opt = gradient_descent.SGD(1.0)
            opt = loss_scale_optimizer.LossScaleOptimizer(
                opt, initial_scale=2.0, dynamic_growth_steps=1
            )
            run_op = opt.minimize(lambda: var * 2, [var])
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)

            self.assertLen(opt.weights, 1)  # The 'iterations' weight
            self.assertEqual(self.evaluate(opt.weights[0]), 1)
            self.assertEqual(opt.get_weights()[0], 1)
            self.assertEqual(self.evaluate(opt.variables()[0]), 1)
            opt.set_weights([np.array(2.0)])
            self.assertEqual(self.evaluate(opt.variables()[0]), 2)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def testHyperParametersExposedLSOV3(self):
        opt = adam_experimental.Adam(learning_rate=1.0, beta_1=0.5, beta_2=0.9)
        lso = loss_scale_optimizer.BaseLossScaleOptimizer(opt)
        lso.learning_rate = tf.Variable(0.005)
        self.assertAllClose(self.evaluate(lso.learning_rate), 0.005)
        self.assertIs(lso.learning_rate, opt.learning_rate)

        lso.use_ema = True
        self.assertEqual(lso.use_ema, True)
        self.assertEqual(opt.use_ema, True)

        lso.ema_momentum = 0.88
        self.assertEqual(lso.ema_momentum, 0.88)
        self.assertEqual(opt.ema_momentum, 0.88)

    def testHyperParametersExposed(self):
        with self.cached_session():
            opt = adam.Adam(learning_rate=1.0, beta_1=0.5, beta_2=0.9)
            lso = loss_scale_optimizer.LossScaleOptimizer(opt)
            # Force hyperparameters to be created
            opt.lr
            self.evaluate(tf.compat.v1.global_variables_initializer())

            self.assertEqual(self.evaluate(lso.beta_1), 0.5)
            self.assertIsInstance(lso.beta_1, tf.Variable)
            self.assertEqual(self.evaluate(lso.lr), 1.0)
            self.assertIs(lso.lr, opt.lr)
            self.assertIs(lso.lr, lso.learning_rate)

            lso.beta_1 = 0.25
            self.assertEqual(self.evaluate(lso.beta_1), 0.25)
            self.assertEqual(self.evaluate(opt.beta_1), 0.25)
            self.assertIs(lso.beta_1, opt.beta_1)
            opt.beta_1 = 0.75
            self.assertEqual(self.evaluate(lso.beta_1), 0.75)
            self.assertEqual(self.evaluate(opt.beta_1), 0.75)
            self.assertIs(lso.beta_1, opt.beta_1)
            lso.lr = 2.0
            self.assertEqual(self.evaluate(lso.lr), 2.0)
            self.assertEqual(self.evaluate(lso.learning_rate), 2.0)
            self.assertEqual(self.evaluate(opt.lr), 2.0)
            self.assertEqual(self.evaluate(opt.learning_rate), 2.0)
            self.assertIs(lso.lr, opt.lr)

            # Test setting attribute that is both attribute on
            # LossScaleOptimizer and hyperparameter on wrapped optimizer.
            class MyOpt(gradient_descent.SGD):
                def __init__(self):
                    super().__init__()
                    self._set_hyper("loss_scale", 123.0)

            opt = MyOpt()
            lso = loss_scale_optimizer.LossScaleOptimizer(opt)
            with self.assertRaises(AttributeError):
                lso.loss_scale = 2.0

    @test_combinations.generate(opt_combinations_only())
    def testArbitraryAttributesNotExposed(self, opt_cls):
        opt = create_sgd(opt_cls)
        lso = create_lso(opt)
        self.assertFalse(opt.nesterov)
        with self.assertRaisesRegex(
            AttributeError,
            "'LossScaleOptimizer(V3)?' object has no attribute 'nesterov'",
        ):
            lso.nesterov

        lso.nesterov = True
        self.assertTrue(lso.nesterov)
        self.assertFalse(opt.nesterov)

    def testDir(self):
        lso = loss_scale_optimizer.LossScaleOptimizer(gradient_descent.SGD())
        dir_result = dir(lso)
        self.assertIn("learning_rate", dir_result)  # Hyperparameter
        self.assertIn("lr", dir_result)  # Hyperparameter
        self.assertIn("minimize", dir_result)  # Attribute
        self.assertIn("loss_scale", dir_result)  # Attribute
        self.assertNotIn("nesterov", dir_result)  # Attribute on inner optimizer
        self.assertIn("nesterov", dir(lso.inner_optimizer))

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def testApplyGradientsGetsUnwrappedTensors(self):
        # Tests that gradients passed to apply_gradients are not wrapped in a
        # DistributionStrategy wrapper, such as PerReplica, but instead are raw
        # Tensors. Optimizer subclasses that override apply_gradients() expect
        # raw Tensors, even though the base Optimizer can handle PerReplica
        # gradients.

        outer_self = self

        class MyOptimizer(gradient_descent.SGD):
            def apply_gradients(
                self,
                grads_and_vars,
                name=None,
                experimental_aggregate_gradients=True,
            ):
                for grad, _ in grads_and_vars:
                    outer_self.assertIsInstance(grad, tf.Tensor)
                return super().apply_gradients(
                    grads_and_vars, name, experimental_aggregate_gradients
                )

        with create_mirrored_strategy().scope() as strategy:
            var = tf.Variable([5.0])
            opt = MyOptimizer(learning_rate=1.0)
            opt = loss_scale_optimizer.LossScaleOptimizer(
                opt, dynamic=False, initial_scale=1
            )
            loss = lambda: var * 2.0
            run_fn = lambda: opt.minimize(loss, [var])
            strategy.experimental_run(run_fn)

    @test_combinations.generate(
        test_combinations.combine(mode="eager", use_tf_function=[False, True])
    )
    def testApplyGradientsGetsUnwrappedTensorsWithNewOptimizer(
        self, use_tf_function
    ):
        outer_self = self

        class MyOptimizer(sgd_experimental.SGD):
            def apply_gradients(
                self,
                grads_and_vars,
                skip_gradients_aggregation=False,
                experimental_aggregate_gradients=True,
            ):
                for grad, _ in grads_and_vars:
                    outer_self.assertIsInstance(grad, tf.Tensor)
                return super().apply_gradients(
                    grads_and_vars,
                    skip_gradients_aggregation=skip_gradients_aggregation,
                )

        with create_mirrored_strategy().scope() as strategy:
            var = tf.Variable([5.0])
            opt = MyOptimizer(learning_rate=1.0)
            opt = loss_scale_optimizer.LossScaleOptimizerV3(
                opt, dynamic=False, initial_scale=1
            )
            loss = lambda: var * 2.0
            run_fn = lambda: opt.minimize(loss, [var])
            if use_tf_function:
                run_fn = tf.function(run_fn)
            strategy.experimental_run(run_fn)

    @test_combinations.generate(opt_combinations_only())
    def testLossScaleDelegationWithWrapper(self, opt_cls):
        # Test learning_rate is exposed when LossScaleOptimizer wraps another
        # wrapper.

        class MyOptimizer(opt_cls):
            def __init__(self):
                super().__init__("MyOptimizer")
                self.inner_optimizer = create_sgd(opt_cls, learning_rate=1.0)

            @property
            def learning_rate(self):
                return self.inner_optimizer.learning_rate

            @learning_rate.setter
            def learning_rate(self, value):
                self.inner_optimizer.learning_rate = value

            def get_config(self):
                return {}

        with self.cached_session():
            opt = MyOptimizer()
            opt = create_lso(opt)

            # Force hyperparameters to be created
            opt.learning_rate
            self.evaluate(tf.compat.v1.global_variables_initializer())

            self.assertEqual(self.evaluate(opt.learning_rate), 1.0)
            self.assertEqual(
                self.evaluate(
                    opt.inner_optimizer.inner_optimizer.learning_rate
                ),
                1.0,
            )
            opt.learning_rate = 2.0
            self.assertEqual(self.evaluate(opt.learning_rate), 2.0)
            self.assertEqual(
                self.evaluate(
                    opt.inner_optimizer.inner_optimizer.learning_rate
                ),
                2.0,
            )

    @test_combinations.generate(
        test_combinations.combine(
            opt_cls=optimizer_v2.OptimizerV2,
            strategy_fn=STRATEGY_FNS,
            mode=["graph", "eager"],
            use_tf_function=False,
            save_with_ls=[False, True],
            restore_with_ls=[False, True],
        )
        + test_combinations.combine(
            opt_cls=optimizer_experimental.Optimizer,
            strategy_fn=STRATEGY_FNS,
            mode="eager",
            use_tf_function=[False, True],
            save_with_ls=[False, True],
            restore_with_ls=[False, True],
        )
    )
    def testCheckpoint(
        self,
        opt_cls,
        strategy_fn,
        use_tf_function,
        save_with_ls,
        restore_with_ls,
    ):

        if not save_with_ls and not restore_with_ls:
            self.skipTest(
                "Skipping because save_with_ls=False and "
                "restore_with_ls=False, which means loss scaling is not "
                "used"
            )

        sgd_cls = type(create_sgd(opt_cls))

        class MySGD(sgd_cls):
            """A custom optimizer that tracks an extra variable."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.my_var = tf.Variable(0.0)
                self._track_trackable(self.my_var, "my_var")

        strategy = strategy_fn()
        replicas = strategy.num_replicas_in_sync
        if (
            isinstance(strategy, tf.distribute.MirroredStrategy)
            and not tf.executing_eagerly()
        ):
            # TODO(b/121381184): Enable running the test in this case.
            return

        with self.test_session(), strategy.scope():
            # Build and run a simple model.
            var = tf.Variable([2.0])
            opt = inner_opt = MySGD(1.0, momentum=1.0)
            if save_with_ls:
                opt = create_lso(
                    opt, initial_scale=1.0, dynamic_growth_steps=2.0
                )
            run_fn = lambda: opt.minimize(
                lambda: var / replicas + 1.0, var_list=[var]
            )
            if use_tf_function:
                run_fn = tf.function(run_fn)
            opt_op = strategy.experimental_run(run_fn)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self.evaluate(strategy.experimental_local_results(opt_op))

            # Assert values.
            self.assertEqual(self.evaluate(var), 1.0)
            if save_with_ls:
                self.assertEqual(self.evaluate(opt.loss_scale), 1.0)
                self.assertEqual(self.evaluate(opt.dynamic_counter), 1)
            if opt_cls == optimizer_v2.OptimizerV2:
                slot_var = opt.get_slot(var, "momentum")
                self.assertEqual(self.evaluate(slot_var).item(), -1)
            self.assertEqual(self.evaluate(opt.iterations), 1)

            # Set optimizer variable to check arbitrary optimizer attributes can
            # be saved/restored
            self.evaluate(inner_opt.my_var.assign(1.0))

            # Save a checkpoint.
            checkpoint = tf.train.Checkpoint(optimizer=opt, var=var)
            prefix = os.path.join(self.get_temp_dir(), "ckpt")
            save_path = checkpoint.save(prefix)

            # Create new model
            var = tf.Variable([2.0])
            opt = inner_opt = MySGD(1.0, momentum=1.0)
            if restore_with_ls:
                opt = create_lso(
                    opt, initial_scale=1.0, dynamic_growth_steps=2.0
                )

            # Restore new model.
            checkpoint = tf.train.Checkpoint(optimizer=opt, var=var)
            status = checkpoint.restore(save_path)
            if save_with_ls:
                status.assert_existing_objects_matched()
            else:
                status.assert_nontrivial_match()

            # Assert restored values. We can only assert in eager mode since the
            # variables are uninitialized in graph mode
            if tf.executing_eagerly():
                self.assertEqual(self.evaluate(var), 1.0)
                if save_with_ls and restore_with_ls:
                    self.assertEqual(self.evaluate(opt.loss_scale), 1.0)
                    self.assertEqual(self.evaluate(opt.dynamic_counter), 1)
                elif restore_with_ls:
                    self.assertEqual(self.evaluate(opt.loss_scale), 1.0)
                    self.assertEqual(self.evaluate(opt.dynamic_counter), 0)
                self.assertEqual(self.evaluate(opt.iterations), 1)

            # Run the model again.
            run_fn = lambda: opt.minimize(
                lambda: var / replicas + 1.0, var_list=[var]
            )
            if use_tf_function:
                run_fn = tf.function(run_fn)
            opt_op = strategy.experimental_run(run_fn)

            # Assert new values.
            self.evaluate(tf.compat.v1.global_variables_initializer())
            status.run_restore_ops()
            self.evaluate(strategy.experimental_local_results(opt_op))
            self.assertEqual(self.evaluate(var), -1)
            if opt_cls == optimizer_v2.OptimizerV2:
                slot_var = opt.get_slot(var, "momentum")
                self.assertEqual(self.evaluate(slot_var).item(), -2)
            self.assertEqual(self.evaluate(opt.iterations), 2)
            self.assertEqual(self.evaluate(inner_opt.my_var), 1)

            # Restore model again to test restoring after slots are created
            status = checkpoint.restore(save_path)
            if save_with_ls and restore_with_ls:
                status.assert_consumed()
            elif save_with_ls:
                status.assert_existing_objects_matched()
            elif restore_with_ls:
                status.assert_nontrivial_match()
            status.run_restore_ops()
            self.assertEqual(self.evaluate(var), 1)
            if opt_cls == optimizer_v2.OptimizerV2:
                self.assertEqual(self.evaluate(slot_var).item(), -1)

    @test_combinations.generate(
        test_combinations.combine(config_version=["v2", "tf2_3"])
        + test_combinations.combine(config_version="v3", mode="eager")
    )
    def testGetConfigFixed(self, config_version):
        # Get a config from LossScaleOptimizer, LossScaleOptimizerV3, or the
        # LossScaleOptimizer from TF 2.3. Then restore the config into a
        # LossScaleOptimizer or LossScaleOptimizerV3
        if config_version == "v2":
            opt = gradient_descent.SGD(2.0, momentum=0.5)
            opt = loss_scale_optimizer.LossScaleOptimizer(
                opt, dynamic=False, initial_scale=2
            )
            config = opt.get_config()
            opt = loss_scale_optimizer.LossScaleOptimizer.from_config(config)
        elif config_version == "v3":
            opt = sgd_experimental.SGD(2.0, momentum=0.5)
            opt = loss_scale_optimizer.LossScaleOptimizerV3(
                opt, dynamic=False, initial_scale=2
            )
            config = opt.get_config()
            opt = loss_scale_optimizer.LossScaleOptimizerV3.from_config(config)
        else:
            self.assertEqual(config_version, "tf2_3")
            config = {
                "optimizer": {
                    "class_name": "SGD",
                    "config": {
                        "learning_rate": 2.0,
                        "momentum": 0.5,
                        "decay": 0.0,
                        "nesterov": False,
                        "name": "SGD",
                    },
                },
                "loss_scale": {
                    "class_name": "FixedLossScale",
                    "config": {"loss_scale_value": 2.0},
                },
            }
            opt = loss_scale_optimizer.LossScaleOptimizer.from_config(config)

        # Force hyperparameters to be created
        opt.learning_rate
        self.evaluate(tf.compat.v1.global_variables_initializer())

        # Test attributes on the optimizer
        self.assertEqual(self.evaluate(opt.learning_rate), 2.0)
        self.assertEqual(self.evaluate(opt.inner_optimizer.learning_rate), 2.0)
        self.assertEqual(
            self._eval_if_tensor(opt.inner_optimizer.momentum), 0.5
        )
        self.assertEqual(self.evaluate(opt.loss_scale), 2.0)
        self.assertEqual(opt.initial_scale, 2.0)
        self.assertIsNone(opt.dynamic_growth_steps)
        self.assertIsNone(opt.dynamic_counter)
        self.assertFalse(opt.dynamic)

        # Ensure the optimizer can be used
        var = tf.Variable([5.0])
        run_op = self._run_fn_with_grad_check(
            tf.distribute.get_strategy(), var, opt, 2
        )()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self._run_if_in_graph_mode(run_op)
        self.assertEqual(self.evaluate(var), [3.0])

    @test_combinations.generate(
        test_combinations.combine(config_version=["v2", "tf2_3"])
        + test_combinations.combine(config_version="v3", mode="eager")
    )
    def testGetConfigDynamic(self, config_version):
        # Get a config from LossScaleOptimizer, LossScaleOptimizerV3, or the
        # LossScaleOptimizer from TF 2.3. Then restore the config into a
        # LossScaleOptimizer or LossScaleOptimizerV3
        if config_version == "v2":
            opt = gradient_descent.SGD(2.0, momentum=0.5)
            opt = loss_scale_optimizer.LossScaleOptimizer(
                opt, initial_scale=2, dynamic_growth_steps=3
            )
            config = opt.get_config()
            opt = loss_scale_optimizer.LossScaleOptimizer.from_config(config)
        elif config_version == "v3":
            opt = sgd_experimental.SGD(2.0, momentum=0.5)
            opt = loss_scale_optimizer.LossScaleOptimizerV3(
                opt, initial_scale=2, dynamic_growth_steps=3
            )
            config = opt.get_config()
            opt = loss_scale_optimizer.LossScaleOptimizerV3.from_config(config)
        else:
            self.assertEqual(config_version, "tf2_3")
            config = {
                "optimizer": {
                    "class_name": "SGD",
                    "config": {
                        "learning_rate": 2.0,
                        "momentum": 0.5,
                        "decay": 0.0,
                        "nesterov": False,
                        "name": "SGD",
                    },
                },
                "loss_scale": {
                    "class_name": "DynamicLossScale",
                    "config": {
                        "initial_loss_scale": 2.0,
                        "increment_period": 3,
                        "multiplier": 2.0,
                    },
                },
            }
            opt = loss_scale_optimizer.LossScaleOptimizer.from_config(config)

        # Force hyperparameters to be created
        opt.learning_rate
        self.evaluate(tf.compat.v1.global_variables_initializer())

        # Test attributes on the optimizer
        self.assertEqual(self.evaluate(opt.learning_rate), 2.0)
        self.assertEqual(self.evaluate(opt.inner_optimizer.learning_rate), 2.0)
        self.assertEqual(
            self._eval_if_tensor(opt.inner_optimizer.momentum), 0.5
        )
        self.assertEqual(self.evaluate(opt.loss_scale), 2.0)
        self.assertEqual(opt.initial_scale, 2.0)
        self.assertEqual(opt.dynamic_growth_steps, 3.0)
        self.assertTrue(opt.dynamic)

        # Ensure the optimizer can be used
        var = tf.Variable([5.0])
        run_op = self._run_fn_with_grad_check(
            tf.distribute.get_strategy(), var, opt, 2
        )()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self._run_if_in_graph_mode(run_op)
        self.assertEqual(self.evaluate(var), [3.0])
        self.assertEqual(self.evaluate(opt.dynamic_counter), 1)

    def test_from_config_with_invalid_multiplier(self):
        config = {
            "optimizer": {
                "class_name": "SGD",
                "config": {
                    "learning_rate": 2.0,
                    "momentum": 0.5,
                    "decay": 0.0,
                    "nesterov": False,
                    "name": "SGD",
                },
            },
            "loss_scale": {
                "class_name": "DynamicLossScale",
                "config": {
                    "initial_loss_scale": 2.0,
                    "increment_period": 3,
                    "multiplier": 4.0,
                },
            },
        }

        expected_error = (
            "Cannot deserialize LossScaleOptimizer with a "
            "DynamicLossScale whose multiplier is not 2. Got "
            "DynamicLossScale: DynamicLossScale\\("
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            loss_scale_optimizer.LossScaleOptimizer.from_config(config)

    @test_combinations.generate(
        test_combinations.combine(lso_type=["v1", "v2"])
        + test_combinations.combine(lso_type="v3", mode="eager")
    )
    def testSerializationWithBuiltInOptimizer(self, lso_type):
        if lso_type in ("v1", "v2"):
            opt = gradient_descent.SGD(2.0, momentum=0.5)
            opt = loss_scale_optimizer.LossScaleOptimizer(
                opt, initial_scale=2.0, dynamic_growth_steps=3.0
            )
            config = optimizers.serialize(opt)
            if lso_type == "v1":
                # LossScaleOptimizerV1 was an older experimental version of LSO
                # that is now deleted. The config had the same format as LSO but
                # the class name was different. This tests that LSO V1 configs
                # can still be deserialized, which are deserialized as a
                # (non-V1) LSO
                config["class_name"] = "LossScaleOptimizerV1"
        else:
            opt = sgd_experimental.SGD(2.0, momentum=0.5)
            opt = loss_scale_optimizer.LossScaleOptimizerV3(
                opt, initial_scale=2.0, dynamic_growth_steps=3
            )
            config = optimizers.serialize(opt)
        opt = optimizers.deserialize(config)
        # Force hyperparameters to be created
        opt.learning_rate
        self.evaluate(tf.compat.v1.global_variables_initializer())

        self.assertEqual(self.evaluate(opt.learning_rate), 2.0)
        self.assertEqual(
            self._eval_if_tensor(opt.inner_optimizer.momentum), 0.5
        )
        self.assertEqual(self.evaluate(opt.loss_scale), 2.0)
        self.assertEqual(opt.dynamic_growth_steps, 3.0)
        self.assertTrue(opt.dynamic)
        if lso_type in ("v1", "v2"):
            self.assertEqual(type(opt), loss_scale_optimizer.LossScaleOptimizer)
        else:
            self.assertEqual(
                type(opt), loss_scale_optimizer.LossScaleOptimizerV3
            )

        # Ensure the optimizer can be used
        var = tf.Variable([5.0])
        run_op = self._run_fn_with_grad_check(
            tf.distribute.get_strategy(), var, opt, 2
        )()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self._run_if_in_graph_mode(run_op)
        self.assertEqual(self.evaluate(var), [3.0])
        self.assertEqual(self.evaluate(opt.dynamic_counter), 1)

    @test_combinations.generate(opt_combinations_only())
    def testSerializationWithCustomOptimizer(self, opt_cls):
        sgd_cls = type(create_sgd(opt_cls))

        class MySGD(sgd_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.my_attribute = 123

        opt = MySGD(2.0, momentum=0.5)
        opt = create_lso(opt, initial_scale=2.0, dynamic_growth_steps=3.0)
        config = optimizers.serialize(opt)
        custom_objects = {"MySGD": MySGD}
        opt = optimizers.deserialize(config, custom_objects=custom_objects)
        # Force hyperparameters to be created
        opt.learning_rate
        self.evaluate(tf.compat.v1.global_variables_initializer())

        self.assertEqual(self.evaluate(opt.learning_rate), 2.0)
        self.assertEqual(
            self._eval_if_tensor(opt.inner_optimizer.momentum), 0.5
        )
        self.assertEqual(self.evaluate(opt.loss_scale), 2.0)
        self.assertEqual(opt.dynamic_growth_steps, 3.0)
        self.assertEqual(opt.inner_optimizer.my_attribute, 123)

    @test_utils.run_v2_only
    def testConvertToLegacyOptimizer(self):
        opt = sgd_experimental.SGD(1.0)
        opt = loss_scale_optimizer.BaseLossScaleOptimizer(opt)
        converted_opt = optimizers.convert_to_legacy_optimizer(opt)
        self.assertEqual(
            type(converted_opt), loss_scale_optimizer.LossScaleOptimizer
        )

        reference_opt = gradient_descent.SGD(1.0)
        reference_opt = loss_scale_optimizer.BaseLossScaleOptimizer(
            reference_opt
        )
        self.assertEqual(converted_opt.get_config(), reference_opt.get_config())

        # Test with a custom learning rate schedule
        class CustomLRSchedule(learning_rate_schedule.LearningRateSchedule):
            def __init__(self, initial_learning_rate):
                self.initial_learning_rate = initial_learning_rate

            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                return self.initial_learning_rate / (step + 1)

            def get_config(self):
                return {"initial_learning_rate": self.initial_learning_rate}

        opt = sgd_experimental.SGD(CustomLRSchedule(1.0))
        opt = loss_scale_optimizer.BaseLossScaleOptimizer(opt)
        converted_opt = optimizers.convert_to_legacy_optimizer(opt)
        self.assertEqual(
            type(converted_opt), loss_scale_optimizer.LossScaleOptimizer
        )

        reference_opt = gradient_descent.SGD(CustomLRSchedule(1.0))
        reference_opt = loss_scale_optimizer.BaseLossScaleOptimizer(
            reference_opt
        )
        self.assertEqual(converted_opt.get_config(), reference_opt.get_config())

    @test_combinations.generate(opt_combinations_only())
    def testUnsupportedStrategy(self, opt_cls):
        strategy = tf.distribute.experimental.CentralStorageStrategy()
        expected_error = (
            "Loss scaling is not supported with the tf.distribute.Strategy: "
            "CentralStorageStrategy. Try using a different Strategy, e.g. a "
            "MirroredStrategy"
        )
        with strategy.scope(), self.assertRaisesRegex(
            ValueError, expected_error
        ):
            create_lso(create_sgd(opt_cls))
        opt = create_lso(create_sgd(opt_cls))
        with strategy.scope():
            var = tf.Variable(1.0)
            loss = lambda: var * 2.0
            run_fn = lambda: opt.minimize(loss, [var])
            with self.assertRaisesRegex(ValueError, expected_error):
                strategy.experimental_run(run_fn)

    @test_combinations.generate(opt_combinations_only())
    def testInvalidArgsWithFixedLossScale(self, opt_cls):
        opt = create_sgd(opt_cls)
        with self.assertRaisesRegex(
            ValueError,
            '"initial_scale" must be specified if "dynamic" is False',
        ):
            create_lso(opt, dynamic=False)
        opt = create_sgd(opt_cls)
        with self.assertRaisesRegex(
            ValueError,
            '"dynamic_growth_steps" must be None if "dynamic" is '
            "False, but got: 2",
        ):
            create_lso(
                opt, dynamic=False, initial_scale=1, dynamic_growth_steps=2
            )

    @test_combinations.generate(opt_combinations_only())
    def testDynamicMustBeBool(self, opt_cls):
        opt = create_sgd(opt_cls)
        with self.assertRaisesRegex(
            TypeError,
            '"dynamic" argument to LossScaleOptimizer.__init__ must be '
            "a bool, but got: 'dynamic'",
        ):
            create_lso(opt, "dynamic")

    @test_combinations.generate(opt_combinations_only())
    def testScalingWarning(self, opt_cls):
        var = tf.Variable(1.0)
        lso = create_lso(create_sgd(opt_cls))
        with mock.patch.object(tf_logging, "warning") as mock_warn:
            lso.apply_gradients([(tf.constant(1.0), var)])
            self.assertIn(
                "You forgot to call LossScaleOptimizer.get_scaled_loss() and "
                "LossScaleOptimizer.get_unscaled_gradients() before",
                mock_warn.call_args_list[0][0][0],
            )
        lso = create_lso(create_sgd(opt_cls))
        with mock.patch.object(tf_logging, "warning") as mock_warn:
            lso.get_scaled_loss(tf.constant(1.0))
            lso.apply_gradients([(tf.constant(1.0), var)])
            self.assertIn(
                "You forgot to call "
                "LossScaleOptimizer.get_unscaled_gradients() before",
                mock_warn.call_args_list[0][0][0],
            )
        lso = create_lso(create_sgd(opt_cls))
        with mock.patch.object(tf_logging, "warning") as mock_warn:
            lso.get_unscaled_gradients([tf.constant(1.0)])
            lso.apply_gradients([(tf.constant(1.0), var)])
            self.assertIn(
                "You forgot to call LossScaleOptimizer.get_scaled_loss() "
                "before",
                mock_warn.call_args_list[0][0][0],
            )

    @test_combinations.generate(opt_combinations_only())
    def testScalingNoWarning(self, opt_cls):
        var = tf.Variable(1.0)
        lso = create_lso(create_sgd(opt_cls))
        with mock.patch.object(tf_logging, "warning") as mock_warn:
            lso.get_scaled_loss(tf.constant(1.0))
            lso.get_unscaled_gradients([tf.constant(1.0)])
            lso.apply_gradients([(tf.constant(1.0), var)])
            mock_warn.assert_not_called()

    @test_combinations.generate(opt_combinations_only())
    def testErrorWhenNesting(self, opt_cls):
        opt = create_sgd(opt_cls)
        opt = create_lso(opt)
        with self.assertRaisesRegex(
            TypeError,
            "LossScaleOptimizer cannot wrap another LossScaleOptimizer",
        ):
            create_lso(opt)

    @test_combinations.generate(opt_combinations_only())
    def testErrorWrappingSameOptimizerMultipleTimes(self, opt_cls):
        inner_opt = create_sgd(opt_cls)
        create_lso(inner_opt)
        with self.assertRaisesRegex(
            ValueError,
            '"inner_optimizer" is already wrapped by a LossScaleOptimizer.',
        ):
            create_lso(inner_opt)

    def testErrorWhenWrappingNonOptimizer(self):
        with self.assertRaisesRegex(
            TypeError,
            '"inner_optimizer" must be an instance of '
            "`tf.keras.optimizers.Optimizer` or "
            "`tf.keras.optimizers.experimental.Optimizer`, but got: 1",
        ):
            loss_scale_optimizer.BaseLossScaleOptimizer(1)

    def testErrorWhenV3LsoWrapsV2Optimizer(self):
        sgd = gradient_descent.SGD()
        with self.assertRaisesRegex(
            TypeError,
            "only the new experimental optimizer "
            "defined in keras/optimizer_expeirmental/optimizer.py can be "
            "passed",
        ):
            loss_scale_optimizer.LossScaleOptimizerV3(sgd)

    def testErrorWhenV2LsoWrapsV3Optimizer(self):
        sgd = sgd_experimental.SGD()
        with self.assertRaisesRegex(
            TypeError,
            "only the classic optimizers subclassing from "
            "`tf.keras.optimizers.Optimizer` can be passed",
        ):
            loss_scale_optimizer.LossScaleOptimizer(sgd)


if __name__ == "__main__":
    tf.test.main()
