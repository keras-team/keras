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
"""Tests keras.Model works properly with mixed precision."""

import os

import numpy as np
import tensorflow.compat.v2 as tf
from absl import flags
from absl.testing import parameterized

from keras import backend
from keras import layers
from keras import models
from keras.applications import densenet
from keras.applications import efficientnet
from keras.applications import inception_resnet_v2
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.applications import nasnet
from keras.applications import resnet
from keras.applications import vgg16
from keras.applications import xception
from keras.engine import base_layer_utils
from keras.engine import input_spec
from keras.engine import sequential
from keras.layers import core
from keras.mixed_precision import loss_scale_optimizer
from keras.mixed_precision import policy
from keras.mixed_precision import test_util as mp_test_util
from keras.optimizers import optimizer_v1
from keras.optimizers import sgd
from keras.optimizers.legacy import gradient_descent
from keras.saving import object_registration
from keras.saving.legacy import save
from keras.saving.serialization_lib import SafeModeScope
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = tf.distribute.get_strategy


def create_mirrored_strategy():
    """Create a MirroredStrategy, using a GPU if it is available."""
    if tf.config.list_logical_devices("GPU"):
        return tf.distribute.MirroredStrategy(["cpu:0", "gpu:0"])
    else:
        return tf.distribute.MirroredStrategy(["cpu:0"])


TESTCASES = (
    {"testcase_name": "base", "strategy_fn": default_strategy_fn},
    {"testcase_name": "distribute", "strategy_fn": create_mirrored_strategy},
)


class KerasModelTest(test_combinations.TestCase):
    """Test mixed precision with Keras models."""

    def _skip_if_strategy_unsupported(self, strategy_fn):
        if (
            strategy_fn != default_strategy_fn
            and test_utils.get_model_type() == "subclass"
        ):
            self.skipTest(
                "Non-default strategies are unsupported with subclassed models"
            )

    def _skip_if_save_format_unsupported(self, save_format):
        model_type = test_utils.get_model_type()
        if save_format == "h5" and model_type == "subclass":
            self.skipTest(
                "Saving subclassed models with the HDF5 format is unsupported"
            )
        if (
            save_format == "tf"
            and model_type == "subclass"
            and not tf.executing_eagerly()
        ):
            self.skipTest(
                "b/148820505: This combination of features is currently broken."
            )

    @test_combinations.run_with_all_model_types
    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        {"testcase_name": "base", "strategy_fn": default_strategy_fn},
        {
            "testcase_name": "distribute",
            "strategy_fn": create_mirrored_strategy,
        },
        {
            "testcase_name": "operator",
            "strategy_fn": create_mirrored_strategy,
            "use_operator": True,
        },
        {
            "testcase_name": "regularizer",
            "strategy_fn": create_mirrored_strategy,
            "use_regularizer": True,
        },
        {
            "testcase_name": "get_config",
            "strategy_fn": create_mirrored_strategy,
            "get_config": True,
            "use_regularizer": True,
        },
        {
            "testcase_name": "saved_model",
            "strategy_fn": default_strategy_fn,
            "save_format": "tf",
            "use_regularizer": True,
        },
        {
            "testcase_name": "saved_model_input_spec",
            "strategy_fn": default_strategy_fn,
            "save_format": "tf",
            "use_regularizer": True,
            "use_input_spec": True,
        },
        {
            "testcase_name": "h5",
            "strategy_fn": default_strategy_fn,
            "save_format": "h5",
            "use_regularizer": True,
        },
        {
            "testcase_name": "saved_model_distribute",
            "strategy_fn": create_mirrored_strategy,
            "save_format": "tf",
            "use_regularizer": True,
        },
        {
            "testcase_name": "saved_model_legacy_distribute",
            "strategy_fn": create_mirrored_strategy,
            "save_format": "tf",
            "use_regularizer": True,
            "use_legacy_optimizer": True,
        },
        {
            "testcase_name": "saved_model_input_spec_distribute",
            "strategy_fn": create_mirrored_strategy,
            "save_format": "tf",
            "use_regularizer": True,
            "use_input_spec": True,
        },
        {
            "testcase_name": "h5_distribute",
            "strategy_fn": create_mirrored_strategy,
            "save_format": "h5",
            "use_regularizer": True,
        },
        {
            "testcase_name": "h5_legacy_distribute",
            "strategy_fn": create_mirrored_strategy,
            "save_format": "h5",
            "use_regularizer": True,
            "use_legacy_optimizer": True,
        },
    )
    def test_model(
        self,
        strategy_fn,
        use_operator=False,
        use_regularizer=False,
        policy_name="mixed_float16",
        get_config=False,
        save_format=None,
        use_input_spec=False,
        use_legacy_optimizer=False,
    ):
        self._skip_if_strategy_unsupported(strategy_fn)
        self._skip_if_save_format_unsupported(save_format)
        if not tf.__internal__.tf2.enabled():
            # The non-legacy optimizer is only supported in TF2
            use_legacy_optimizer = True
        if use_regularizer:
            weight_regularizer = mp_test_util.IdentityRegularizer()
            activity_regularizer = mp_test_util.ReduceSumRegularizer()
        else:
            weight_regularizer = activity_regularizer = None
        with strategy_fn().scope():
            with policy.policy_scope(policy_name):
                layer = mp_test_util.MultiplyLayer(
                    assert_type=tf.float16,
                    use_operator=use_operator,
                    regularizer=weight_regularizer,
                    activity_regularizer=activity_regularizer,
                    input_shape=(1,),
                )
                if use_input_spec:
                    layer.input_spec = input_spec.InputSpec(shape=(None, 1))
                model = test_utils.get_model_from_layers(
                    [layer], input_shape=(1,), input_dtype=tf.float16
                )
                if get_config:
                    config = model.get_config()
                    model = model.__class__.from_config(
                        config,
                        custom_objects={
                            "MultiplyLayer": mp_test_util.MultiplyLayer
                        },
                    )
                    (layer,) = (
                        layer
                        for layer in model.layers
                        if isinstance(layer, mp_test_util.MultiplyLayer)
                    )

                def loss_fn(y_true, y_pred):
                    del y_true
                    return tf.reduce_mean(y_pred)

                # Learning rate is small enough that if applied to a float16
                # variable, the variable will not change. So this tests the
                # learning rate not applied to a float16 value, but instead the
                # float32 variable.
                learning_rate = 2**-14
                if use_legacy_optimizer:
                    opt = gradient_descent.SGD(learning_rate)
                else:
                    opt = sgd.SGD(learning_rate)
                # Use a fixed loss scale, as this test will fail if gradients
                # are skipped for a step due to dynamic loss scaling.
                opt = loss_scale_optimizer.BaseLossScaleOptimizer(
                    opt, dynamic=False, initial_scale=8
                )
                model.compile(
                    opt,
                    loss=loss_fn,
                    run_eagerly=test_utils.should_run_eagerly(),
                )

        x = np.ones((2, 1))
        y = np.ones((2, 1))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        model.fit(dataset)
        # Variable starts at 1, and should have gradient of 2 ** -14 subtracted
        # from it.
        expected = 1 - 2**-14
        if use_regularizer:
            # Weight and activity regularizer each add another 2 ** -14 to the
            # gradient.
            expected -= 2 * 2**-14
        self.assertEqual(backend.eval(layer.v), expected)

        if save_format:
            with object_registration.CustomObjectScope(
                {
                    "MultiplyLayer": mp_test_util.MultiplyLayer,
                    "loss_fn": loss_fn,
                }
            ):
                self._test_saving(model, dataset, save_format, use_regularizer)

    def _test_saving(self, model, dataset, save_format, use_regularizer):
        # Save and load model, asserting variable does not change
        save_path = os.path.join(self.get_temp_dir(), "model")
        model.save(save_path, save_format=save_format)
        model = save.load_model(save_path)
        (layer,) = (
            layer
            for layer in model.layers
            if "MultiplyLayer" in layer.__class__.__name__
        )
        expected = 1 - 2**-14
        if use_regularizer:
            expected -= 2 * 2**-14
        self.assertEqual(backend.eval(layer.v), expected)

        # Continue training, and assert variable is correct value
        model.fit(dataset)
        new_expected = expected - 2**-14
        if use_regularizer:
            new_expected -= 2 * 2**-14
        self.assertEqual(backend.eval(layer.v), new_expected)

        # Load saved model again, and assert variable is previous value
        model = save.load_model(save_path)
        (layer,) = (
            layer
            for layer in model.layers
            if "MultiplyLayer" in layer.__class__.__name__
        )
        self.assertEqual(backend.eval(layer.v), expected)

        # Ensure various dtype-related aspects of the layer are correct
        self.assertEqual(layer.dtype, "float32")
        self.assertEqual(layer.dtype_policy.name, "mixed_float16")
        self.assertEqual(layer.v.dtype, "float32")
        self.assertEqual(layer(np.ones((2, 1))).dtype, "float16")

        self.assertEqual(type(model.dtype_policy), policy.Policy)
        if tf.__internal__.tf2.enabled():
            self.assertEqual(
                layer.get_config()["dtype"],
                {
                    "module": "keras.mixed_precision",
                    "class_name": "Policy",
                    "config": {"name": "mixed_float16"},
                    "registered_name": None,
                },
            )
        else:
            self.assertEqual(
                layer.get_config()["dtype"],
                {
                    "class_name": "Policy",
                    "config": {"name": "mixed_float16"},
                },
            )

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        {"testcase_name": "base", "strategy_fn": default_strategy_fn},
        {
            "testcase_name": "distribute",
            "strategy_fn": create_mirrored_strategy,
        },
    )
    def test_fixed_loss_scaling(self, strategy_fn):
        # The non-legacy optimizer is only supported in TF2
        use_legacy_optimizer = not tf.__internal__.tf2.enabled()
        # Note: We do not test mixed precision in this method, only loss
        # scaling.
        loss_scale = 8.0
        batch_size = 4
        with strategy_fn().scope():
            x = layers.Input(shape=(1,), batch_size=batch_size)
            layer = mp_test_util.MultiplyLayer()
            y = layer(x)

            # The gradient of 'y' at this point is 1. With loss scaling, the
            # gradient is 'loss_scale'. We divide by the batch size since the
            # loss is averaged across batch elements.
            expected_gradient = loss_scale / batch_size
            identity_with_grad_check_fn = (
                mp_test_util.create_identity_with_grad_check_fn(
                    [expected_gradient]
                )
            )
            y = core.Lambda(identity_with_grad_check_fn)(y)
            model = models.Model(inputs=x, outputs=y)

            def loss_fn(y_true, y_pred):
                del y_true
                return tf.reduce_mean(y_pred)

            if use_legacy_optimizer:
                opt = gradient_descent.SGD(1.0)
            else:
                opt = sgd.SGD(1.0)
            opt = loss_scale_optimizer.BaseLossScaleOptimizer(
                opt, dynamic=False, initial_scale=loss_scale
            )
            model.compile(
                opt, loss=loss_fn, run_eagerly=test_utils.should_run_eagerly()
            )

        self.assertEqual(backend.eval(layer.v), 1)
        x = np.ones((batch_size, 1))
        y = np.ones((batch_size, 1))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        model.fit(dataset)
        # Variable starts at 1, and should have gradient of 1 subtracted from
        # it.
        expected = 0
        self.assertEqual(backend.eval(layer.v), expected)

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        {"testcase_name": "base", "strategy_fn": default_strategy_fn},
        {
            "testcase_name": "distribute",
            "strategy_fn": create_mirrored_strategy,
        },
        {
            "testcase_name": "loss_scaling",
            "strategy_fn": create_mirrored_strategy,
            "use_loss_scaling": True,
        },
    )
    def test_advanced_model(self, strategy_fn, use_loss_scaling=False):
        # The advanced model tests mixed-precision-related features that would
        # occur in a resnet50 model. It tests a model that has:
        #  * Multiple layers, some which use auto-cast variables and some which
        #    do not
        #  * Regularization on some variables and not others.
        #  * A fixed loss scale (if use_loss_scaling is True)

        strategy = strategy_fn()
        if use_loss_scaling:
            loss_scale = 8.0
        learning_rate = 2**-14
        # The non-legacy optimizer is only supported in TF2
        use_legacy_optimizer = not tf.__internal__.tf2.enabled()

        with strategy.scope():
            with policy.policy_scope(policy.Policy("mixed_float16")):
                x = layers.Input(shape=(1,), batch_size=2)
                layer1 = mp_test_util.MultiplyLayer(
                    assert_type=tf.float16,
                    regularizer=mp_test_util.IdentityRegularizer(),
                    use_operator=True,
                )
                layer2 = mp_test_util.MultiplyLayerWithoutAutoCast(
                    assert_type=tf.float16, use_operator=True
                )
                layer3 = mp_test_util.MultiplyLayer(
                    assert_type=tf.float16, use_operator=False
                )
                layer4 = mp_test_util.MultiplyLayerWithoutAutoCast(
                    assert_type=tf.float16,
                    regularizer=mp_test_util.IdentityRegularizer(),
                    use_operator=False,
                )
                y = layer1(x)
                y = layer2(y)
                y = layer3(y)
                y = layer4(y)
                if use_loss_scaling:
                    # The gradient of 'y' at this point is 1. With loss scaling,
                    # the gradient is 'loss_scale'. We divide by the batch size
                    # of 2 since the loss is averaged across batch elements.
                    expected_gradient = loss_scale / 2
                    identity_with_grad_check_fn = (
                        mp_test_util.create_identity_with_grad_check_fn(
                            expected_dtype=tf.float16,
                            expected_gradient=[expected_gradient],
                        )
                    )
                    y = core.Lambda(identity_with_grad_check_fn)(y)
                model = models.Model(inputs=x, outputs=y)

                def loss_fn(y_true, y_pred):
                    del y_true
                    return tf.reduce_mean(y_pred)

                if use_legacy_optimizer:
                    opt = gradient_descent.SGD(learning_rate)
                else:
                    opt = sgd.SGD(learning_rate)
                if use_loss_scaling:
                    opt = loss_scale_optimizer.BaseLossScaleOptimizer(
                        opt, dynamic=False, initial_scale=loss_scale
                    )
                model.compile(
                    opt,
                    loss=loss_fn,
                    run_eagerly=test_utils.should_run_eagerly(),
                )

        x = np.ones((2, 1))
        y = np.ones((2, 1))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
        model.fit(dataset)
        for layer in (layer1, layer2, layer3, layer4):
            if layer.losses:
                # Layer has weight regularizer
                self.assertEqual(backend.eval(layer.v), 1 - 2 * learning_rate)
            else:
                # Layer does not have weight regularizer
                self.assertEqual(backend.eval(layer.v), 1 - learning_rate)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        {"testcase_name": "base", "strategy_fn": default_strategy_fn},
        {
            "testcase_name": "distribute",
            "strategy_fn": create_mirrored_strategy,
        },
        {
            "testcase_name": "get_config",
            "strategy_fn": create_mirrored_strategy,
            "get_config": True,
        },
    )
    def test_dynamic_loss_scaling(self, strategy_fn, get_config=False):
        strategy = strategy_fn()
        initial_loss_scale = 2.0
        batch_size = 4
        expected_gradient = backend.variable(
            [initial_loss_scale / batch_size], dtype=tf.float16
        )
        # If this variable is set to True, the model below will have NaN
        # gradients
        have_nan_gradients = backend.variable(False, dtype=tf.bool)
        with strategy.scope():
            opt = sgd.SGD(1.0)
            opt = loss_scale_optimizer.BaseLossScaleOptimizer(
                opt, initial_scale=initial_loss_scale, dynamic_growth_steps=2
            )
            with policy.policy_scope("mixed_float16"):
                x = layers.Input(
                    shape=(1,), batch_size=batch_size, dtype=tf.float16
                )
                layer = mp_test_util.MultiplyLayer(assert_type=tf.float16)
                y = layer(x)
                identity_with_nan_grads = (
                    mp_test_util.create_identity_with_nan_gradients_fn(
                        have_nan_gradients
                    )
                )
                y = core.Lambda(identity_with_nan_grads)(y)
                identity_with_grad_check_fn = (
                    mp_test_util.create_identity_with_grad_check_fn(
                        expected_dtype=tf.float16,
                        expected_gradient=expected_gradient,
                    )
                )
                y = core.Lambda(identity_with_grad_check_fn)(y)
                model = models.Model(inputs=x, outputs=y)
                if get_config:
                    config = model.get_config()
                    with SafeModeScope(safe_mode=False):
                        model = model.__class__.from_config(
                            config,
                            custom_objects={
                                "MultiplyLayer": mp_test_util.MultiplyLayer
                            },
                        )
                    (layer,) = (
                        layer
                        for layer in model.layers
                        if isinstance(layer, mp_test_util.MultiplyLayer)
                    )

                def loss_fn(y_true, y_pred):
                    del y_true
                    return tf.reduce_mean(y_pred)

                model.compile(
                    opt,
                    loss=loss_fn,
                    run_eagerly=test_utils.should_run_eagerly(),
                )

        self.assertEqual(backend.eval(layer.v), 1)
        x = np.ones((batch_size, 1))
        y = np.ones((batch_size, 1))
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        model.fit(dataset)
        # The variables starts with 1 and has a gradient of 1, so will go down
        # by 1 each step.
        self.assertEqual(backend.eval(layer.v), 0)

        model.fit(dataset)
        self.assertEqual(backend.eval(layer.v), -1)

        # There have been two steps without NaNs, so the loss scale will double
        backend.set_value(
            expected_gradient, backend.get_value(expected_gradient * 2)
        )
        model.fit(dataset)
        self.assertEqual(backend.eval(layer.v), -2)

        # Next test with NaN gradients.
        backend.set_value(have_nan_gradients, True)
        model.fit(dataset)
        # Variable should not be updated
        self.assertEqual(backend.eval(layer.v), -2)

        # Test with finite gradients again
        backend.set_value(have_nan_gradients, False)
        # The loss scale will be halved due to the NaNs, so the gradient will
        # also be halved
        backend.set_value(
            expected_gradient, backend.get_value(expected_gradient / 2)
        )
        model.fit(dataset)
        self.assertEqual(backend.eval(layer.v), -3)

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_compile_wraps_with_loss_scale_optimizer(self):
        x = layers.Input(shape=(1,))
        y = mp_test_util.MultiplyLayer()(x)

        # The non-legacy optimizer is only supported in TF2
        use_legacy_optimizer = (
            not tf.__internal__.tf2.enabled() or not tf.executing_eagerly()
        )

        with policy.policy_scope("mixed_float16"):
            # Test optimizer is automatically wrapped with LSO
            model = models.Model(x, y)
            if use_legacy_optimizer:
                optimizer = gradient_descent.SGD(1.0)
            else:
                optimizer = sgd.SGD(1.0)
            model.compile(optimizer, "mse")
            self.assertIsInstance(
                model.optimizer, loss_scale_optimizer.BaseLossScaleOptimizer
            )
            self.assertEqual(
                backend.get_value(model.optimizer.learning_rate), 1.0
            )

            # Test optimizer specified as string is automatically wrapped in LSO
            model = models.Model(x, y)
            model.compile("sgd", "mse")
            self.assertIsInstance(
                model.optimizer, loss_scale_optimizer.BaseLossScaleOptimizer
            )

            # Test if an LSO is passed, optimizer is not automatically wrapped
            # with another LSO
            model = models.Model(x, y)
            if use_legacy_optimizer:
                optimizer = gradient_descent.SGD(1.0)
            else:
                optimizer = sgd.SGD(1.0)
            optimizer = loss_scale_optimizer.BaseLossScaleOptimizer(
                optimizer, dynamic_growth_steps=2
            )
            model.compile(optimizer, "mse")
            self.assertIsInstance(
                model.optimizer, loss_scale_optimizer.BaseLossScaleOptimizer
            )
            self.assertEqual(model.optimizer.dynamic_growth_steps, 2)

        with policy.policy_scope("mixed_bfloat16"):
            # Test mixed_bfloat16 models are not automatically wrapped with LSO
            model = models.Model(x, y)
            if use_legacy_optimizer:
                optimizer = gradient_descent.SGD(1.0)
            else:
                optimizer = sgd.SGD(1.0)
            model.compile(optimizer, "mse")
            self.assertNotIsInstance(
                model.optimizer, loss_scale_optimizer.BaseLossScaleOptimizer
            )
            self.assertIsInstance(
                model.optimizer,
                gradient_descent.SGD if use_legacy_optimizer else sgd.SGD,
            )

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_pass_invalid_optimizer_with_loss_scaling(self):
        with policy.policy_scope(policy.Policy("mixed_float16")):
            x = layers.Input(shape=(1,))
            y = mp_test_util.MultiplyLayer()(x)
            model = models.Model(x, y)
            if tf.executing_eagerly():
                error_msg = "Use a `tf.keras` Optimizer instead"
            else:
                error_msg = 'optimizer" must be an instance of '
            with self.assertRaisesRegex(ValueError, error_msg):
                model.compile(optimizer_v1.SGD(1.0), "mse")

    @test_combinations.generate(
        test_combinations.combine(mode=["graph", "eager"])
    )
    def test_functional_model_loss_dtype(self):
        with policy.policy_scope("float16"):
            x = layers.Input(shape=(1,))
            y = mp_test_util.MultiplyLayer()(x)
            model = models.Model(x, y)
            model.add_loss(tf.cast(y, "float32"))
            # The loss should not be casted to the policy's dtype.
            self.assertEqual(model.losses[0].dtype, "float32")

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        {
            "testcase_name": "base",
            "strategy_fn": default_strategy_fn,
        },
        {
            "testcase_name": "distribute",
            "strategy_fn": create_mirrored_strategy,
        },
        {
            "testcase_name": "base_h5",
            "strategy_fn": default_strategy_fn,
            "h5": True,
        },
        {
            "testcase_name": "distribute_h5",
            "strategy_fn": create_mirrored_strategy,
            "h5": True,
        },
    )
    def test_save_weights_with_autocast_vars(self, strategy_fn, h5=False):
        with strategy_fn().scope():
            with policy.policy_scope("mixed_float16"):
                x = layers.Input(shape=(1,), batch_size=2)
                layer = mp_test_util.MultiplyLayer(assert_type=tf.float16)
                y = layer(x)
                model = models.Model(inputs=x, outputs=y)

        model.set_weights([np.array(100.0)])
        x = np.ones((2, 1))
        self.assertAllClose(backend.get_value(model(x)), x * 100.0)
        suffix = ".h5" if h5 else ""
        weights_file = os.path.join(self.get_temp_dir(), "weights" + suffix)
        model.save_weights(weights_file)

        model.set_weights([np.array(200.0)])
        self.assertAllClose(backend.get_value(model(x)), x * 200.0)
        model.load_weights(weights_file)
        self.assertAllClose(backend.get_value(model(x)), x * 100.0)
        self.assertEqual(model.get_weights(), [np.array(100.0)])

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        {
            "testcase_name": "base",
            "strategy_fn": default_strategy_fn,
        },
        {
            "testcase_name": "distribute",
            "strategy_fn": create_mirrored_strategy,
        },
        {
            "testcase_name": "distribute_legacy",
            "strategy_fn": create_mirrored_strategy,
            "use_legacy_optimizer": True,
        },
        {
            "testcase_name": "different_var_name",
            "strategy_fn": default_strategy_fn,
            "var_name": "w",
        },
        {
            "testcase_name": "different_var_name_distribute",
            "strategy_fn": create_mirrored_strategy,
            "var_name": "w",
        },
    )
    def test_save_slot_variables_with_autocast_vars(
        self, strategy_fn, var_name="v", use_legacy_optimizer=False
    ):
        if not tf.__internal__.tf2.enabled():
            # The non-legacy optimizer is only supported in TF2
            use_legacy_optimizer = True
        p = policy.Policy("mixed_float16")
        with strategy_fn().scope(), policy.policy_scope(p):
            x = layers.Input(shape=(2,), batch_size=2)
            # Having a var_name other than 'v' tests that a fixed bug
            # (b/134713714) does not reoccur. The bug was that a crash would
            # occur when saving a checkpoint where an AutoCastVariable with a
            # slot variable would have a different name than the layer
            # attribute's name (layer.v in this case).
            layer = mp_test_util.MultiplyLayer(
                assert_type=tf.float16, var_name=var_name
            )
            y = layer(x)
            model = models.Model(inputs=x, outputs=y)
            if use_legacy_optimizer:
                opt = gradient_descent.SGD(1.0, 1.0)
            else:
                opt = sgd.SGD(1.0, 1.0)
            opt = loss_scale_optimizer.BaseLossScaleOptimizer(
                opt, dynamic=False, initial_scale=1
            )
            model.compile(
                optimizer=opt,
                loss="mse",
                run_eagerly=test_utils.should_run_eagerly(),
            )

        def get_momentum_slot():
            if use_legacy_optimizer:
                return opt.get_slot(layer.v, "momentum")
            else:
                return opt.inner_optimizer.momentums[0]

        model.fit(np.ones((2, 2)), np.zeros((2, 2)), batch_size=2)
        weights_file = os.path.join(self.get_temp_dir(), "weights")
        model.save_weights(weights_file)
        saved_slot = backend.get_value(get_momentum_slot())

        model.fit(np.ones((2, 2)), np.zeros((2, 2)), batch_size=2)
        new_slot = backend.get_value(get_momentum_slot())
        self.assertNotEqual(new_slot, saved_slot)

        model.load_weights(weights_file)
        restored_slot = backend.get_value(get_momentum_slot())
        self.assertEqual(restored_slot, saved_slot)

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(*TESTCASES)
    def test_save_weights_with_dynamic_loss_scaling(self, strategy_fn):
        strategy = strategy_fn()
        if (
            isinstance(strategy, tf.distribute.MirroredStrategy)
            and not tf.executing_eagerly()
        ):
            # TODO(b/121381184): Enable running the test in this case.
            return

        # The non-legacy optimizer is only supported in TF2
        use_legacy_optimizer = not tf.__internal__.tf2.enabled()

        # Create and run model.
        with strategy.scope():
            x = layers.Input(shape=(2,), batch_size=2, dtype=tf.float32)
            y = mp_test_util.MultiplyLayer(assert_type=tf.float32)(x)
            model = models.Model(inputs=x, outputs=y)

            if use_legacy_optimizer:
                opt = gradient_descent.SGD(1.0)
            else:
                opt = sgd.SGD(1.0)
            opt = loss_scale_optimizer.BaseLossScaleOptimizer(
                opt, initial_scale=1.0, dynamic_growth_steps=2.0
            )
            model.compile(
                optimizer=opt,
                loss="mse",
                run_eagerly=test_utils.should_run_eagerly(),
            )
        # Run for 3 steps (6 examples with a batch size of 2)
        model.fit(np.zeros((6, 2)), np.zeros((6, 2)), batch_size=2)
        self.assertEqual(backend.get_value(opt.loss_scale), 2)
        self.assertEqual(backend.get_value(opt.dynamic_counter), 1)

        # Save model weights.
        save_prefix = os.path.join(self.get_temp_dir(), "ckpt")
        model.save_weights(save_prefix)

        # Run model again for 1 step (2 examples with a batch size of 2)
        model.fit(np.zeros((2, 2)), np.zeros((2, 2)), batch_size=2)
        self.assertEqual(backend.get_value(opt.loss_scale), 4)
        self.assertEqual(backend.get_value(opt.dynamic_counter), 0)

        # Load model weights and ensure loss scale weights are restored.
        model.load_weights(save_prefix)
        self.assertEqual(backend.get_value(opt.loss_scale), 2)
        self.assertEqual(backend.get_value(opt.dynamic_counter), 1)

    @test_combinations.run_all_keras_modes
    def test_restore_old_loss_scale_checkpoint(self):
        # Ensure a checkpoint from TF 2.2 can be loaded. The checkpoint format
        # of LossScaleOptimizer changed, but old checkpoints can still be loaded
        # into the legacy optimizers.
        opt = gradient_descent.SGD(0.1, momentum=0.1)
        opt = loss_scale_optimizer.LossScaleOptimizer(opt)
        model = sequential.Sequential(
            [
                core.Dense(
                    2,
                )
            ]
        )

        # The checkpoint and expected values were obtained from the program in
        # testdata/BUILD.
        ckpt_dir = os.path.join(
            flags.FLAGS["test_srcdir"].value,
            "org_keras/keras",
            "mixed_precision/testdata/lso_ckpt_tf2.2",
        )
        # ckpt_dir = test.test_src_dir_path(
        #     'python/keras/mixed_precision/testdata/lso_ckpt_tf2.2')
        model.load_weights(os.path.join(ckpt_dir, "ckpt"))
        model.compile(opt, "mse", run_eagerly=test_utils.should_run_eagerly())
        model(np.zeros((2, 2)))  # Create model weights
        opt._create_all_weights(model.weights)
        expected_kernel = np.array(
            [[9.229685, 10.901115], [10.370763, 9.757362]]
        )
        expected_slot = np.array([[10.049943, 9.917691], [10.049943, 9.917691]])
        self.assertAllClose(self.evaluate(model.weights[0]), expected_kernel)
        self.assertAllClose(
            self.evaluate(opt.get_slot(model.weights[0], "momentum")),
            expected_slot,
        )
        self.assertEqual(self.evaluate(opt.loss_scale), 32768)
        self.assertEqual(self.evaluate(opt.dynamic_counter), 1)

        # Check restoring works even after the model is compiled and the weights
        # have been created.
        model.fit(np.random.normal(size=(2, 2)), np.random.normal(size=(2, 2)))
        self.assertNotAllClose(self.evaluate(model.weights[0]), expected_kernel)
        self.assertNotAllClose(
            self.evaluate(opt.get_slot(model.weights[0], "momentum")),
            expected_slot,
        )
        model.load_weights(os.path.join(ckpt_dir, "ckpt"))
        self.assertAllClose(self.evaluate(model.weights[0]), expected_kernel)
        self.assertAllClose(
            self.evaluate(opt.get_slot(model.weights[0], "momentum")),
            expected_slot,
        )
        self.assertEqual(self.evaluate(opt.loss_scale), 32768)
        self.assertEqual(self.evaluate(opt.dynamic_counter), 1)

    def test_restore_old_saved_model(self):
        saved_model_dir = os.path.join(
            flags.FLAGS["test_srcdir"].value,
            "org_keras/keras",
            "mixed_precision/testdata/lso_savedmodel_tf2.2",
        )
        # saved_model_dir = test.test_src_dir_path(
        #     'python/keras/mixed_precision/testdata/'
        #     'lso_savedmodel_tf2.2')
        model = save.load_model(saved_model_dir)
        expected_kernel = np.array(
            [[9.229685, 10.901115], [10.370763, 9.757362]]
        )
        self.assertAllClose(backend.eval(model.weights[0]), expected_kernel)
        self.assertEqual(
            type(model.optimizer), loss_scale_optimizer.LossScaleOptimizer
        )

    @test_combinations.run_all_keras_modes
    @parameterized.named_parameters(
        {
            "testcase_name": "base",
            "strategy_fn": default_strategy_fn,
        },
        {
            "testcase_name": "distribute",
            "strategy_fn": create_mirrored_strategy,
        },
        {
            "testcase_name": "base_h5",
            "strategy_fn": default_strategy_fn,
            "h5": True,
        },
        {
            "testcase_name": "distribute_h5",
            "strategy_fn": create_mirrored_strategy,
            "h5": True,
        },
    )
    def test_save_model_with_dynamic_loss_scaling(self, strategy_fn, h5=False):
        # TODO(reedwm): Support and test saving model with a mixed_[b]float16
        # policy as well.
        strategy = strategy_fn()
        if (
            isinstance(strategy, tf.distribute.MirroredStrategy)
            and not tf.executing_eagerly()
        ):
            # TODO(b/121381184): Enable running the test in this case.
            return

        # Create and run model.
        with strategy.scope():
            x = layers.Input(shape=(2,), batch_size=2, dtype=tf.float32)
            y = mp_test_util.MultiplyLayer()(x)
            model = models.Model(inputs=x, outputs=y)

            # Only test the legacy optimizer. The new optimizer does not
            # support saving optimizer weights.
            opt = gradient_descent.SGD(1.0)
            opt = loss_scale_optimizer.LossScaleOptimizer(
                opt, initial_scale=1.0, dynamic_growth_steps=2.0
            )
            model.compile(
                optimizer=opt,
                loss="mse",
                run_eagerly=test_utils.should_run_eagerly(),
            )
        # Run for 3 steps (6 examples with a batch size of 2)
        model.fit(np.ones((6, 2)), np.zeros((6, 2)), batch_size=2)
        self.assertEqual(backend.get_value(opt.loss_scale), 2)
        self.assertEqual(backend.get_value(opt.dynamic_counter), 1)
        (weight,) = model.trainable_weights
        orig_weight = backend.get_value(weight)

        # Save model weights.
        save_path = os.path.join(self.get_temp_dir(), "model")
        model.save(save_path, save_format="h5" if h5 else "tf")

        # Run model again for 1 step (2 examples with a batch size of 2)
        model.fit(np.ones((2, 2)), np.zeros((2, 2)), batch_size=2)
        new_weight = backend.get_value(weight)
        self.assertNotEqual(new_weight, orig_weight)
        self.assertEqual(backend.get_value(opt.loss_scale), 4)
        self.assertEqual(backend.get_value(opt.dynamic_counter), 0)

        # Load model weights and ensure loss scale weights are restored.
        model = save.load_model(
            save_path,
            custom_objects={"MultiplyLayer": mp_test_util.MultiplyLayer},
        )
        (weight,) = model.trainable_weights
        loaded_weight = backend.get_value(weight)
        self.assertEqual(loaded_weight, orig_weight)
        # Currently the loss scale isn't always saved when the model is saved
        # with Model.save(). So we assert the loss scale either has the value
        # when it was saved, or the value it was initialized with.
        # TODO(reedwm): Always save/restore the loss scale with Model.save().
        self.assertIn(backend.get_value(model.optimizer.loss_scale), (1, 2))
        self.assertIn(
            backend.get_value(model.optimizer.dynamic_counter), (0, 1)
        )

        # Test optimizer attributes and type
        self.assertEqual(model.optimizer.initial_scale, 1.0)
        self.assertEqual(model.optimizer.dynamic_growth_steps, 2.0)
        self.assertEqual(
            type(model.optimizer), loss_scale_optimizer.LossScaleOptimizer
        )


class ApplicationModelTest(test_combinations.TestCase):
    """Tests that application models can be built with mixed precision.

    This does not test that such models can be trained in mixed precision, as
    doing so takes too much time for a unit test.
    """

    @parameterized.named_parameters(
        ("densenet", densenet.DenseNet121),
        ("efficientnet", efficientnet.EfficientNetB0),
        ("inception_resnet_v2", inception_resnet_v2.InceptionResNetV2),
        ("inception_v3", inception_v3.InceptionV3),
        ("mobilenet", mobilenet.MobileNet),
        ("nasnet", nasnet.NASNetMobile),
        ("vgg16", vgg16.VGG16),
        ("xception", xception.Xception),
        ("resnet50", resnet.ResNet50),
    )
    def test_application_model(self, app):
        # Run on CPU since model weights may exhaust GPU memory
        with policy.policy_scope("mixed_float16"), tf.device("/CPU:0"):
            app(weights=None)


if __name__ == "__main__":
    base_layer_utils.enable_v2_dtype_behavior()
    tf.test.main()
