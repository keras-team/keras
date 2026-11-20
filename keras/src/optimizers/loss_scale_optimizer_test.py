import io
import os
import warnings
import zipfile

import h5py
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.backend import config
from keras.src.optimizers.adam import Adam
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.optimizers.sgd import SGD


class LossScaleOptimizerTest(testing.TestCase):
    def _skip_test_for_stateless(self, stateless):
        if not stateless and config.backend() == "jax":
            self.skipTest(
                "LossScaleOptimizer must use stateless_apply with JAX."
            )
        if stateless and config.backend() == "tensorflow":
            self.skipTest(
                "stateless_apply is not supported with the TF backend."
            )

    def test_config(self):
        inner_optimizer = SGD(
            learning_rate=0.5,
            momentum=0.06,
            nesterov=True,
            weight_decay=0.004,
        )
        optimizer = LossScaleOptimizer(inner_optimizer)
        self.run_class_serialization_test(optimizer)

    def test_apply_with_no_vars(self):
        self._skip_test_for_stateless(False)

        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(inner_optimizer)
        grads = [ops.array([1.0, 6.0, 7.0, 2.0]) * optimizer.initial_scale]
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        optimizer.build(vars)
        optimizer.apply(grads)
        self.assertAllClose(
            vars[0], [0.5, -1.0, -0.5, 3.0], rtol=1e-4, atol=1e-4
        )

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_finite_step(self, stateless):
        self._skip_test_for_stateless(stateless)

        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(inner_optimizer)
        grads = [ops.array([1.0, 6.0, 7.0, 2.0]) * optimizer.initial_scale]
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        if stateless:
            optimizer.build(vars)
            vars, _ = optimizer.stateless_apply(
                [v.value for v in optimizer.variables],
                grads,
                [v.value for v in vars],
            )
        else:
            optimizer.apply(grads, vars)
        self.assertAllClose(
            vars[0], [0.5, -1.0, -0.5, 3.0], rtol=1e-4, atol=1e-4
        )

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_finite_step_with_inner_loss_scale(self, stateless):
        self._skip_test_for_stateless(stateless)

        # Ensure that the inner loss scale does not interfere with the update.
        inner_optimizer = SGD(learning_rate=0.5, loss_scale_factor=100)
        optimizer = LossScaleOptimizer(inner_optimizer)
        grads = [ops.array([1.0, 6.0, 7.0, 2.0]) * optimizer.initial_scale]
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        if stateless:
            optimizer.build(vars)
            vars, _ = optimizer.stateless_apply(
                [v.value for v in optimizer.variables],
                grads,
                [v.value for v in vars],
            )
        else:
            optimizer.apply(grads, vars)
        self.assertAllClose(
            vars[0], [0.5, -1.0, -0.5, 3.0], rtol=1e-4, atol=1e-4
        )

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_infinite_step(self, stateless):
        self._skip_test_for_stateless(stateless)

        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(inner_optimizer)
        grads = [ops.array([np.inf, np.inf, np.inf, np.inf])]
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        if stateless:
            optimizer.build(vars)
            vars, _ = optimizer.stateless_apply(
                [v.value for v in optimizer.variables],
                grads,
                [v.value for v in vars],
            )
        else:
            optimizer.apply(grads, vars)
        self.assertAllClose(vars[0], [1.0, 2.0, 3.0, 4.0], rtol=1e-4, atol=1e-4)

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_finite_step_with_overwrite(self, stateless):
        self._skip_test_for_stateless(stateless)

        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(inner_optimizer)
        grads = [ops.array([1.0, 6.0, 7.0, 2.0])]
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        vars[0].overwrite_with_gradient = True

        if stateless:
            optimizer.build(vars)
            vars, _ = optimizer.stateless_apply(
                [v.value for v in optimizer.variables],
                grads,
                [v.value for v in vars],
            )
        else:
            optimizer.apply(grads, vars)
        self.assertAllClose(vars[0], grads[0])

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_downscaling(self, stateless):
        self._skip_test_for_stateless(stateless)

        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(inner_optimizer, initial_scale=400.0)
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        optimizer.build(vars)
        opt_var_values = [v.value for v in optimizer.variables]
        grads = [ops.array([np.inf, np.inf, np.inf, np.inf])]
        for _ in range(4):
            if stateless:
                _, opt_var_values = optimizer.stateless_apply(
                    opt_var_values, grads, [v.value for v in vars]
                )
                for ref_v, v in zip(optimizer.variables, opt_var_values):
                    ref_v.assign(v)
            else:
                optimizer.apply(grads, vars)
        self.assertAllClose(optimizer.scale_loss(1.0), 25.0)

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_upscaling(self, stateless):
        self._skip_test_for_stateless(stateless)

        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(
            inner_optimizer,
            initial_scale=2.0,
            dynamic_growth_steps=2,
        )
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        optimizer.build(vars)
        opt_var_values = [v.value for v in optimizer.variables]
        grads = [ops.array([1.0, 6.0, 7.0, 2.0])]
        for _ in range(8):
            if stateless:
                _, opt_var_values = optimizer.stateless_apply(
                    opt_var_values, grads, [v.value for v in vars]
                )
                for ref_v, v in zip(optimizer.variables, opt_var_values):
                    ref_v.assign(v)
            else:
                optimizer.apply(grads, vars)
        self.assertAllClose(optimizer.scale_loss(1.0), 32.0)

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_iterations_update(self, stateless):
        self._skip_test_for_stateless(stateless)

        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(inner_optimizer)
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        optimizer.build(vars)
        opt_var_values = [v.value for v in optimizer.variables]
        grads = [ops.array([1.0, 6.0, 7.0, 2.0])]

        self.assertEqual(optimizer.iterations.value, 0)

        for i in range(3):
            if stateless:
                _, opt_var_values = optimizer.stateless_apply(
                    opt_var_values, grads, [v.value for v in vars]
                )
                for ref_v, v in zip(optimizer.variables, opt_var_values):
                    ref_v.assign(v)
            else:
                optimizer.apply(grads, vars)
            self.assertEqual(optimizer.iterations.value, i + 1)

    def test_serialization(self):
        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(
            inner_optimizer,
            initial_scale=3.0,
            dynamic_growth_steps=2,
            name="test_opt",
        )
        config = optimizer.get_config()
        self.assertLen(config, 4)
        self.assertEqual(config["name"], "test_opt")
        self.assertEqual(config["initial_scale"], 3.0)
        self.assertEqual(config["dynamic_growth_steps"], 2)
        self.assertIn("inner_optimizer", config)
        LossScaleOptimizer.from_config(config)

    def test_init_dynamic_arg(self):
        inner_optimizer = SGD(learning_rate=0.5)

        # dynamic=True is supported
        LossScaleOptimizer(inner_optimizer, dynamic=True)

        # dynamic=False is not supported
        with self.assertRaisesRegex(ValueError, "set `loss_scale_factor`"):
            LossScaleOptimizer(inner_optimizer, dynamic=False)

    def test_init_unsupported_arg(self):
        inner_optimizer = SGD(learning_rate=0.5)
        with self.assertRaisesRegex(ValueError, "arguments: `foo`, `bar`"):
            LossScaleOptimizer(inner_optimizer, foo=True, bar=3)

    @parameterized.named_parameters(
        ("weight_decay", "weight_decay", 0.5),
        ("clipnorm", "clipnorm", 0.5),
        ("global_clipnorm", "global_clipnorm", 0.5),
        ("clipvalue", "clipvalue", 0.5),
        ("use_ema", "use_ema", True),
        ("ema_momentum", "ema_momentum", 0.5),
        ("ema_overwrite_frequency", "ema_overwrite_frequency", 2),
        ("loss_scale_factor", "loss_scale_factor", 0.5),
        ("gradient_accumulation_steps", "gradient_accumulation_steps", 2),
    )
    def test_init_base_optimizer_unsupported_args(self, arg_name, arg_value):
        inner_optimizer = SGD(learning_rate=0.5)
        with self.assertRaisesRegex(ValueError, "on the `inner_optimizer`"):
            LossScaleOptimizer(inner_optimizer, **{arg_name: arg_value})

    def test_deserialization_backwards_compatibility(self):
        # Test deserializing with a config that has all the unsupported
        # arguments from the base optimizer (which are no longer serialized)
        config = {
            "name": "loss_scale_optimizer",
            "weight_decay": None,
            "clipnorm": None,
            "global_clipnorm": None,
            "clipvalue": None,
            "use_ema": False,
            "ema_momentum": 0.99,
            "ema_overwrite_frequency": None,
            "loss_scale_factor": None,
            "gradient_accumulation_steps": None,
            "inner_optimizer": {
                "module": "keras.optimizers",
                "class_name": "SGD",
                "config": {
                    "name": "SGD",
                    "learning_rate": 0.5,
                    "weight_decay": None,
                    "clipnorm": None,
                    "global_clipnorm": None,
                    "clipvalue": None,
                    "use_ema": False,
                    "ema_momentum": 0.99,
                    "ema_overwrite_frequency": None,
                    "loss_scale_factor": None,
                    "gradient_accumulation_steps": None,
                    "momentum": 0.0,
                    "nesterov": False,
                },
                "registered_name": None,
            },
            "initial_scale": 2.0,
            "dynamic_growth_steps": 2,
        }
        LossScaleOptimizer.from_config(config)

    def test_save_and_load_own_variables(self):
        inner_optimizer = Adam(learning_rate=0.01)
        optimizer = LossScaleOptimizer(
            inner_optimizer, initial_scale=128.0, dynamic_growth_steps=4
        )
        var = backend.Variable([1.0, 2.0])
        optimizer.build([var])
        optimizer.step_counter.assign(3)
        optimizer.dynamic_scale.assign(64.0)

        store = {}
        optimizer.save_own_variables(store)
        self.assertLen(store, len(optimizer._variables))

        restored = LossScaleOptimizer(
            Adam(learning_rate=0.01),
            initial_scale=128.0,
            dynamic_growth_steps=4,
        )
        restored.build([backend.Variable([1.0, 2.0])])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored.load_own_variables(store)
        self.assertEmpty(caught)
        self.assertEqual(int(restored.step_counter.numpy()), 3)
        self.assertAllClose(restored.dynamic_scale.numpy(), 64.0)

    def test_load_own_variables_legacy_store(self):
        inner_optimizer = Adam(learning_rate=0.01)
        optimizer = LossScaleOptimizer(
            inner_optimizer, initial_scale=128.0, dynamic_growth_steps=4
        )
        var = backend.Variable([1.0, 2.0])
        optimizer.build([var])
        optimizer.step_counter.assign(2)
        optimizer.dynamic_scale.assign(32.0)
        inner_optimizer._iterations.assign(5)

        legacy_store = {
            str(i): v.numpy() for i, v in enumerate(optimizer.variables)
        }
        self.assertLen(legacy_store, len(optimizer.variables))

        restored = LossScaleOptimizer(
            Adam(learning_rate=0.01),
            initial_scale=128.0,
            dynamic_growth_steps=4,
        )
        restored.build([backend.Variable([1.0, 2.0])])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored.load_own_variables(legacy_store)
        self.assertEmpty(caught)
        self.assertEqual(int(restored.step_counter.numpy()), 2)
        self.assertAllClose(restored.dynamic_scale.numpy(), 32.0)
        self.assertEqual(int(restored.iterations.numpy()), 5)

    @pytest.mark.requires_trainable_backend
    def test_model_save_does_not_duplicate_inner_optimizer_variables(self):
        model = models.Sequential([layers.Input(shape=(4,)), layers.Dense(2)])
        optimizer = LossScaleOptimizer(
            Adam(learning_rate=0.01),
            initial_scale=16.0,
            dynamic_growth_steps=2,
        )
        model.compile(optimizer=optimizer, loss="mse")
        x = np.ones((2, 4), dtype=np.float32)
        y = np.ones((2, 2), dtype=np.float32)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)
        model.train_on_batch(x, y)

        filepath = os.path.join(self.get_temp_dir(), "lso_model.keras")
        model.save(filepath)

        with zipfile.ZipFile(filepath, "r") as archive:
            weights_bytes = io.BytesIO(archive.read("model.weights.h5"))
        with h5py.File(weights_bytes, "r") as weights:
            optimizer_datasets = []
            weights.visititems(
                lambda path, item: (
                    optimizer_datasets.append(path)
                    if isinstance(item, h5py.Dataset)
                    and path.startswith("optimizer/")
                    else None
                )
            )

        own_paths = [
            p for p in optimizer_datasets if p.startswith("optimizer/vars/")
        ]
        inner_paths = [
            p
            for p in optimizer_datasets
            if p.startswith("optimizer/inner_optimizer/vars/")
        ]
        self.assertLen(own_paths, len(model.optimizer._variables))
        self.assertLen(
            inner_paths, len(model.optimizer.inner_optimizer._variables)
        )
        self.assertLen(
            optimizer_datasets,
            len(model.optimizer._variables)
            + len(model.optimizer.inner_optimizer._variables),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reloaded = saving.load_model(filepath)
        self.assertEmpty(caught)
        for orig_var, loaded_var in zip(
            model.optimizer.variables, reloaded.optimizer.variables
        ):
            self.assertAllClose(orig_var.numpy(), loaded_var.numpy())
