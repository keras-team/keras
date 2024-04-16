import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.optimizers.sgd import SGD


class LossScaleOptimizerTest(testing.TestCase, parameterized.TestCase):
    def _skip_test_for_stateless(self, stateless):
        if not stateless and backend.backend() == "jax":
            self.skipTest(
                "LossScaleOptimizer must use stateless_apply with JAX."
            )
        if stateless and backend.backend() == "tensorflow":
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
                optimizer.variables, grads, vars
            )
        else:
            optimizer.apply(grads, vars)
        self.assertAllClose(
            vars, [[0.5, -1.0, -0.5, 3.0]], rtol=1e-4, atol=1e-4
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
                optimizer.variables, grads, vars
            )
        else:
            optimizer.apply(grads, vars)
        self.assertAllClose(vars, [[1.0, 2.0, 3.0, 4.0]], rtol=1e-4, atol=1e-4)

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_downscaling(self, stateless):
        self._skip_test_for_stateless(stateless)

        inner_optimizer = SGD(learning_rate=0.5)
        optimizer = LossScaleOptimizer(inner_optimizer, initial_scale=400.0)
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        optimizer.build(vars)
        opt_vars = optimizer.variables
        grads = [ops.array([np.inf, np.inf, np.inf, np.inf])]
        for _ in range(4):
            if stateless:
                _, opt_vars = optimizer.stateless_apply(opt_vars, grads, vars)
                for ref_v, v in zip(optimizer.variables, opt_vars):
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
        opt_vars = optimizer.variables
        grads = [ops.array([1.0, 6.0, 7.0, 2.0])]
        for _ in range(8):
            if stateless:
                _, opt_vars = optimizer.stateless_apply(opt_vars, grads, vars)
                for ref_v, v in zip(optimizer.variables, opt_vars):
                    ref_v.assign(v)
            else:
                optimizer.apply(grads, vars)
        self.assertAllClose(optimizer.scale_loss(1.0), 32.0)
