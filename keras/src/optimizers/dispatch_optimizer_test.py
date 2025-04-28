from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.optimizers.dispatch_optimizer import DispatchOptimizer
from keras.src.optimizers.sgd import SGD


class DispatchOptimizerTest(testing.TestCase):
    def _skip_test_for_stateless(self, stateless):
        if not stateless and backend.backend() == "jax":
            self.skipTest(
                "DispatchOptimizer must use stateless_apply with JAX."
            )
        if stateless and backend.backend() == "tensorflow":
            self.skipTest(
                "stateless_apply is not supported with the TF backend."
            )

    def test_config(self):
        default_optimizer = SGD(
            learning_rate=0.5,
            momentum=0.06,
            nesterov=True,
            weight_decay=0.004,
        )
        optimizer = DispatchOptimizer(default_optimizer)
        self.run_class_serialization_test(optimizer)

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_step(self, stateless):
        self._skip_test_for_stateless(stateless)

        default_optimizer = SGD(learning_rate=0.5)
        optimizer = DispatchOptimizer(default_optimizer)
        grads = [ops.array([1.0, 6.0, 7.0, 2.0])]
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
    def test_per_variable_optimizers(self, stateless):
        self._skip_test_for_stateless(stateless)

        default_optimizer = SGD(learning_rate=0.5)
        per_variable_optimizer = SGD(learning_rate=0.1)

        optimizer = DispatchOptimizer(default_optimizer)
        grads = [
            ops.array([1.0, 6.0, 7.0, 2.0]),
            ops.array([2.0, 12.0, 14.0, 4.0]),
            ops.array([3.0, 18.0, 21.0, 6.0]),
        ]
        vars = [
            backend.Variable([1.0, 2.0, 3.0, 4.0]),
            backend.Variable([5.0, 6.0, 7.0, 8.0]),
            backend.Variable([9.0, 10.0, 11.0, 12.0]),
        ]
        # Two variables share the same optimizer.
        vars[1].optimizer = per_variable_optimizer
        vars[2].optimizer = per_variable_optimizer

        if stateless:
            optimizer.build(vars)
            vars, _ = optimizer.stateless_apply(
                optimizer.variables, grads, vars
            )
        else:
            optimizer.apply(grads, vars)

        # Verify the per-variable optimizer was used.
        self.assertEqual(len(default_optimizer._trainable_variables), 1)
        self.assertEqual(len(per_variable_optimizer._trainable_variables), 2)
        self.assertAllClose(
            vars,
            [
                [0.5, -1.0, -0.5, 3.0],
                [4.8, 4.8, 5.6, 7.6],
                [8.7, 8.2, 8.9, 11.4],
            ],
            rtol=1e-4,
            atol=1e-4,
        )

    @parameterized.named_parameters(("stateless", True), ("stateful", False))
    def test_iterations_update(self, stateless):
        self._skip_test_for_stateless(stateless)

        default_optimizer = SGD(learning_rate=0.5)
        optimizer = DispatchOptimizer(default_optimizer)
        vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
        optimizer.build(vars)
        opt_vars = optimizer.variables
        grads = [ops.array([1.0, 6.0, 7.0, 2.0])]

        self.assertEqual(optimizer.iterations.value, 0)

        for i in range(3):
            if stateless:
                _, opt_vars = optimizer.stateless_apply(opt_vars, grads, vars)
                for ref_v, v in zip(optimizer.variables, opt_vars):
                    ref_v.assign(v)
            else:
                optimizer.apply(grads, vars)
            self.assertEqual(optimizer.iterations.value, i + 1)
