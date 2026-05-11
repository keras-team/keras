import numpy as np

from keras.src import backend
from keras.src import callbacks
from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.saving.serialization_lib import enable_unsafe_deserialization


class MultiOptimizerTest(testing.TestCase):
    def test_optimizer_map_matching(self):
        with backend.name_scope("dense"):
            w_dense = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("conv"):
            w_conv = backend.Variable([[1.0]], name="kernel")
        w_other = backend.Variable([[1.0]], name="bias")

        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)

        # Test regex matching
        var_identifier = ["^dense/.*", "^conv/.*"]
        opt_map = optimizers.OptimizerMap([opt_adam, opt_sgd], var_identifier)

        self.assertEqual(opt_map(w_dense), opt_adam)
        self.assertEqual(opt_map(w_conv), opt_sgd)
        self.assertIsNone(opt_map(w_other))

        # Test direct variable reference matching
        opt_map_direct = optimizers.OptimizerMap([opt_adam], [[w_dense]])
        self.assertEqual(opt_map_direct(w_dense), opt_adam)
        self.assertIsNone(opt_map_direct(w_conv))

        # Test callable matching
        opt_map_callable = optimizers.OptimizerMap(
            [opt_adam], [lambda x: "dense" in (x.path or x.name)]
        )
        self.assertEqual(opt_map_callable(w_dense), opt_adam)
        self.assertIsNone(opt_map_callable(w_conv))

        # Test conflict detection
        opt_map_conflict = optimizers.OptimizerMap(
            [opt_adam, opt_sgd], [".*kernel.*", ".*dense.*"]
        )
        with self.assertRaises(ValueError):
            opt_map_conflict(w_dense)

    def test_multi_optimizer_initialization(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)
        default_opt = optimizers.RMSprop(learning_rate=1e-4)

        opt_map = optimizers.OptimizerMap(
            [opt_adam, opt_sgd], ["^dense_1/.*", "^dense_2/.*"]
        )

        multi_opt = optimizers.MultiOptimizer(opt_map, default_opt)

        # Verify unique optimizers list, with default_optimizer at the end
        self.assertEqual(multi_opt.num_optimizers, 3)
        self.assertEqual(multi_opt.get_optimizer(0), opt_adam)
        self.assertEqual(multi_opt.get_optimizer(1), opt_sgd)
        self.assertEqual(multi_opt.get_optimizer(2), default_opt)

    def test_learning_rate_get_set(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)
        default_opt = optimizers.RMSprop(learning_rate=1e-4)

        opt_map = optimizers.OptimizerMap(
            [opt_adam, opt_sgd], ["^dense_1/.*", "^dense_2/.*"]
        )
        multi_opt = optimizers.MultiOptimizer(opt_map, default_opt)

        # Getter should return the first optimizer's learning rate
        self.assertAllClose(multi_opt.learning_rate, 1e-3)

        # Setter should only update the first optimizer's learning rate
        multi_opt.learning_rate = 5e-3
        self.assertAllClose(opt_adam.learning_rate, 5e-3)
        self.assertAllClose(opt_sgd.learning_rate, 1e-2)
        self.assertAllClose(default_opt.learning_rate, 1e-4)

    def test_default_fallback(self):
        with backend.name_scope("dense_1"):
            w_mapped = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("other"):
            w_unmapped = backend.Variable([[1.0]], name="kernel")

        opt_adam = optimizers.Adam(learning_rate=1e-3)
        default_opt = optimizers.SGD(learning_rate=1e-2)

        opt_map = optimizers.OptimizerMap([opt_adam], ["^dense_1/.*"])
        multi_opt = optimizers.MultiOptimizer(opt_map, default_opt)

        multi_opt.build([w_mapped, w_unmapped])

        self.assertEqual(
            multi_opt._get_optimizer_for_variable(w_mapped), opt_adam
        )
        self.assertEqual(
            multi_opt._get_optimizer_for_variable(w_unmapped), default_opt
        )

    def test_stateful_apply(self):
        if backend.backend() == "jax":
            self.skipTest(
                "stateful_apply is not supported with the JAX backend"
            )
        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[2.0, 2.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[3.0, 3.0]], name="kernel")
        with backend.name_scope("other"):
            w_fallback = backend.Variable([[4.0, 4.0]], name="kernel")

        opt_sgd_1 = optimizers.SGD(learning_rate=0.1)
        opt_sgd_2 = optimizers.SGD(learning_rate=1.0)
        default_opt = optimizers.SGD(learning_rate=0.0)

        opt_map = optimizers.OptimizerMap(
            [opt_sgd_1, opt_sgd_2], ["^dense_1/.*", "^dense_2/.*"]
        )
        multi_opt = optimizers.MultiOptimizer(opt_map, default_opt)

        multi_opt.build([w1, w2, w_fallback])

        grads = [
            backend.convert_to_tensor([[1.0, 1.0]]),
            backend.convert_to_tensor([[1.0, 1.0]]),
            backend.convert_to_tensor([[1.0, 1.0]]),
        ]

        multi_opt.apply(grads, [w1, w2, w_fallback])

        # w1: updated by opt_sgd_1 (lr=0.1) -> 2.0 - 0.1 * 1.0 = 1.9
        self.assertAllClose(w1.numpy(), [[1.9, 1.9]])
        # w2: updated by opt_sgd_2 (lr=1.0) -> 3.0 - 1.0 * 1.0 = 2.0
        self.assertAllClose(w2.numpy(), [[2.0, 2.0]])
        # w_fallback: updated by default_opt (lr=0.0) -> Remains 4.0
        self.assertAllClose(w_fallback.numpy(), [[4.0, 4.0]])

        # Verify iteration increment
        self.assertEqual(int(multi_opt.iterations), 1)

    def test_stateless_apply(self):
        if backend.backend() == "tensorflow":
            self.skipTest(
                "stateless_apply is not supported with the TensorFlow backend "
                "(as it is incompatible with tf.distribute)."
            )

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[2.0, 2.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[3.0, 3.0]], name="kernel")

        opt_sgd_1 = optimizers.SGD(learning_rate=0.1)
        opt_sgd_2 = optimizers.SGD(learning_rate=1.0)

        opt_map = optimizers.OptimizerMap([opt_sgd_1], ["^dense_1/.*"])
        multi_opt = optimizers.MultiOptimizer(opt_map, opt_sgd_2)

        multi_opt.build([w1, w2])

        grads = [
            backend.convert_to_tensor([[1.0, 1.0]]),
            backend.convert_to_tensor([[1.0, 1.0]]),
        ]

        trainable_variables = [w1.numpy(), w2.numpy()]
        optimizer_variables = [v.numpy() for v in multi_opt.variables]

        new_trainable_vars, new_opt_vars = multi_opt.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # w1 updated by SGD(0.1): 2.0 - 0.1 = 1.9
        self.assertAllClose(new_trainable_vars[0], [[1.9, 1.9]])
        # w2 updated by fallback SGD(1.0): 3.0 - 1.0 = 2.0
        self.assertAllClose(new_trainable_vars[1], [[2.0, 2.0]])

    def test_callback_integration(self):
        if backend.backend() in ["numpy", "openvino"]:
            self.skipTest(
                "skipped test due to backend does not support fit function"
            )
        # Simple linear model
        model = models.Sequential(
            [
                layers.Input(shape=(2,)),
                layers.Dense(4, name="dense_1"),
                layers.Dense(1, name="dense_2"),
            ]
        )

        opt_adam = optimizers.Adam(learning_rate=1e-1)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)

        opt_map = optimizers.OptimizerMap([opt_adam], ["^dense_1/.*"])
        multi_opt = optimizers.MultiOptimizer(opt_map, opt_sgd)

        model.compile(optimizer=multi_opt, loss="mse")

        x = np.random.uniform(size=(16, 2))
        y = np.random.uniform(size=(16, 1))

        # Custom schedule that forces an LR adjustment
        lr_scheduler = callbacks.LearningRateScheduler(lambda epoch: 5e-2)

        # Getter/setter tests inside fit
        model.fit(
            x, y, epochs=1, batch_size=8, callbacks=[lr_scheduler], verbose=0
        )

        # Verify callback correctly modified ONLY the
        # first optimizer (opt_adam)
        self.assertAllClose(opt_adam.learning_rate, 5e-2)
        self.assertAllClose(opt_sgd.learning_rate, 1e-2)

    def test_optimizer_map_comprehensive(self):
        with backend.name_scope("dense_1"):
            w_dense1_kernel = backend.Variable([[1.0]], name="kernel")
            w_dense1_bias = backend.Variable([[1.0]], name="bias")
        with backend.name_scope("dense_2"):
            w_dense2_kernel = backend.Variable([[1.0]], name="kernel")
        w_exact = backend.Variable([[1.0]], name="exact_var")
        w_other1 = backend.Variable([[1.0]], name="other1")
        w_other2 = backend.Variable([[1.0]], name="other2")

        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)

        # Comprehensive mapping covering all 5 types of identifiers:
        # 1. String substring matching
        # 2. Regex matching
        # 3. Keras Variable instance direct match
        # 4. Callable match
        # 5. List/Tuple of Keras variables
        opt_map = optimizers.OptimizerMap(
            optimizer=[opt_adam, opt_sgd, opt_adam, opt_sgd, opt_adam],
            variable_identifier=[
                "exact_var",
                "^dense_1/.*",
                w_dense2_kernel,
                lambda var: "bias" in (var.path or var.name),
                [w_other1, w_other2],
            ],
        )

        # 1. Substring match -> opt_adam
        self.assertEqual(opt_map(w_exact), opt_adam)
        # 2. Regex match -> opt_sgd
        self.assertEqual(opt_map(w_dense1_kernel), opt_sgd)
        # 3. Keras Variable direct match -> opt_adam
        self.assertEqual(opt_map(w_dense2_kernel), opt_adam)
        # 4. Callable match -> opt_sgd
        # (and matches dense_1/.* also mapping to opt_sgd)
        self.assertEqual(opt_map(w_dense1_bias), opt_sgd)
        # 5. List/Tuple match -> opt_adam
        self.assertEqual(opt_map(w_other1), opt_adam)
        self.assertEqual(opt_map(w_other2), opt_adam)

    def test_serialization(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)
        default_opt = optimizers.RMSprop(learning_rate=1e-4)

        opt_map = optimizers.OptimizerMap(
            [opt_adam, opt_sgd], ["^dense_1/.*", "^dense_2/.*"]
        )
        multi_opt = optimizers.MultiOptimizer(opt_map, default_opt)

        config = optimizers.serialize(multi_opt)
        reconstructed = optimizers.deserialize(config)

        self.assertEqual(reconstructed.num_optimizers, 3)
        self.assertEqual(
            reconstructed.get_optimizer(0).__class__.__name__, "Adam"
        )
        self.assertEqual(
            reconstructed.get_optimizer(1).__class__.__name__, "SGD"
        )
        self.assertEqual(
            reconstructed.get_optimizer(2).__class__.__name__, "RMSprop"
        )

        # Test with default mapping behavior after reconstruct
        with backend.name_scope("dense_1"):
            w = backend.Variable([[1.0]], name="kernel")
        reconstructed.build([w])
        self.assertEqual(
            reconstructed._get_optimizer_for_variable(w).__class__.__name__,
            "Adam",
        )

    def test_loss_scaling(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)
        default_opt = optimizers.RMSprop(learning_rate=1e-4)

        opt_map = optimizers.OptimizerMap(
            [opt_adam, opt_sgd], ["^dense_1/.*", "^dense_2/.*"]
        )
        multi_opt = optimizers.MultiOptimizer(opt_map, default_opt)

        # Assert that inner optimizers have their loss
        # scaling disabled to prevent double scaling
        self.assertIsNone(opt_adam.loss_scale_factor)
        self.assertIsNone(opt_sgd.loss_scale_factor)
        self.assertIsNone(default_opt.loss_scale_factor)

        # Assert that wrapper scales loss
        # correctly when loss_scale_factor is set
        multi_opt.loss_scale_factor = 100.0
        loss = backend.convert_to_tensor(2.0)
        scaled_loss = multi_opt.scale_loss(loss)
        self.assertAllClose(scaled_loss, 200.0)

    def test_serialization_with_variables(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        default_opt = optimizers.SGD(learning_rate=1e-2)

        with backend.name_scope("dense_1"):
            w = backend.Variable([[1.0]], name="kernel")

        # Mapping uses direct Keras Variable instance in a list
        opt_map = optimizers.OptimizerMap([opt_adam], [[w]])
        multi_opt = optimizers.MultiOptimizer(opt_map, default_opt)

        # Serializing should succeed without TypeError
        config = optimizers.serialize(multi_opt)
        reconstructed = optimizers.deserialize(config)

        # Reconstructed optimizer should build and match variables correctly
        reconstructed.build([w])
        self.assertEqual(
            reconstructed._get_optimizer_for_variable(w).__class__.__name__,
            "Adam",
        )

    def test_serialization_with_callable(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        default_opt = optimizers.SGD(learning_rate=1e-2)

        with backend.name_scope("dense_1"):
            w = backend.Variable([[1.0]], name="kernel")

        # Mapping uses a callable function (serializable lambda)
        opt_map = optimizers.OptimizerMap(
            [opt_adam], [lambda var: "kernel" in (var.path or var.name)]
        )
        multi_opt = optimizers.MultiOptimizer(opt_map, default_opt)

        # Enable unsafe deserialization for lambda loading in test
        enable_unsafe_deserialization()

        config = optimizers.serialize(multi_opt)
        reconstructed = optimizers.deserialize(config)

        # Reconstructed optimizer should build and match
        #  correctly via the deserialized lambda
        reconstructed.build([w])
        self.assertEqual(
            reconstructed._get_optimizer_for_variable(w).__class__.__name__,
            "Adam",
        )
