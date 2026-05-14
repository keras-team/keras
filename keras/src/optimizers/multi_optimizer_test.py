import numpy as np

from keras.src import backend
from keras.src import callbacks
from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import testing


class MultiOptimizerTest(testing.TestCase):
    def test_optimizer_map_matching(self):
        with backend.name_scope("dense"):
            w_dense = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("conv"):
            w_conv = backend.Variable([[1.0]], name="kernel")
        w_other = backend.Variable([[1.0]], name="bias")

        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)
        default_opt = optimizers.RMSprop()

        # Test regex matching
        opt_map = optimizers.OptimizerMap(default_optimizer=default_opt)
        opt_map["^dense/.*"] = opt_adam
        opt_map["^conv/.*"] = opt_sgd

        self.assertEqual(opt_map(w_dense), opt_adam)
        self.assertEqual(opt_map(w_conv), opt_sgd)
        self.assertEqual(opt_map(w_other), default_opt)

        # Test conflict detection
        opt_map_conflict = optimizers.OptimizerMap(
            default_optimizer=default_opt
        )
        opt_map_conflict[".*kernel.*"] = opt_adam
        opt_map_conflict[".*dense.*"] = opt_sgd
        with self.assertRaises(ValueError):
            opt_map_conflict(w_dense)

    def test_multi_optimizer_initialization(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)
        default_opt = optimizers.RMSprop(learning_rate=1e-4)

        opt_map = optimizers.OptimizerMap(default_optimizer=default_opt)
        opt_map["^dense_1/.*"] = opt_adam
        opt_map["^dense_2/.*"] = opt_sgd

        multi_opt = optimizers.MultiOptimizer(opt_map)

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("other"):
            w3 = backend.Variable([[1.0]], name="kernel")

        multi_opt.build([w1, w2, w3])

        # Verify unique optimizers list
        self.assertEqual(multi_opt.num_optimizers, 3)
        self.assertEqual(multi_opt.get_optimizer(0), opt_adam)
        self.assertEqual(multi_opt.get_optimizer(1), opt_sgd)
        self.assertEqual(multi_opt.get_optimizer(2), default_opt)

    def test_learning_rate_get_set(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)
        default_opt = optimizers.RMSprop(learning_rate=1e-4)

        opt_map = optimizers.OptimizerMap(default_optimizer=default_opt)
        opt_map["^dense_1/.*"] = opt_adam
        opt_map["^dense_2/.*"] = opt_sgd
        multi_opt = optimizers.MultiOptimizer(opt_map)

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[1.0]], name="kernel")

        multi_opt.build([w1, w2])

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

        opt_map = optimizers.OptimizerMap(default_optimizer=default_opt)
        opt_map["^dense_1/.*"] = opt_adam
        multi_opt = optimizers.MultiOptimizer(opt_map)

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

        opt_map = optimizers.OptimizerMap(default_optimizer=default_opt)
        opt_map["^dense_1/.*"] = opt_sgd_1
        opt_map["^dense_2/.*"] = opt_sgd_2
        multi_opt = optimizers.MultiOptimizer(opt_map)

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

        opt_map = optimizers.OptimizerMap(default_optimizer=opt_sgd_2)
        opt_map["^dense_1/.*"] = opt_sgd_1
        multi_opt = optimizers.MultiOptimizer(opt_map)

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

        opt_map = optimizers.OptimizerMap(default_optimizer=opt_sgd)
        opt_map["^sequential/dense_1/.*"] = opt_adam
        multi_opt = optimizers.MultiOptimizer(opt_map)

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

    def test_serialization(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)
        default_opt = optimizers.RMSprop(learning_rate=1e-4)

        opt_map = optimizers.OptimizerMap(default_optimizer=default_opt)
        opt_map["^dense_1/.*"] = opt_adam
        opt_map["^dense_2/.*"] = opt_sgd
        multi_opt = optimizers.MultiOptimizer(opt_map)

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("other"):
            w3 = backend.Variable([[1.0]], name="kernel")

        multi_opt.build([w1, w2, w3])

        config = optimizers.serialize(multi_opt)
        reconstructed = optimizers.deserialize(config)

        reconstructed.build([w1, w2, w3])

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

        self.assertEqual(
            reconstructed._get_optimizer_for_variable(w1).__class__.__name__,
            "Adam",
        )

    def test_loss_scaling(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3, loss_scale_factor=3.0)
        opt_sgd = optimizers.SGD(learning_rate=1e-2, loss_scale_factor=4.0)
        default_opt = optimizers.RMSprop(
            learning_rate=1e-4, loss_scale_factor=8.0
        )

        opt_map = optimizers.OptimizerMap(default_optimizer=default_opt)
        opt_map["^dense_1/.*"] = opt_adam
        opt_map["^dense_2/.*"] = opt_sgd
        multi_opt = optimizers.MultiOptimizer(opt_map, loss_scale_factor=5.0)

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[1.0]], name="kernel")
        multi_opt.build([w1])

        loss = backend.convert_to_tensor(2.0)
        scaled_loss = multi_opt.scale_loss(loss)
        self.assertAllClose(scaled_loss, 10.0)

    def test_gradient_unscaling(self):
        with backend.name_scope("dense_1"):
            w = backend.Variable([[2.0, 2.0]], name="kernel")

        # SGD with learning rate 0.1
        opt_sgd = optimizers.SGD(learning_rate=0.1, loss_scale_factor=2.0)
        opt_map = optimizers.OptimizerMap(default_optimizer=opt_sgd)
        opt_map["^dense_1/.*"] = opt_sgd
        multi_opt = optimizers.MultiOptimizer(opt_map, loss_scale_factor=5.0)

        multi_opt.build([w])

        # Gradient is 10.0 (which is scaled by loss_scale_factor=5.0)
        # Expected unscaled gradient is 10.0 / 5.0 = 2.0
        # Expected weight update is:
        # 2.0 - lr * unscaled_grad = 2.0 - 0.1 * 2.0 = 1.8
        grads = [backend.convert_to_tensor([[10.0, 10.0]])]

        multi_opt.apply(grads, [w])

        self.assertAllClose(w.numpy(), [[1.8, 1.8]])

    def test_custom_callable_function(self):
        opt_adam = optimizers.Adam(learning_rate=1e-3)
        opt_sgd = optimizers.SGD(learning_rate=1e-2)

        def optimizer_fn(variable):
            if "dense_1" in getattr(variable, "path", ""):
                return opt_adam
            else:
                return opt_sgd

        multi_opt = optimizers.MultiOptimizer(optimizer_fn)

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[1.0]], name="kernel")

        multi_opt.build([w1, w2])

        self.assertAllClose(multi_opt.learning_rate, 1e-3)

        # Verify dynamic sub-optimizer discovery
        self.assertEqual(multi_opt.num_optimizers, 2)
        self.assertEqual(multi_opt.get_optimizer(0), opt_adam)
        self.assertEqual(multi_opt.get_optimizer(1), opt_sgd)

        # Verify gradient application via custom callable routing
        grads = [
            backend.convert_to_tensor([[1.0]]),
            backend.convert_to_tensor([[1.0]]),
        ]
        multi_opt.apply(grads, [w1, w2])

        # w1 updated by Adam, w2 updated by SGD(0.01) -> 1.0 - 0.01 = 0.99
        self.assertAllClose(w2.numpy(), [[0.99]])
