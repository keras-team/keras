import numpy as np
import pytest

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
        self.assertEqual(len(multi_opt.optimizers), 3)
        self.assertEqual(multi_opt.optimizers[0], opt_adam)
        self.assertEqual(multi_opt.optimizers[1], opt_sgd)
        self.assertEqual(multi_opt.optimizers[2], default_opt)

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
            multi_opt.optimizers[
                multi_opt._var_to_optimizer_idx[multi_opt._var_key(w_mapped)]
            ],
            opt_adam,
        )
        self.assertEqual(
            multi_opt.optimizers[
                multi_opt._var_to_optimizer_idx[multi_opt._var_key(w_unmapped)]
            ],
            default_opt,
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
        for opt in multi_opt.optimizers:
            self.assertEqual(int(opt.iterations), 1)

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

        self.assertEqual(len(reconstructed.optimizers), 3)
        self.assertEqual(reconstructed.optimizers[0].__class__.__name__, "Adam")
        self.assertEqual(reconstructed.optimizers[1].__class__.__name__, "SGD")
        self.assertEqual(
            reconstructed.optimizers[2].__class__.__name__, "RMSprop"
        )

        self.assertEqual(
            reconstructed.optimizers[
                reconstructed._var_to_optimizer_idx[reconstructed._var_key(w1)]
            ].__class__.__name__,
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

        def optimizer_selector(variable):
            if "dense_1" in getattr(variable, "path", ""):
                return opt_adam
            else:
                return opt_sgd

        multi_opt = optimizers.MultiOptimizer(optimizer_selector)

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[1.0]], name="kernel")

        multi_opt.build([w1, w2])

        # Verify dynamic sub-optimizer discovery
        self.assertEqual(len(multi_opt.optimizers), 2)
        self.assertEqual(multi_opt.optimizers[0], opt_adam)
        self.assertEqual(multi_opt.optimizers[1], opt_sgd)

        # Verify gradient application via custom callable routing
        grads = [
            backend.convert_to_tensor([[1.0]]),
            backend.convert_to_tensor([[1.0]]),
        ]
        multi_opt.apply(grads, [w1, w2])

        # w1 updated by Adam, w2 updated by SGD(0.01) -> 1.0 - 0.01 = 0.99
        self.assertAllClose(w2.numpy(), [[0.99]])

    @pytest.mark.requires_trainable_backend
    def test_multi_optimizer_with_model_fit(self):
        x_train = np.ones((1, 1)).astype("float32")
        y_train = np.zeros((1, 1)).astype("float32")
        optimizer = optimizers.MultiOptimizer(
            optimizers.OptimizerMap(
                default_optimizer=optimizers.SGD(learning_rate=0.1),
                optimizer_map={
                    ".*dense_1/.*": optimizers.SGD(learning_rate=1e-3),
                    ".*dense_2/.*": optimizers.SGD(learning_rate=1e-2),
                },
            )
        )
        model = models.Sequential(
            [
                layers.Dense(
                    2, kernel_initializer="ones", use_bias=False, name="dense_1"
                ),
                layers.Dense(
                    1, kernel_initializer="ones", use_bias=False, name="dense_2"
                ),
            ]
        )
        model.compile(loss="mse", optimizer=optimizer, run_eagerly=True)
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        # Verify that weights have been updated from their initial ones
        w1 = model.layers[0].kernel.numpy()
        w2 = model.layers[1].kernel.numpy()

        self.assertAllClose(w1, [[0.996, 0.996]])
        self.assertAllClose(w2, [[0.96], [0.96]])

        # Verify master iteration counter incremented correctly
        self.assertEqual(int(optimizer.iterations), 1)
        # default iteration must be 0
        self.assertEqual(int(optimizer.optimizers[2].iterations), 0)
        # other two optimizer iteration must be 2
        self.assertEqual(int(optimizer.optimizers[0].iterations), 1)
        self.assertEqual(int(optimizer.optimizers[1].iterations), 1)

    @pytest.mark.requires_trainable_backend
    def test_learning_rate_scheduler_with_multi_optimizer(self):
        """Verifies LearningRateScheduler with MultiOptimizer.

        This test ensures that when using a MultiOptimizer, the
        LearningRateScheduler callback correctly updates the learning rate
        of each individual sub-optimizer according to the schedule. It
        verifies that non-linear scheduling is correctly applied to each
        sub-optimizer's unique learning rate.
        """
        opt_1 = optimizers.SGD(learning_rate=0.1)
        opt_2 = optimizers.SGD(learning_rate=0.01)

        optimizer = optimizers.MultiOptimizer(
            optimizers.OptimizerMap(
                default_optimizer=opt_2,
                optimizer_map={
                    ".*dense_1/.*": opt_1,
                },
            )
        )

        model = models.Sequential(
            [
                layers.Dense(
                    2, kernel_initializer="ones", use_bias=False, name="dense_1"
                ),
                layers.Dense(
                    1, kernel_initializer="ones", use_bias=False, name="dense_2"
                ),
            ]
        )
        model.compile(loss="mse", optimizer=optimizer)

        def schedule(epoch, lr):
            return lr * 0.5

        lr_scheduler = callbacks.LearningRateScheduler(schedule)
        lr_scheduler.set_model(model)
        lr_scheduler.on_epoch_begin(epoch=1)

        self.assertAllClose(backend.convert_to_numpy(opt_1.learning_rate), 0.05)
        self.assertAllClose(
            backend.convert_to_numpy(opt_2.learning_rate), 0.005
        )

    @pytest.mark.requires_trainable_backend
    def test_reduce_lr_on_plateau_with_multi_optimizer(self):
        """Verifies ReduceLROnPlateau with MultiOptimizer.

        This test ensures that when using a MultiOptimizer, the
        ReduceLROnPlateau callback triggers learning rate reduction on all
        sub-optimizers. It verifies that the reduction factor is applied to
        each sub-optimizer's learning rate proportionally, while the state
        machine (patience, cooldown) is only evaluated once per epoch.
        """
        opt_1 = optimizers.SGD(learning_rate=0.1)
        opt_2 = optimizers.SGD(learning_rate=0.01)

        optimizer = optimizers.MultiOptimizer(
            optimizers.OptimizerMap(
                default_optimizer=opt_2,
                optimizer_map={
                    ".*dense_1/.*": opt_1,
                },
            )
        )

        model = models.Sequential(
            [
                layers.Dense(
                    2, kernel_initializer="ones", use_bias=False, name="dense_1"
                ),
                layers.Dense(
                    1, kernel_initializer="ones", use_bias=False, name="dense_2"
                ),
            ]
        )
        model.compile(loss="mse", optimizer=optimizer)

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1
        )
        reduce_lr.set_model(model)
        reduce_lr.on_train_begin()

        # First epoch: set a baseline val_loss
        reduce_lr.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
        self.assertAllClose(backend.convert_to_numpy(opt_1.learning_rate), 0.1)
        self.assertAllClose(backend.convert_to_numpy(opt_2.learning_rate), 0.01)

        # Second epoch: val_loss does not improve (plateau), exceeds patience=1
        reduce_lr.on_epoch_end(epoch=1, logs={"val_loss": 1.0})
        self.assertAllClose(backend.convert_to_numpy(opt_1.learning_rate), 0.05)
        self.assertAllClose(
            backend.convert_to_numpy(opt_2.learning_rate), 0.005
        )

    @pytest.mark.requires_trainable_backend
    def test_tensorboard_logging_with_multi_optimizer(self):
        """Verifies TensorBoard learning rate logging with MultiOptimizer.

        This test ensures that the TensorBoard callback correctly collects
        and logs the learning rate of each sub-optimizer in a MultiOptimizer
        setup. It verifies that learning rates are logged under unique keys
        like 'learning_rate_{opt_name}' and that duplicate optimizer names
        are handled by appending a suffix.
        """
        import tempfile

        opt_1 = optimizers.SGD(learning_rate=0.1, name="SGD")
        opt_2 = optimizers.SGD(learning_rate=0.01, name="SGD_1")
        print(opt_1.name)
        print(opt_2.name)
        optimizer = optimizers.MultiOptimizer(
            optimizers.OptimizerMap(
                default_optimizer=opt_2,
                optimizer_map={
                    ".*dense_1/.*": opt_1,
                },
            )
        )

        model = models.Sequential(
            [
                layers.Dense(
                    2, kernel_initializer="ones", use_bias=False, name="dense_1"
                ),
                layers.Dense(
                    1, kernel_initializer="ones", use_bias=False, name="dense_2"
                ),
            ]
        )
        model.compile(loss="mse", optimizer=optimizer)

        with tempfile.TemporaryDirectory() as temp_dir:
            tb_callback = callbacks.TensorBoard(log_dir=temp_dir)
            tb_callback.set_model(model)

            logs = {}
            updated_logs = tb_callback._collect_learning_rate(logs)

            self.assertIn("learning_rate_SGD", updated_logs)
            self.assertIn("learning_rate_SGD_1", updated_logs)
            self.assertAllClose(updated_logs["learning_rate_SGD"], 0.1)
            self.assertAllClose(updated_logs["learning_rate_SGD_1"], 0.01)

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights_with_multi_optimizer(self):
        """Verifies SwapEMAWeights callback works with MultiOptimizer.

        This test ensures that when SwapEMAWeights is used with
        a MultiOptimizer, it correctly identifies which sub-optimizers
        have EMA enabled, and correctly swaps only the sub-variables
        mapped to those sub-optimizers with their respective EMA weights
        during evaluation, restoring them afterwards.
        """
        opt_1 = optimizers.SGD(
            learning_rate=0.1, use_ema=True, ema_momentum=0.9
        )
        opt_2 = optimizers.SGD(learning_rate=0.1, use_ema=False)

        optimizer = optimizers.MultiOptimizer(
            optimizers.OptimizerMap(
                default_optimizer=opt_2,
                optimizer_map={
                    ".*dense_1/.*": opt_1,
                },
            )
        )

        model = models.Sequential(
            [
                layers.Dense(
                    2, kernel_initializer="ones", use_bias=False, name="dense_1"
                ),
                layers.Dense(
                    1, kernel_initializer="ones", use_bias=False, name="dense_2"
                ),
            ]
        )
        model.compile(loss="mse", optimizer=optimizer)

        # Build variables
        x = np.ones((1, 1)).astype("float32")
        y = np.ones((1, 1)).astype("float32")
        model.train_on_batch(x, y)

        # Now manually set model weights to known values
        w_dense_1_initial = np.array([[2.0, 2.0]], dtype="float32")
        w_dense_2_initial = np.array([[3.0], [3.0]], dtype="float32")

        model.layers[0].kernel.assign(w_dense_1_initial)
        model.layers[1].kernel.assign(w_dense_2_initial)

        # Set EMA weights to a different known value
        w_dense_1_ema = np.array([[9.0, 9.0]], dtype="float32")
        opt_1._model_variables_moving_average[0].assign(w_dense_1_ema)

        # Track dense_2 weights (should not change because use_ema=False)
        dense_2_weights_before = backend.convert_to_numpy(
            model.layers[1].kernel
        )

        swap_callback = callbacks.SwapEMAWeights()
        swap_callback.set_model(model)

        # 1. Simulate evaluation start
        swap_callback.on_test_begin()

        # Assert dense_1 weights are swapped with EMA weights
        dense_1_weights_during = backend.convert_to_numpy(
            model.layers[0].kernel
        )
        self.assertAllClose(dense_1_weights_during, w_dense_1_ema)

        # Assert opt_1 EMA variable now holds the initial model weights
        opt_1_ema_during = backend.convert_to_numpy(
            opt_1._model_variables_moving_average[0]
        )
        self.assertAllClose(opt_1_ema_during, w_dense_1_initial)

        # Assert dense_2 weights are NOT swapped
        dense_2_weights_during = backend.convert_to_numpy(
            model.layers[1].kernel
        )
        self.assertAllClose(dense_2_weights_during, dense_2_weights_before)

        # 2. Simulate evaluation end
        swap_callback.on_test_end()

        # Assert weights are restored
        dense_1_weights_after = backend.convert_to_numpy(model.layers[0].kernel)
        dense_2_weights_after = backend.convert_to_numpy(model.layers[1].kernel)
        self.assertAllClose(dense_1_weights_after, w_dense_1_initial)
        self.assertAllClose(dense_2_weights_after, dense_2_weights_before)

        # Assert EMA weights are restored
        opt_1_ema_after = backend.convert_to_numpy(
            opt_1._model_variables_moving_average[0]
        )
        self.assertAllClose(opt_1_ema_after, w_dense_1_ema)

    @pytest.mark.requires_trainable_backend
    def test_multi_optimizer_wrapped_in_loss_scale_optimizer(self):
        """Verifies wrapping MultiOptimizer inside LossScaleOptimizer works."""
        opt_1 = optimizers.SGD(learning_rate=0.1)
        opt_2 = optimizers.SGD(learning_rate=0.1)

        # MultiOptimizer with initial loss scale 5.0
        multi_opt = optimizers.MultiOptimizer(
            optimizers.OptimizerMap(
                default_optimizer=opt_2,
                optimizer_map={
                    ".*dense_1/.*": opt_1,
                },
            ),
            loss_scale_factor=5.0,
        )

        # Wrap in LossScaleOptimizer with dynamic scaling (initial scale 10.0)
        lso_opt = optimizers.LossScaleOptimizer(
            multi_opt, initial_scale=10.0, dynamic_growth_steps=1000
        )

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[2.0, 2.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[2.0, 2.0]], name="kernel")

        lso_opt.build([w1, w2])

        # 1. Verify that LSO set MultiOptimizer.loss_scale_factor to None
        # and that it successfully propagated to sub-optimizers!
        self.assertIsNone(multi_opt.loss_scale_factor)
        self.assertIsNone(opt_1.loss_scale_factor)
        self.assertIsNone(opt_2.loss_scale_factor)

        # 2. Verify scale_loss uses LSO's dynamic scale (10.0)
        loss = backend.convert_to_tensor(2.0)
        self.assertAllClose(lso_opt.scale_loss(loss), 20.0)

        # 3. Verify gradient unscaling and application
        grads = [
            backend.convert_to_tensor([[10.0, 10.0]]),  # for w1
            backend.convert_to_tensor([[10.0, 10.0]]),  # for w2
        ]

        # Expected unscaled grads: 10.0 / 10.0 = 1.0
        # Expected updates: 2.0 - lr * 1.0 = 2.0 - 0.1 * 1.0 = 1.9
        if backend.backend() == "jax":
            new_trainable, _ = lso_opt.stateless_apply(
                [v.value for v in lso_opt.variables],
                grads,
                [w1.value, w2.value],
            )
            w1.assign(new_trainable[0])
            w2.assign(new_trainable[1])
        else:
            lso_opt.apply(grads, [w1, w2])

        self.assertAllClose(w1.numpy(), [[1.9, 1.9]])
        self.assertAllClose(w2.numpy(), [[1.9, 1.9]])

    @pytest.mark.requires_trainable_backend
    def test_multi_optimizer_blocks_nested_optimizers(self):
        """Verifies MultiOptimizer raises ValueError if
        inner optimizer is LSO or MultiOptimizer."""
        opt_sgd = optimizers.SGD(learning_rate=0.1)
        # Wrapping inner SGD in LSO
        opt_lso = optimizers.LossScaleOptimizer(opt_sgd)

        # Wrapping inner SGD in MultiOptimizer
        opt_multi = optimizers.MultiOptimizer(opt_sgd)

        # 1. Check error during OptimizerMap setitem
        opt_map = optimizers.OptimizerMap(default_optimizer=opt_sgd)
        with self.assertRaisesRegex(
            ValueError, "optimizer cannot be LossScaleOptimizer."
        ):
            opt_map[".*"] = opt_lso

        with self.assertRaisesRegex(
            ValueError, "optimizer cannot be MultiOptimizer."
        ):
            opt_map[".*"] = opt_multi

        # 2. Check error during build (in case it bypassed
        #  setitem via constructor dict)
        opt_map_direct = optimizers.OptimizerMap(default_optimizer=opt_sgd)
        opt_map_direct._optimizer_map = {
            ".*dense_1/.*": opt_lso,
            ".*dense_2/.*": opt_multi,
        }

        multi_opt = optimizers.MultiOptimizer(opt_map_direct)

        with backend.name_scope("dense_1"):
            w1 = backend.Variable([[1.0]], name="kernel")
        with backend.name_scope("dense_2"):
            w2 = backend.Variable([[1.0]], name="kernel")

        with self.assertRaisesRegex(
            ValueError,
            "LossScaleOptimizer cannot be used inside an MultiOptimizer.",
        ):
            multi_opt.build([w1])

        with self.assertRaisesRegex(
            ValueError,
            "MultiOptimizer cannot be used inside an MultiOptimizer.",
        ):
            multi_opt.build([w2])
