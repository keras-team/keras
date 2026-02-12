import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.optimizers.schedule_free_adamw import ScheduleFreeAdamW


class ScheduleFreeAdamWTest(testing.TestCase):
    def test_config(self):
        optimizer = ScheduleFreeAdamW(
            learning_rate=0.005,
            beta_1=0.95,
            beta_2=0.99,
            epsilon=1e-6,
            warmup_steps=100,
        )
        self.run_class_serialization_test(optimizer)

    def test_single_step(self):
        optimizer = ScheduleFreeAdamW(learning_rate=0.5)
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        vars = backend.Variable([1.0, 2.0, 3.0, 4.0])
        optimizer.apply_gradients(zip([grads], [vars]))
        # After one step, the parameters should have changed
        self.assertNotAllClose(vars, [1.0, 2.0, 3.0, 4.0], rtol=1e-4, atol=1e-4)

    def test_weight_decay(self):
        grads, var1, var2 = (
            ops.zeros(()),
            backend.Variable(2.0),
            backend.Variable(2.0, name="exclude"),
        )
        optimizer_1 = ScheduleFreeAdamW(learning_rate=1.0, weight_decay=0.004)
        optimizer_1.apply_gradients(zip([grads], [var1]))

        optimizer_2 = ScheduleFreeAdamW(learning_rate=1.0, weight_decay=0.004)
        optimizer_2.exclude_from_weight_decay(var_names=["exclude"])
        optimizer_2.apply_gradients(zip([grads, grads], [var1, var2]))

        # var2 should be unchanged since it's excluded from weight decay
        self.assertAlmostEqual(var2.numpy(), 2.0, decimal=6)

    def test_train_eval_mode_switching(self):
        """Test that train/eval mode switching works correctly."""
        optimizer = ScheduleFreeAdamW(learning_rate=0.1, beta_1=0.9)
        var = backend.Variable([1.0, 2.0, 3.0])
        grads = ops.array([0.1, 0.2, 0.3])

        # Initial state
        initial_values = var.numpy().copy()

        # Apply multiple gradient steps so x and z diverge
        for _ in range(5):
            optimizer.apply_gradients(zip([grads], [var]))

        # Values should have changed
        after_train_steps = var.numpy().copy()
        self.assertFalse(np.allclose(initial_values, after_train_steps))

        # Switch to eval mode
        optimizer.swap_to_eval()
        eval_values = var.numpy().copy()

        # Switch back to train mode
        optimizer.swap_to_train()
        train_values = var.numpy().copy()

        # After multiple steps, x and z should have diverged,
        # making y (train) different from x (eval)
        self.assertFalse(np.allclose(eval_values, train_values))

    def test_warmup(self):
        """Test that warmup affects the learning rate."""
        optimizer_no_warmup = ScheduleFreeAdamW(
            learning_rate=0.5, warmup_steps=0
        )
        optimizer_with_warmup = ScheduleFreeAdamW(
            learning_rate=0.5, warmup_steps=10
        )

        grads = ops.array([1.0, 1.0, 1.0])
        var1 = backend.Variable([1.0, 2.0, 3.0])
        var2 = backend.Variable([1.0, 2.0, 3.0])

        # Apply single gradient step
        optimizer_no_warmup.apply_gradients(zip([grads], [var1]))
        optimizer_with_warmup.apply_gradients(zip([grads], [var2]))

        # The optimizer with warmup should have made a smaller update
        # because effective lr = lr * (step / warmup_steps) = 0.5 * 0.1 = 0.05
        diff_no_warmup = np.abs(var1.numpy() - [1.0, 2.0, 3.0])
        diff_with_warmup = np.abs(var2.numpy() - [1.0, 2.0, 3.0])

        # With warmup, the update should be smaller
        self.assertTrue(np.all(diff_with_warmup < diff_no_warmup))

    def test_multiple_steps(self):
        """Test that the optimizer works over multiple steps."""
        optimizer = ScheduleFreeAdamW(learning_rate=0.01)
        var = backend.Variable([1.0, 2.0, 3.0])

        for _ in range(10):
            grads = ops.array([0.1, 0.1, 0.1])
            optimizer.apply_gradients(zip([grads], [var]))

        # Parameters should have decreased
        final_values = var.numpy()
        self.assertTrue(np.all(final_values < [1.0, 2.0, 3.0]))

    def test_train_eval_aliases(self):
        """Test that train() and eval() are aliases."""
        optimizer = ScheduleFreeAdamW(learning_rate=0.1)
        var = backend.Variable([1.0, 2.0])
        grads = ops.array([0.1, 0.1])
        optimizer.apply_gradients(zip([grads], [var]))

        # Test that swap methods work without error
        optimizer.eval()
        optimizer.train()

    @pytest.mark.requires_trainable_backend
    def test_with_model(self):
        """Test that the optimizer works with a Keras model."""
        model = keras.Sequential([keras.layers.Dense(10)])
        optimizer = ScheduleFreeAdamW(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss="mse")

        x = keras.ops.ones((4, 5))
        y = keras.ops.zeros((4, 10))

        # Training
        optimizer.swap_to_train()
        model.fit(x, y, epochs=2, verbose=0)

        # Evaluation
        optimizer.swap_to_eval()
        loss = model.evaluate(x, y, verbose=0)
        self.assertIsNotNone(loss)

    def test_clip_norm(self):
        optimizer = ScheduleFreeAdamW(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = ScheduleFreeAdamW(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])
