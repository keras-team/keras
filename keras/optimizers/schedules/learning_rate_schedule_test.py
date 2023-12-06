"""Tests for learning rate schedule API."""

import math

import numpy as np
import pytest

from keras import backend
from keras import layers
from keras import optimizers
from keras import testing
from keras.models import Sequential
from keras.optimizers import schedules


class TestFitLRSchedulesFlow(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_fit_lr_correctness(self):
        model = Sequential(
            [
                layers.Dense(
                    2, kernel_initializer="ones", bias_initializer="ones"
                )
            ]
        )
        optimizer = optimizers.Adam(
            learning_rate=schedules.ExponentialDecay(
                initial_learning_rate=0.05, decay_steps=1, decay_rate=0.9
            )
        )
        self.assertEqual(len(optimizer.variables), 1)
        self.assertEqual(optimizer.variables[0], 0)

        model.compile(optimizer=optimizer, loss="mse")
        x = np.arange(32).reshape((16, 2))
        y = np.arange(32).reshape((16, 2))
        history = model.fit(x, y, epochs=3, batch_size=4, shuffle=False)
        self.assertEqual(optimizer.variables[0], 4 * 3)
        self.assertAllClose(
            history.history["loss"],
            [230.79457092285156, 128.30319213867188, 79.33648681640625],
            rtol=5e-5,
        )


class ExponentialDecayTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            schedules.ExponentialDecay(
                initial_learning_rate=0.05,
                decay_steps=10,
                decay_rate=0.96,
                staircase=True,
                name="my_ed",
            )
        )

    def test_continuous(self):
        step = 5
        decayed_lr = schedules.ExponentialDecay(0.05, 10, 0.96)
        expected = 0.05 * 0.96 ** (5.0 / 10.0)
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_staircase(self):
        step = backend.Variable(1.0)
        decayed_lr = schedules.ExponentialDecay(0.1, 3, 0.96, staircase=True)

        # No change to learning rate due to staircase
        expected = 0.1
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

        expected = 0.1
        step.assign(2)
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

        # Decayed learning rate
        expected = 0.1 * 0.96 ** (100 // 3)
        step.assign(100)
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_variables(self):
        step = backend.Variable(1.0)
        decayed_lr = schedules.ExponentialDecay(0.1, 3, 0.96, staircase=True)

        # No change to learning rate
        step.assign(1)
        self.assertAllClose(decayed_lr(step), 0.1, 1e-6)
        step.assign(2)
        self.assertAllClose(decayed_lr(step), 0.1, 1e-6)
        # Decayed learning rate
        step.assign(100)
        expected = 0.1 * 0.96 ** (100 // 3)
        self.assertAllClose(decayed_lr(step), expected, 1e-6)


class PiecewiseConstantDecayTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            schedules.PiecewiseConstantDecay(
                boundaries=[10, 20], values=[1, 2, 3], name="my_pcd"
            )
        )

    def test_piecewise_values(self):
        x = backend.Variable(-999.0)
        decayed_lr = schedules.PiecewiseConstantDecay(
            [100, 110, 120], [1.0, 0.1, 0.01, 0.001]
        )

        self.assertAllClose(decayed_lr(x), 1.0, 1e-6)
        x.assign(100)
        self.assertAllClose(decayed_lr(x), 1.0, 1e-6)
        x.assign(105)
        self.assertAllClose(decayed_lr(x), 0.1, 1e-6)
        x.assign(110)
        self.assertAllClose(decayed_lr(x), 0.1, 1e-6)
        x.assign(120)
        self.assertAllClose(decayed_lr(x), 0.01, 1e-6)
        x.assign(999)
        self.assertAllClose(decayed_lr(x), 0.001, 1e-6)

    def test_boundary_values(self):
        # Test casting boundaries from int32 to int64.
        x_int64 = backend.Variable(0, dtype="int64", trainable=False)
        boundaries, values = [1, 2, 3], [0.4, 0.5, 0.6, 0.7]
        decayed_lr = schedules.PiecewiseConstantDecay(boundaries, values)

        self.assertAllClose(decayed_lr(x_int64), 0.4, 1e-6)
        x_int64.assign(1)
        self.assertAllClose(decayed_lr(x_int64), 0.4, 1e-6)
        x_int64.assign(2)
        self.assertAllClose(decayed_lr(x_int64), 0.5, 1e-6)
        x_int64.assign(3)
        self.assertAllClose(decayed_lr(x_int64), 0.6, 1e-6)
        x_int64.assign(4)
        self.assertAllClose(decayed_lr(x_int64), 0.7, 1e-6)


class LinearDecayTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            schedules.PolynomialDecay(
                initial_learning_rate=0.1,
                decay_steps=100,
                end_learning_rate=0.005,
                power=1.0,
                cycle=False,
                name="my_ld",
            )
        )

    def test_halfway(self):
        step = 5
        lr = 0.05
        end_lr = 0.0
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr)
        expected = lr * 0.5
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_end(self):
        step = 10
        lr = 0.05
        end_lr = 0.001
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr)
        expected = end_lr
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_halfway_with_end(self):
        step = 5
        lr = 0.05
        end_lr = 0.001
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr)
        expected = (lr + end_lr) * 0.5
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_beyond_end(self):
        step = 15
        lr = 0.05
        end_lr = 0.001
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr)
        expected = end_lr
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_beyond_end_with_cycle(self):
        step = 15
        lr = 0.05
        end_lr = 0.001
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr, cycle=True)
        expected = (lr - end_lr) * 0.25 + end_lr
        self.assertAllClose(decayed_lr(step), expected, 1e-6)


class SqrtDecayTest(testing.TestCase):
    def test_halfway(self):
        step = 5
        lr = 0.05
        end_lr = 0.0
        power = 0.5
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr, power=power)
        expected = lr * 0.5**power
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_end(self):
        step = 10
        lr = 0.05
        end_lr = 0.001
        power = 0.5
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr, power=power)
        expected = end_lr
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_halfway_with_end(self):
        step = 5
        lr = 0.05
        end_lr = 0.001
        power = 0.5
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr, power=power)
        expected = (lr - end_lr) * 0.5**power + end_lr
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_beyond_end(self):
        step = 15
        lr = 0.05
        end_lr = 0.001
        power = 0.5
        decayed_lr = schedules.PolynomialDecay(lr, 10, end_lr, power=power)
        expected = end_lr
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_beyond_end_with_cycle(self):
        step = 15
        lr = 0.05
        end_lr = 0.001
        power = 0.5
        decayed_lr = schedules.PolynomialDecay(
            lr, 10, end_lr, power=power, cycle=True
        )
        expected = (lr - end_lr) * 0.25**power + end_lr
        self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_begin_with_cycle(self):
        lr = 0.001
        decay_steps = 10
        step = 0
        decayed_lr = schedules.PolynomialDecay(lr, decay_steps, cycle=True)
        expected = lr
        self.assertAllClose(decayed_lr(step), expected, 1e-6)


class InverseTimeDecayTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            schedules.InverseTimeDecay(
                initial_learning_rate=0.05,
                decay_steps=10,
                decay_rate=0.96,
                staircase=True,
                name="my_itd",
            )
        )

    def test_decay(self):
        initial_lr = 0.1
        k = 10
        decay_rate = 0.96
        step = backend.Variable(0.0)
        decayed_lr = schedules.InverseTimeDecay(initial_lr, k, decay_rate)

        for i in range(k + 1):
            expected = initial_lr / (1 + i / k * decay_rate)
            self.assertAllClose(decayed_lr(step), expected, 1e-6)
            step.assign(step + 1)

    def test_staircase(self):
        initial_lr = 0.1
        k = 10
        decay_rate = 0.96
        step = backend.Variable(0.0)
        decayed_lr = schedules.InverseTimeDecay(
            initial_lr, k, decay_rate, staircase=True
        )

        for i in range(k + 1):
            expected = initial_lr / (1 + decay_rate * (i // k))
            self.assertAllClose(decayed_lr(step), expected, 1e-6)
            step.assign(step + 1)


class CosineDecayTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            schedules.CosineDecay(
                initial_learning_rate=0.05,
                decay_steps=10,
                alpha=0.1,
                warmup_target=0.2,
                warmup_steps=2,
                name="my_cd",
            )
        )

    def np_cosine_decay(self, step, decay_steps, alpha=0.0):
        step = min(step, decay_steps)
        completed_fraction = step / decay_steps
        decay = 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
        return (1.0 - alpha) * decay + alpha

    def test_decay(self):
        num_training_steps = 1000
        initial_lr = 1.0
        for step in range(0, 1500, 250):
            decayed_lr = schedules.CosineDecay(initial_lr, num_training_steps)
            expected = self.np_cosine_decay(step, num_training_steps)
            self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def linear_warmup(self, step, warmup_steps, initial_lr, target_lr):
        completed_fraction = step / warmup_steps
        total_delta = target_lr - initial_lr
        return completed_fraction * total_delta

    def test_warmup(self):
        warmup_steps = 1500
        initial_lr = 0.0
        target_lr = 10.0
        for step in range(0, 1500, 250):
            lr = schedules.CosineDecay(
                initial_lr,
                10,
                warmup_target=target_lr,
                warmup_steps=warmup_steps,
            )
            expected = self.linear_warmup(
                step, warmup_steps, initial_lr, target_lr
            )
            self.assertAllClose(lr(step), expected)

    def test_alpha(self):
        num_training_steps = 1000
        initial_lr = 1.0
        alpha = 0.1
        for step in range(0, 1500, 250):
            decayed_lr = schedules.CosineDecay(
                initial_lr, num_training_steps, alpha
            )
            expected = self.np_cosine_decay(step, num_training_steps, alpha)
            self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_float64(self):
        num_training_steps = 1000
        initial_lr = np.float64(1.0)
        for step in range(0, 1500, 250):
            decayed_lr = schedules.CosineDecay(initial_lr, num_training_steps)
            expected = self.np_cosine_decay(step, num_training_steps)
            self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_warmup_decay(self):
        warmup_steps = 2000
        decay_steps = 1000
        initial_lr = 0.0
        target_lr = 10.0
        for step in range(0, 3000, 250):
            lr = schedules.CosineDecay(
                initial_lr,
                decay_steps,
                warmup_target=target_lr,
                warmup_steps=warmup_steps,
            )
            if step < warmup_steps + 1:
                expected = self.linear_warmup(
                    step, warmup_steps, initial_lr, target_lr
                )
            else:
                expected = target_lr * self.np_cosine_decay(
                    step - warmup_steps, decay_steps
                )
            self.assertAllClose(lr(step), expected)


class CosineDecayRestartsTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            schedules.CosineDecayRestarts(
                initial_learning_rate=0.05,
                first_decay_steps=10,
                alpha=0.1,
                t_mul=3.0,
                m_mul=4.0,
                name="my_cdr",
            )
        )

    def np_cosine_decay_restarts(
        self, step, decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0
    ):
        fac = 1.0
        while step >= decay_steps:
            step -= decay_steps
            decay_steps *= t_mul
            fac *= m_mul

        completed_fraction = step / decay_steps
        decay = fac * 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
        return (1.0 - alpha) * decay + alpha

    def test_decay(self):
        num_training_steps = 1000
        initial_lr = 1.0
        for step in range(0, 1500, 250):
            decayed_lr = schedules.CosineDecayRestarts(
                initial_lr, num_training_steps
            )
            expected = self.np_cosine_decay_restarts(step, num_training_steps)
            self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_float64(self):
        num_training_steps = 1000
        initial_lr = np.float64(1.0)
        for step in range(0, 1500, 250):
            decayed_lr = schedules.CosineDecayRestarts(
                initial_lr, num_training_steps
            )
            expected = self.np_cosine_decay_restarts(step, num_training_steps)
            self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_alpha(self):
        num_training_steps = 1000
        initial_lr = 1.0
        alpha = 0.1
        for step in range(0, 1500, 250):
            decayed_lr = schedules.CosineDecayRestarts(
                initial_lr, num_training_steps, alpha=alpha
            )
            expected = self.np_cosine_decay_restarts(
                step, num_training_steps, alpha=alpha
            )
            self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_mmul(self):
        num_training_steps = 1000
        initial_lr = 1.0
        m_mul = 0.9
        for step in range(0, 1500, 250):
            decayed_lr = schedules.CosineDecayRestarts(
                initial_lr, num_training_steps, m_mul=m_mul
            )
            expected = self.np_cosine_decay_restarts(
                step, num_training_steps, m_mul=m_mul
            )
            self.assertAllClose(decayed_lr(step), expected, 1e-6)

    def test_tmul(self):
        num_training_steps = 1000
        initial_lr = 1.0
        t_mul = 1.0
        for step in range(0, 1500, 250):
            decayed_lr = schedules.CosineDecayRestarts(
                initial_lr, num_training_steps, t_mul=t_mul
            )
            expected = self.np_cosine_decay_restarts(
                step, num_training_steps, t_mul=t_mul
            )
            self.assertAllClose(decayed_lr(step), expected, 1e-6)
