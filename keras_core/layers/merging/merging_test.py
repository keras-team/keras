import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import operations as ops
from keras_core import testing


class MergingLayersTest(testing.TestCase):
    def test_add_basic(self):
        self.run_layer_test(
            layers.Add,
            init_kwargs={},
            input_shape=[(2, 3), (2, 3)],
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes.",
    )
    def test_add_correctness_dynamic(self):
        x1 = np.random.rand(2, 4, 5)
        x2 = np.random.rand(2, 4, 5)
        x3 = ops.convert_to_tensor(x1 + x2)

        input_1 = layers.Input(shape=(4, 5))
        input_2 = layers.Input(shape=(4, 5))
        add_layer = layers.Add()
        out = add_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 4, 5))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            add_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                add_layer.compute_mask(
                    [input_1, input_2],
                    [backend.Variable(x1), backend.Variable(x2)],
                )
            )
        )

    def test_add_correctness_static(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.convert_to_tensor(x1 + x2)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        add_layer = layers.Add()
        out = add_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, *shape))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            add_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                add_layer.compute_mask(
                    [input_1, input_2],
                    [backend.Variable(x1), backend.Variable(x2)],
                )
            )
        )

    def test_add_errors(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        add_layer = layers.Add()

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            add_layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            add_layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            add_layer.compute_mask([input_1, input_2], [None])

    def test_subtract_basic(self):
        self.run_layer_test(
            layers.Subtract,
            init_kwargs={},
            input_shape=[(2, 3), (2, 3)],
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes.",
    )
    def test_subtract_correctness_dynamic(self):
        x1 = np.random.rand(2, 4, 5)
        x2 = np.random.rand(2, 4, 5)
        x3 = ops.convert_to_tensor(x1 - x2)

        input_1 = layers.Input(shape=(4, 5))
        input_2 = layers.Input(shape=(4, 5))
        subtract_layer = layers.Subtract()
        out = subtract_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 4, 5))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            subtract_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                subtract_layer.compute_mask(
                    [input_1, input_2],
                    [backend.Variable(x1), backend.Variable(x2)],
                )
            )
        )

    def test_subtract_correctness_static(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.convert_to_tensor(x1 - x2)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        subtract_layer = layers.Subtract()
        out = subtract_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, *shape))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            subtract_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                subtract_layer.compute_mask(
                    [input_1, input_2],
                    [backend.Variable(x1), backend.Variable(x2)],
                )
            )
        )

    def test_subtract_errors(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        input_3 = layers.Input(shape=shape, batch_size=batch_size)
        subtract_layer = layers.Subtract()

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            subtract_layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            subtract_layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            subtract_layer.compute_mask([input_1, input_2], [None])
        with self.assertRaisesRegex(
            ValueError, "layer should be called on exactly 2 inputs"
        ):
            layers.Subtract()([input_1, input_2, input_3])
        with self.assertRaisesRegex(
            ValueError, "layer should be called on exactly 2 inputs"
        ):
            layers.Subtract()([input_1])
