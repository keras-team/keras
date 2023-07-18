import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import ops
from keras_core import testing


@pytest.mark.requires_trainable_backend
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
                backend.convert_to_numpy(
                    add_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
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
                backend.convert_to_numpy(
                    add_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
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
                backend.convert_to_numpy(
                    subtract_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
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
                backend.convert_to_numpy(
                    subtract_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
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

    def test_minimum_basic(self):
        self.run_layer_test(
            layers.Minimum,
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
    def test_minimum_correctness_dynamic(self):
        x1 = np.random.rand(2, 4, 5)
        x2 = np.random.rand(2, 4, 5)
        x3 = ops.minimum(x1, x2)

        input_1 = layers.Input(shape=(4, 5))
        input_2 = layers.Input(shape=(4, 5))
        merge_layer = layers.Minimum()
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 4, 5))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_minimum_correctness_static(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.minimum(x1, x2)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Minimum()
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, *shape))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_minimum_errors(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Minimum()

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            merge_layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            merge_layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            merge_layer.compute_mask([input_1, input_2], [None])

    def test_maximum_basic(self):
        self.run_layer_test(
            layers.Maximum,
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
    def test_maximum_correctness_dynamic(self):
        x1 = np.random.rand(2, 4, 5)
        x2 = np.random.rand(2, 4, 5)
        x3 = ops.maximum(x1, x2)

        input_1 = layers.Input(shape=(4, 5))
        input_2 = layers.Input(shape=(4, 5))
        merge_layer = layers.Maximum()
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 4, 5))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_maximum_correctness_static(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.maximum(x1, x2)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Maximum()
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, *shape))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_maximum_errors(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Maximum()

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            merge_layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            merge_layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            merge_layer.compute_mask([input_1, input_2], [None])

    def test_multiply_basic(self):
        self.run_layer_test(
            layers.Multiply,
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
    def test_multiply_correctness_dynamic(self):
        x1 = np.random.rand(2, 4, 5)
        x2 = np.random.rand(2, 4, 5)
        x3 = ops.convert_to_tensor(x1 * x2)

        input_1 = layers.Input(shape=(4, 5))
        input_2 = layers.Input(shape=(4, 5))
        merge_layer = layers.Multiply()
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 4, 5))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_multiply_correctness_static(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.convert_to_tensor(x1 * x2)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Multiply()
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, *shape))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_multiply_errors(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Multiply()

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            merge_layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            merge_layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            merge_layer.compute_mask([input_1, input_2], [None])

    def test_average_basic(self):
        self.run_layer_test(
            layers.Average,
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
    def test_average_correctness_dynamic(self):
        x1 = np.random.rand(2, 4, 5)
        x2 = np.random.rand(2, 4, 5)
        x3 = ops.average(np.array([x1, x2]), axis=0)

        input_1 = layers.Input(shape=(4, 5))
        input_2 = layers.Input(shape=(4, 5))
        merge_layer = layers.Average()
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 4, 5))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_average_correctness_static(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.average(np.array([x1, x2]), axis=0)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Average()
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, *shape))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_average_errors(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Average()

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            merge_layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            merge_layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            merge_layer.compute_mask([input_1, input_2], [None])

    def test_concatenate_basic(self):
        self.run_layer_test(
            layers.Concatenate,
            init_kwargs={"axis": 1},
            input_shape=[(2, 3), (2, 3)],
            expected_output_shape=(2, 6),
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
    def test_concatenate_correctness_dynamic(self):
        x1 = np.random.rand(2, 4, 5)
        x2 = np.random.rand(2, 4, 5)
        axis = 1

        x3 = ops.concatenate([x1, x2], axis=axis)

        input_1 = layers.Input(shape=(4, 5))
        input_2 = layers.Input(shape=(4, 5))
        merge_layer = layers.Concatenate(axis=axis)
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 8, 5))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [backend.Variable(x1), backend.Variable(x2)],
                    )
                )
            )
        )

    def test_concatenate_correctness_static(self):
        batch_size = 2
        shape = (4, 5)
        axis = 1
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.concatenate([x1, x2], axis=axis)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Concatenate(axis=axis)
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, 8, 5))
        self.assertAllClose(res, x3, atol=1e-4)
        self.assertIsNone(
            merge_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                backend.convert_to_numpy(
                    merge_layer.compute_mask(
                        [input_1, input_2],
                        [x1, x2],
                    )
                )
            )
        )

    def test_concatenate_errors(self):
        batch_size = 2
        shape = (4, 5)
        axis = 1
        x1 = np.random.rand(batch_size, *shape)

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Concatenate(axis=axis)

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            merge_layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            merge_layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            merge_layer.compute_mask([input_1, input_2], [None])

    def test_dot_basic(self):
        self.run_layer_test(
            layers.Dot,
            init_kwargs={"axes": -1},
            input_shape=[(4, 3), (4, 3)],
            expected_output_shape=(4, 1),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=None,
        )

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes.",
    )
    def test_dot_correctness_dynamic(self):
        x1 = np.random.rand(2, 4)
        x2 = np.random.rand(2, 4)
        axes = 1

        expected = np.zeros((2, 1))
        expected[0, 0] = np.dot(x1[0], x2[0])
        expected[1, 0] = np.dot(x1[1], x2[1])

        input_1 = layers.Input(shape=(4,))
        input_2 = layers.Input(shape=(4,))
        merge_layer = layers.Dot(axes=axes)
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 1))
        self.assertAllClose(res, expected, atol=1e-4)

    def test_dot_correctness_static(self):
        batch_size = 2
        shape = (4,)
        axes = 1

        x1 = np.random.rand(2, 4)
        x2 = np.random.rand(2, 4)
        expected = np.zeros((2, 1))
        expected[0, 0] = np.dot(x1[0], x2[0])
        expected[1, 0] = np.dot(x1[1], x2[1])

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        merge_layer = layers.Dot(axes=axes)
        out = merge_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (2, 1))
        self.assertAllClose(res, expected, atol=1e-4)
