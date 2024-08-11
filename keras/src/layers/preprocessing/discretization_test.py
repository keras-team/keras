import os

import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.saving import saving_api
from keras.src.testing.test_utils import named_product


class DiscretizationTest(testing.TestCase, parameterized.TestCase):
    def test_discretization_basics(self):
        self.run_layer_test(
            layers.Discretization,
            init_kwargs={
                "bin_boundaries": [0.0, 0.5, 1.0],
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_adapt_flow(self):
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        output = layer(np.array([[0.0, 0.1, 0.3]]))
        self.assertTrue(output.dtype, "int32")

    @parameterized.named_parameters(
        named_product(
            [
                {
                    "testcase_name": "int",
                    "output_mode": "int",
                    "input_array": [[-1.0, 0.0, 0.1, 0.8, 1.2]],
                    "expected_output": [[0, 1, 1, 2, 3]],
                },
                {
                    "testcase_name": "one_hot_rank_1",
                    "output_mode": "one_hot",
                    "input_array": [0.1, 0.8],
                    "expected_output": [[0, 1, 0, 0], [0, 0, 1, 0]],
                },
                {
                    "testcase_name": "multi_hot_rank_2",
                    "output_mode": "multi_hot",
                    "input_array": [[0.1, 0.8]],
                    "expected_output": [[0, 1, 1, 0]],
                },
                {
                    "testcase_name": "one_hot_rank_3",
                    "output_mode": "one_hot",
                    "input_array": [[[0.15, 0.75], [0.85, 0.45]]],
                    "expected_output": [
                        [
                            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        ]
                    ],
                },
                {
                    "testcase_name": "multi_hot_rank_3",
                    "output_mode": "multi_hot",
                    "input_array": [[[0.15, 0.75], [0.85, 0.45]]],
                    "expected_output": [
                        [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
                    ],
                },
                {
                    "testcase_name": "count",
                    "output_mode": "count",
                    "input_array": [[0.1, 0.8, 0.9]],
                    "expected_output": [[0, 1, 2, 0]],
                },
            ],
            sparse=(
                [True, False] if backend.SUPPORTS_SPARSE_TENSORS else [False]
            ),
        )
    )
    def test_correctness(
        self, output_mode, input_array, expected_output, sparse
    ):
        if output_mode == "int" and sparse:
            pytest.skip("sparse=True cannot be combined with output_mode=int")

        input_array = np.array(input_array)
        expected_output = np.array(expected_output)

        layer = layers.Discretization(
            bin_boundaries=[0.0, 0.5, 1.0],
            output_mode=output_mode,
            sparse=sparse,
        )
        output = layer(input_array)
        self.assertSparse(output, sparse)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, expected_output)

    def test_tf_data_compatibility(self):
        # With fixed bins
        layer = layers.Discretization(
            bin_boundaries=[0.0, 0.35, 0.5, 1.0], dtype="float32"
        )
        x = np.array([[-1.0, 0.0, 0.1, 0.2, 0.4, 0.5, 1.0, 1.2, 0.98]])
        self.assertAllClose(layer(x), np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))
        ds = tf_data.Dataset.from_tensor_slices(x).batch(1).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))

        # With adapt flow
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        x = np.array([[0.0, 0.1, 0.3]])
        ds = tf_data.Dataset.from_tensor_slices(x).batch(1).map(layer)
        for output in ds.take(1):
            output.numpy()

    def test_saving(self):
        # With fixed bins
        layer = layers.Discretization(bin_boundaries=[0.0, 0.35, 0.5, 1.0])
        model = models.Sequential(
            [
                layers.Input((2,)),
                layer,
            ]
        )
        fpath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(fpath)
        model = saving_api.load_model(fpath)
        x = np.array([[-1.0, 0.0, 0.1, 0.2, 0.4, 0.5, 1.0, 1.2, 0.98]])
        self.assertAllClose(layer(x), np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))

        # With adapt flow
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        ref_input = np.random.random((1, 2))
        ref_output = layer(ref_input)
        model = models.Sequential(
            [
                layers.Input((2,)),
                layer,
            ]
        )
        fpath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(fpath)
        model = saving_api.load_model(fpath)
        self.assertAllClose(layer(ref_input), ref_output)
