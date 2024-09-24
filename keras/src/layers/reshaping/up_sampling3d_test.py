import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import testing


class UpSampling3dTest(testing.TestCase):
    @parameterized.product(
        data_format=["channels_first", "channels_last"],
        length_dim1=[2, 3],
        length_dim2=[2],
        length_dim3=[3],
    )
    @pytest.mark.requires_trainable_backend
    def test_upsampling_3d(
        self, data_format, length_dim1, length_dim2, length_dim3
    ):
        num_samples = 2
        stack_size = 2
        input_len_dim1 = 10
        input_len_dim2 = 11
        input_len_dim3 = 12

        if data_format == "channels_first":
            inputs = np.random.rand(
                num_samples,
                stack_size,
                input_len_dim1,
                input_len_dim2,
                input_len_dim3,
            )
        else:
            inputs = np.random.rand(
                num_samples,
                input_len_dim1,
                input_len_dim2,
                input_len_dim3,
                stack_size,
            )

        # basic test
        if data_format == "channels_first":
            expected_output_shape = (2, 2, 20, 22, 24)
        else:
            expected_output_shape = (2, 20, 22, 24, 2)

        self.run_layer_test(
            layers.UpSampling3D,
            init_kwargs={"size": (2, 2, 2), "data_format": data_format},
            input_shape=inputs.shape,
            expected_output_shape=expected_output_shape,
            expected_output_dtype="float32",
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

        layer = layers.UpSampling3D(
            size=(length_dim1, length_dim2, length_dim3),
            data_format=data_format,
        )
        layer.build(inputs.shape)
        np_output = layer(inputs=backend.Variable(inputs))
        if data_format == "channels_first":
            assert np_output.shape[2] == length_dim1 * input_len_dim1
            assert np_output.shape[3] == length_dim2 * input_len_dim2
            assert np_output.shape[4] == length_dim3 * input_len_dim3
        else:  # tf
            assert np_output.shape[1] == length_dim1 * input_len_dim1
            assert np_output.shape[2] == length_dim2 * input_len_dim2
            assert np_output.shape[3] == length_dim3 * input_len_dim3

        # compare with numpy
        if data_format == "channels_first":
            expected_out = np.repeat(inputs, length_dim1, axis=2)
            expected_out = np.repeat(expected_out, length_dim2, axis=3)
            expected_out = np.repeat(expected_out, length_dim3, axis=4)
        else:  # tf
            expected_out = np.repeat(inputs, length_dim1, axis=1)
            expected_out = np.repeat(expected_out, length_dim2, axis=2)
            expected_out = np.repeat(expected_out, length_dim3, axis=3)

        self.assertAllClose(np_output, expected_out)

    def test_upsampling_3d_correctness(self):
        input_shape = (2, 1, 2, 1, 3)
        x = np.arange(np.prod(input_shape)).reshape(input_shape)
        expected_output = np.array(
            [
                [
                    [
                        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                        [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                        [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                    ],
                    [
                        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                        [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                        [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                    ],
                ],
                [
                    [
                        [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]],
                        [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]],
                        [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                        [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                    ],
                    [
                        [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]],
                        [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]],
                        [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                        [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                    ],
                ],
            ]
        )
        if backend.config.image_data_format() == "channels_first":
            expected_output = expected_output.transpose((0, 4, 1, 2, 3))
            x = x.transpose((0, 4, 1, 2, 3))
        self.assertAllClose(
            layers.UpSampling3D(size=(2, 2, 2))(x), expected_output
        )
