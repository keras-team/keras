# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Keras initializers."""

import warnings

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import backend
from keras import initializers
from keras import models
from keras.engine import input_layer
from keras.layers import core
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

RANDOM_INITIALIZERS = [
    initializers.RandomUniformV2,
    initializers.RandomNormalV2,
    initializers.OrthogonalV2,
    # TODO(scottzhu): Enable this after the forward compat period expires for
    # TruncatedNormalV2
    # initializers.TruncatedNormalV2,
    initializers.VarianceScalingV2,
    initializers.LecunUniformV2,
    initializers.LecunNormalV2,
    initializers.GlorotUniformV2,
    initializers.GlorotNormalV2,
    initializers.HeNormalV2,
    initializers.HeUniformV2,
]


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
      shape: Integer shape tuple or TF tensor shape.

    Returns:
      A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class KerasInitializersTest(tf.test.TestCase, parameterized.TestCase):
    def _runner(
        self,
        init,
        shape,
    ):
        # The global seed is set so that we can get the same random streams
        # between eager and graph mode when stateful op is used.
        tf.random.set_seed(1337)
        variable = backend.variable(init(shape))
        output = backend.get_value(variable)
        # Test serialization (assumes deterministic behavior).
        config = init.get_config()
        reconstructed_init = init.__class__.from_config(config)

        tf.random.set_seed(1337)
        variable = backend.variable(reconstructed_init(shape))
        output_2 = backend.get_value(variable)
        self.assertAllClose(output, output_2, atol=1e-4)

    def test_uniform(self):
        tensor_shape = (3, 2, 3)
        with self.cached_session():
            self._runner(
                initializers.RandomUniformV2(minval=-1, maxval=1, seed=124),
                tensor_shape,
            )

    def test_normal(self):
        tensor_shape = (8, 12, 99)
        with self.cached_session():
            self._runner(
                initializers.RandomNormalV2(mean=0, stddev=1, seed=153),
                tensor_shape,
            )

    def test_truncated_normal(self):
        tensor_shape = (12, 99, 7)
        with self.cached_session():
            self._runner(
                initializers.TruncatedNormalV2(mean=0, stddev=1, seed=126),
                tensor_shape,
            )

    def test_constant(self):
        tensor_shape = (5, 6, 4)
        with self.cached_session():
            self._runner(initializers.ConstantV2(2.0), tensor_shape)

    def test_lecun_uniform(self):
        tensor_shape = (5, 6, 4, 2)
        with self.cached_session():
            self._runner(initializers.LecunUniformV2(seed=123), tensor_shape)

    def test_glorot_uniform(self):
        tensor_shape = (5, 6, 4, 2)
        with self.cached_session():
            self._runner(initializers.GlorotUniformV2(seed=123), tensor_shape)

    def test_he_uniform(self):
        tensor_shape = (5, 6, 4, 2)
        with self.cached_session():
            self._runner(initializers.HeUniformV2(seed=123), tensor_shape)

    def test_lecun_normal(self):
        tensor_shape = (5, 6, 4, 2)
        with self.cached_session():
            self._runner(initializers.LecunNormalV2(seed=123), tensor_shape)

    def test_glorot_normal(self):
        tensor_shape = (5, 6, 4, 2)
        with self.cached_session():
            self._runner(initializers.GlorotNormalV2(seed=123), tensor_shape)

    def test_he_normal(self):
        tensor_shape = (5, 6, 4, 2)
        with self.cached_session():
            self._runner(initializers.HeNormalV2(seed=123), tensor_shape)

    def test_orthogonal(self):
        tensor_shape = (20, 20)
        with self.cached_session():
            self._runner(initializers.OrthogonalV2(seed=123), tensor_shape)

    def test_identity(self):
        with self.cached_session():
            tensor_shape = (3, 4, 5)
            with self.assertRaises(ValueError):
                self._runner(initializers.IdentityV2(), tensor_shape)

            tensor_shape = (3, 3)
            self._runner(initializers.IdentityV2(), tensor_shape)

    def test_zero(self):
        tensor_shape = (4, 5)
        with self.cached_session():
            self._runner(initializers.ZerosV2(), tensor_shape)

    def test_one(self):
        tensor_shape = (4, 5)
        with self.cached_session():
            self._runner(initializers.OnesV2(), tensor_shape)

    def test_default_random_uniform(self):
        ru = initializers.get("uniform")
        self.assertEqual(ru.minval, -0.05)
        self.assertEqual(ru.maxval, 0.05)

    def test_default_random_normal(self):
        rn = initializers.get("normal")
        self.assertEqual(rn.mean, 0.0)
        self.assertEqual(rn.stddev, 0.05)

    def test_default_truncated_normal(self):
        tn = initializers.get("truncated_normal")
        self.assertEqual(tn.mean, 0.0)
        self.assertEqual(tn.stddev, 0.05)

    def test_custom_initializer_saving(self):
        def my_initializer(shape, dtype=None):
            return tf.ones(shape, dtype=dtype)

        inputs = input_layer.Input((10,))
        outputs = core.Dense(1, kernel_initializer=my_initializer)(inputs)
        model = models.Model(inputs, outputs)
        model2 = model.from_config(
            model.get_config(),
            custom_objects={"my_initializer": my_initializer},
        )
        self.assertEqual(model2.layers[1].kernel_initializer, my_initializer)

    @test_utils.run_v2_only
    def test_load_external_variance_scaling_v2(self):
        external_serialized_json = {
            "class_name": "VarianceScaling",
            "config": {
                "distribution": "normal",
                "mode": "fan_avg",
                "scale": 1.0,
                "seed": None,
            },
        }
        initializer = initializers.deserialize(external_serialized_json)
        self.assertEqual(initializer.distribution, "truncated_normal")

    @parameterized.named_parameters(
        ("Zeros", initializers.ZerosV2, {}),
        ("Ones", initializers.OnesV2, {}),
        ("Constant", initializers.ConstantV2, {}),
        ("RandomUniform", initializers.RandomUniformV2, {}),
        ("RandomUniform_seeded", initializers.RandomUniformV2, {"seed": 123}),
        ("RandomNormal", initializers.RandomNormalV2, {}),
        ("RandomNormal_seeded", initializers.RandomNormalV2, {"seed": 123}),
        # TODO(scottzhu): Enable these tests after the forward compat period
        # expires for TruncatedNormalV2.
        # ("TruncatedNormal", initializers.TruncatedNormalV2, {}),
        # (
        #     "TruncatedNormal_seeded",
        #     initializers.TruncatedNormalV2,
        #     {"seed": 123},
        # ),
        ("LecunUniform", initializers.LecunUniformV2, {}),
        ("LecunUniform_seeded", initializers.LecunUniformV2, {"seed": 123}),
        ("GlorotUniform", initializers.GlorotUniformV2, {}),
        ("GlorotUniform_seeded", initializers.GlorotUniformV2, {"seed": 123}),
        ("HeUniform", initializers.HeUniformV2, {}),
        ("HeUniform_seeded", initializers.HeUniformV2, {"seed": 123}),
    )
    def test_partition(self, initializer_cls, kwargs):
        with self.cached_session():
            initializer = initializer_cls(**kwargs)
            result = initializer(
                shape=(4, 2), partition_shape=(2, 2), partition_offset=(0, 0)
            )
            self.assertEqual(result.shape, (2, 2))

            if hasattr(initializer, "seed"):
                # Make sure the result are different when the partition_shape is
                # same, but partition_offset is different, for random related
                # initializers.
                result_2 = initializer(
                    shape=(4, 2),
                    partition_shape=(2, 2),
                    partition_offset=(1, 0),
                )
                self.assertNotAllClose(result, result_2)

                # Make sure initializer produce same result when provide same
                # partition offset.
                result_3 = initializer(
                    shape=(4, 2),
                    partition_shape=(2, 2),
                    partition_offset=(1, 0),
                )
                self.assertAllClose(result_2, result_3)

    @parameterized.named_parameters(
        ("Orthogonal", initializers.OrthogonalV2),
        ("Identity", initializers.IdentityV2),
    )
    def test_partition_unsupported(self, initializer_cls):
        with self.assertRaisesRegex(
            ValueError,
            "initializer doesn't support partition-related arguments",
        ):
            initializer_cls()(
                shape=(4, 2), partition_shape=(2, 2), partition_offset=(0, 0)
            )

    @parameterized.parameters(RANDOM_INITIALIZERS)
    def test_stateless(self, initializer_cl):
        with self.cached_session():
            initializer = initializer_cl()
            output1 = initializer(shape=[2, 3])
            output2 = initializer(shape=[2, 3])
            initializer2 = initializer_cl()
            output3 = initializer2(shape=[2, 3])
            output4 = initializer2(shape=[2, 3])

            self.assertAllClose(output1, output2)
            self.assertAllClose(output3, output4)
            self.assertNotAllClose(output1, output3)

            with warnings.catch_warnings(record=True) as w:
                initializer(shape=[2, 3])
                self.assertLen(w, 1)
                self.assertIn("being called multiple times", str(w[0].message))

    @parameterized.parameters(RANDOM_INITIALIZERS)
    def test_seed_stateless(self, initializer_cl):
        with self.cached_session():
            seed = 1337
            initializer = initializer_cl(seed=seed)
            output1 = initializer(shape=[2, 3])
            output2 = initializer(shape=[2, 3])
            initializer2 = initializer_cl(seed=seed)
            output3 = initializer2(shape=[2, 3])
            output4 = initializer2(shape=[2, 3])

            self.assertAllClose(output1, output2)
            self.assertAllClose(output3, output4)
            self.assertAllClose(output1, output3)

            # We don't raise warning for seeded initializer.
            with warnings.catch_warnings(record=True) as w:
                initializer(shape=[2, 3])
                self.assertEmpty(w)


if __name__ == "__main__":
    tf.test.main()
