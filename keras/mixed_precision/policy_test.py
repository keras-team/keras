# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests Policies."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.engine import base_layer_utils
from keras.mixed_precision import device_compatibility_check
from keras.mixed_precision import policy as mp_policy
from keras.optimizers.legacy import gradient_descent
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

# isort: off
from tensorflow.python.platform import tf_logging


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class PolicyTest(tf.test.TestCase, parameterized.TestCase):
    """Tests Policies."""

    @test_utils.enable_v2_dtype_behavior
    def test_dtype_attributes(self):
        for dtype in "int32", "bool", "float16", "float32":
            policy = mp_policy.Policy(dtype)
            self.assertEqual(policy.name, dtype)
            self.assertEqual(policy.compute_dtype, dtype)
            self.assertEqual(policy.variable_dtype, dtype)

        for dtype in "float16", "bfloat16":
            policy = mp_policy.Policy("mixed_" + dtype)
            self.assertEqual(policy.name, "mixed_" + dtype)
            self.assertEqual(policy.compute_dtype, dtype)
            self.assertEqual(policy.variable_dtype, "float32")

        policy = mp_policy.Policy("_infer")
        self.assertEqual(policy.compute_dtype, None)
        self.assertEqual(policy.variable_dtype, None)

    @test_utils.enable_v2_dtype_behavior
    def test_repr(self):
        # Test Policy repr
        for policy in (
            "float32",
            "int8",
            "mixed_float16",
            "mixed_bfloat16",
            "_infer",
        ):
            self.assertEqual(
                repr(mp_policy.Policy(policy)), f'<Policy "{policy}">'
            )

    @test_utils.enable_v2_dtype_behavior
    def test_policy_errors(self):
        # Test passing invalid strings

        with self.assertRaisesRegex(
            ValueError, "Cannot convert value abc to a mixed precision Policy."
        ):
            mp_policy.Policy("abc")

        # Test passing a DType
        with self.assertRaisesRegex(
            TypeError, "'name' must be a string, not a DType. "
        ):
            mp_policy.Policy(tf.float16)

        # Test passing a non-DType invalid type
        with self.assertRaisesRegex(
            TypeError, "'name' must be a string, but got: 5"
        ):
            mp_policy.Policy(5)

        # Test passing a now-removed policy ending in float32_vars
        with self.assertRaisesRegex(
            ValueError,
            "Policies ending in '_float32_vars' have been removed "
            "from TensorFlow. Please use the 'mixed_float16' or "
            "'mixed_bfloat16' policy instead. Got policy name: "
            "'infer_float32_vars'",
        ):
            mp_policy.Policy("infer_float32_vars")
        with self.assertRaisesRegex(
            ValueError,
            "Policies ending in '_float32_vars' have been removed "
            "from TensorFlow. Please use the 'mixed_float16' policy "
            "instead. Got policy name: 'float16_with_float32_vars'",
        ):
            mp_policy.Policy("float16_with_float32_vars")
        with self.assertRaisesRegex(
            ValueError,
            "Policies ending in '_float32_vars' have been removed "
            "from TensorFlow. Please use the 'mixed_bfloat16' policy "
            "instead. Got policy name: 'bfloat16_with_float32_vars'",
        ):
            mp_policy.Policy("bfloat16_with_float32_vars")
        with self.assertRaisesRegex(
            ValueError,
            "Policies ending in '_float32_vars' have been removed "
            "from TensorFlow. Got policy name: "
            "'int8_with_float32_vars'",
        ):
            mp_policy.Policy("int8_with_float32_vars")

    @test_utils.enable_v2_dtype_behavior
    def test_global_policy(self):
        if base_layer_utils.v2_dtype_behavior_enabled():
            default_policy = "float32"
        else:
            default_policy = "_infer"
        self.assertEqual(mp_policy.global_policy().name, default_policy)
        try:
            mp_policy.set_global_policy("mixed_float16")
            self.assertEqual(mp_policy.global_policy().name, "mixed_float16")
            # Policies are not associated with a graph
            with tf.Graph().as_default():
                self.assertEqual(
                    mp_policy.global_policy().name, "mixed_float16"
                )
            mp_policy.set_global_policy("_infer")
            self.assertEqual(mp_policy.global_policy().name, "_infer")
            policy = mp_policy.Policy("mixed_bfloat16")
            mp_policy.set_global_policy(policy)
            self.assertIs(mp_policy.global_policy(), policy)
        finally:
            mp_policy.set_global_policy(None)

    @test_utils.enable_v2_dtype_behavior
    def test_global_policy_dtype_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "set_global_policy can only be used to set the global policy to "
            'floating-point policies, such as "float32" and "mixed_float16", '
            "but got policy: int32",
        ):
            mp_policy.set_global_policy("int32")
        with self.assertRaisesRegex(
            ValueError,
            "set_global_policy can only be used to set the global policy to "
            'floating-point policies, such as "float32" and "mixed_float16", '
            "but got policy: complex64",
        ):
            mp_policy.set_global_policy(mp_policy.Policy("complex64"))

    @test_utils.enable_v2_dtype_behavior
    def test_device_compatibility_warning(self):
        if not tf.executing_eagerly():
            self.skipTest("Run in eager mode only.")

        device_compatibility_check._logged_compatibility_check = False
        with tf.compat.v1.test.mock.patch.object(
            tf_logging, "warning"
        ) as mock_warn:
            mp_policy.Policy("mixed_float16")
        if tf.config.list_physical_devices("GPU"):
            mock_warn.assert_not_called()
        else:
            self.assertRegex(
                mock_warn.call_args[0][0],
                r"Mixed precision compatibility check \(mixed_float16\): "
                r"WARNING.*",
            )

        if tf.config.list_physical_devices("GPU"):
            # Assert message is only logged once
            with tf.compat.v1.test.mock.patch.object(
                tf_logging, "warning"
            ) as mock_warn:
                mp_policy.Policy("mixed_float16")
            mock_warn.assert_not_called()

    @test_utils.enable_v2_dtype_behavior
    def test_policy_scope(self):
        if base_layer_utils.v2_dtype_behavior_enabled():
            default_policy = "float32"
        else:
            default_policy = "_infer"
        with mp_policy.policy_scope("mixed_float16"):
            self.assertEqual(mp_policy.global_policy().name, "mixed_float16")
            with mp_policy.policy_scope("_infer"):
                self.assertEqual(mp_policy.global_policy().name, "_infer")
            self.assertEqual(mp_policy.global_policy().name, "mixed_float16")
        self.assertEqual(mp_policy.global_policy().name, default_policy)

    @test_utils.enable_v2_dtype_behavior
    def test_config(self):
        for policy in (
            mp_policy.Policy("float16"),
            mp_policy.Policy("float32"),
            mp_policy.Policy("int16"),
            mp_policy.Policy("mixed_float16"),
            mp_policy.Policy("mixed_bfloat16"),
            mp_policy.Policy("_infer"),
        ):
            config = policy.get_config()
            new_policy = mp_policy.Policy.from_config(config)
            # Comparing strings is the easiest way to ensure the policies are
            # the same, as policy does not override the == operator.
            self.assertEqual(str(policy), str(new_policy))

    @test_utils.enable_v2_dtype_behavior
    def test_serialization(self):
        # Test policies that are equivalent to a single dtype
        for policy_name in "float16", "float32", "int8", "string", "bool":
            policy = mp_policy.Policy(policy_name)
            config = mp_policy.serialize(policy)
            self.assertEqual(config, policy_name)
            new_policy = mp_policy.deserialize(config)
            self.assertEqual(str(policy), str(new_policy))

        # Test "_infer" policy
        policy = mp_policy.Policy("_infer")
        config = mp_policy.serialize(policy)
        self.assertIsNone(config)
        new_policy = mp_policy.deserialize(config)
        self.assertEqual(str(policy), str(new_policy))

        class MyPolicy(mp_policy.Policy):
            pass

        # Test policies that are not equivalent to a single dtype
        for policy in (
            mp_policy.Policy("mixed_float16"),
            mp_policy.Policy("mixed_bfloat16"),
            MyPolicy("float32"),
        ):
            config = mp_policy.serialize(policy)
            if tf.__internal__.tf2.enabled():
                if policy.name == "float32":
                    self.assertEqual(
                        config,
                        {
                            "module": None,
                            "class_name": policy.__class__.__name__,
                            "config": {"name": policy.name},
                            "registered_name": "MyPolicy",
                        },
                    )
                else:
                    self.assertEqual(
                        config,
                        {
                            "module": "keras.mixed_precision",
                            "class_name": policy.__class__.__name__,
                            "config": {"name": policy.name},
                            "registered_name": None,
                        },
                    )
            else:
                self.assertEqual(
                    config,
                    {
                        "class_name": policy.__class__.__name__,
                        "config": {"name": policy.name},
                    },
                )
            new_policy = mp_policy.deserialize(
                config, custom_objects={"MyPolicy": MyPolicy}
            )
            self.assertEqual(str(policy), str(new_policy))

    @test_utils.enable_v2_dtype_behavior
    def test_error_if_graph_rewrite_enabled(self):
        try:
            tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
                gradient_descent.SGD(1.0)
            )
            with self.assertRaisesRegex(
                ValueError,
                'cannot be set to "mixed_float16", .* the mixed '
                "precision graph rewrite has already been enabled",
            ):
                mp_policy.set_global_policy("mixed_float16")
            with mp_policy.policy_scope("float64"):
                pass  # Non-mixed policies are allowed
        finally:
            tf.compat.v1.mixed_precision.disable_mixed_precision_graph_rewrite()

    @test_utils.disable_v2_dtype_behavior
    def test_v1_dtype_behavior(self):
        # Setting global policies are not allowed with V1 dtype behavior
        with self.assertRaisesRegex(
            ValueError, "global policy can only be set in TensorFlow 2"
        ):
            with mp_policy.policy_scope(mp_policy.Policy("_infer")):
                pass
        with self.assertRaisesRegex(
            ValueError, "global policy can only be set in TensorFlow 2"
        ):
            with mp_policy.policy_scope(mp_policy.Policy("float32")):
                pass
        with self.assertRaisesRegex(
            ValueError, "global policy can only be set in TensorFlow 2"
        ):
            with mp_policy.policy_scope(mp_policy.Policy("mixed_float16")):
                pass


if __name__ == "__main__":
    tf.test.main()
