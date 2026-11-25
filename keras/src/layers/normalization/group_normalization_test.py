import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src import regularizers
from keras.src import testing
from keras.src.backend.common import AutocastScope


class GroupNormalizationTest(testing.TestCase):
    @parameterized.named_parameters(("eager", False), ("graph", True))
    def test_input_gradient_scaling_near_constant(self, graph):
        if backend.backend() != "tensorflow":
            self.skipTest("Uses TensorFlow gradients.")
        import tensorflow as tf

        # The sum of normalized values has zero input gradient when gamma=1.
        # Near-constant inputs expose cancellation in that gradient.
        inputs = tf.constant(
            10.0
            + np.random.default_rng(0)
            .normal(scale=1e-3, size=(2, 16, 16, 32))
            .astype("float32")
        )
        layer = layers.GroupNormalization(groups=2, epsilon=1e-4)
        layer.build(inputs.shape)

        def gradients(inputs):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inputs)
                outputs = layer(inputs)
            upstream = tf.ones_like(outputs)
            first = tape.gradient(outputs, inputs, output_gradients=upstream)
            scaled = tape.gradient(
                outputs, inputs, output_gradients=-3.0 * upstream
            )
            del tape
            return first, scaled

        if graph:
            gradients = tf.function(gradients)
        first, scaled = gradients(inputs)
        self.assertAllClose(scaled, -3.0 * first, atol=1e-5, rtol=1e-3)
        self.assertAllClose(first, tf.zeros_like(first), atol=1e-5)

    def test_near_constant_inputs_match_float64_reference(self):
        inputs = (
            10.0
            + np.random.default_rng(0).normal(scale=1e-3, size=(2, 16, 16, 32))
        ).astype("float32")
        gamma = np.linspace(0.5, 1.5, 32)
        layer = layers.GroupNormalization(
            groups=2,
            epsilon=1e-4,
            gamma_initializer=initializers.Constant(gamma),
            beta_initializer=initializers.Constant(0.25),
        )
        grouped = inputs.astype("float64").reshape(2, 16, 16, 2, 16)
        mean = np.mean(grouped, axis=(1, 2, 4), keepdims=True)
        variance = np.var(grouped, axis=(1, 2, 4), keepdims=True)
        expected = (grouped - mean) / np.sqrt(variance + 1e-4)
        expected = expected.reshape(inputs.shape) * gamma + 0.25
        self.assertAllClose(layer(inputs), expected, atol=1e-6, rtol=1e-5)

    @parameterized.named_parameters(("eager", False), ("graph", True))
    def test_centered_mean_overflow_preserves_beta(self, graph):
        if graph and backend.backend() != "tensorflow":
            self.skipTest("Uses TensorFlow graph execution.")
        inputs = ops.convert_to_tensor(
            [[1.7e38, 0.8e38, -2.7e38, -2.4e38]], dtype="float32"
        )
        layer = layers.GroupNormalization(
            groups=1, beta_initializer=initializers.Constant(5.0)
        )
        # Variance is infinite in float32, so finite centered values normalize
        # to zero. Overflow in the extra mean reduction must not create NaNs.
        call = layer
        if graph:
            import tensorflow as tf

            call = tf.function(layer)
        self.assertAllClose(call(inputs), ops.full((1, 4), 5.0))

    @parameterized.named_parameters(
        ("nan", float("nan")),
        ("positive_inf", float("inf")),
        ("negative_inf", -float("inf")),
    )
    def test_nonfinite_inputs_propagate(self, value):
        inputs = ops.convert_to_tensor([[0.0, value, 0.0, 0.0]])
        layer = layers.GroupNormalization(groups=1)
        outputs = ops.convert_to_numpy(layer(inputs))
        self.assertTrue(np.all(np.isnan(outputs)))

    @parameterized.named_parameters(("eager", False), ("graph", True))
    def test_infinite_variance_input_gradient(self, graph):
        if backend.backend() != "tensorflow":
            self.skipTest("Uses TensorFlow gradients.")
        import tensorflow as tf

        inputs = tf.constant([[1.7e38, 0.8e38, -2.7e38, -2.4e38]])
        upstream = tf.constant([[0.001, -0.002, 0.0015, -0.0001]])
        layer = layers.GroupNormalization(groups=1)
        layer.build(inputs.shape)

        def gradient(inputs):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                outputs = layer(inputs)
            return tape.gradient(outputs, inputs, output_gradients=upstream)

        if graph:
            gradient = tf.function(gradient)
        self.assertAllClose(gradient(inputs), tf.zeros_like(inputs))

    def test_groupnorm(self):
        self.run_layer_test(
            layers.GroupNormalization,
            init_kwargs={
                "gamma_regularizer": regularizers.L2(0.01),
                "beta_regularizer": regularizers.L2(0.01),
            },
            input_shape=(3, 4, 32),
            expected_output_shape=(3, 4, 32),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=2,
            supports_masking=True,
        )

        self.run_layer_test(
            layers.GroupNormalization,
            init_kwargs={
                "groups": 4,
                "gamma_constraint": constraints.UnitNorm(),
                "beta_constraint": constraints.UnitNorm(),
            },
            input_shape=(3, 4, 4),
            expected_output_shape=(3, 4, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_undefined_dim_error(self):
        inputs = layers.Input(shape=(2, 2, 2, None))
        layer = layers.GroupNormalization()
        with self.assertRaisesRegex(
            ValueError,
            (
                "input tensor should have a defined dimension but the layer "
                "received an input with shape"
            ),
        ):
            _ = layer(inputs)

    def test_groups_bigger_than_dim_error(self):
        inputs = np.ones(shape=(2, 2, 2, 4))
        layer = layers.GroupNormalization(groups=5)
        with self.assertRaisesRegex(
            ValueError,
            "cannot be more than the number of channels",
        ):
            _ = layer(inputs)

    def test_groups_not_a_multiple_of_dim_error(self):
        inputs = np.ones(shape=(2, 2, 2, 4))
        layer = layers.GroupNormalization(groups=3)
        with self.assertRaisesRegex(
            ValueError,
            "must be a multiple of the number of channels",
        ):
            _ = layer(inputs)

    def test_groups_rejects_invalid_values(self):
        for bad in (0, -2):
            with self.assertRaisesRegex(ValueError, "argument `groups`"):
                layers.GroupNormalization(groups=bad)
        for bad in (1.5, "32"):
            with self.assertRaisesRegex(
                TypeError, "Expected `groups` to be an integer"
            ):
                layers.GroupNormalization(groups=bad)
        # `-1` is the documented sentinel for instance normalization.
        layers.GroupNormalization(groups=-1)

    def test_epsilon_rejects_non_positive_values(self):
        for bad in (None, 0, -1e-3):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                layers.GroupNormalization(epsilon=bad)
        # Boundary: a small positive epsilon is accepted.
        layers.GroupNormalization(epsilon=1e-12)

    def test_groups_instance_norm(self):
        # GroupNormalization with groups=-1 will become InstanceNormalization
        instance_norm_layer_1 = layers.GroupNormalization(
            groups=-1, axis=-1, scale=False, center=False
        )
        instance_norm_layer_2 = layers.GroupNormalization(
            groups=4, axis=-1, scale=False, center=False
        )
        inputs = np.array([[[-1.0, 1.0, 0, 2.0], [1.0, 3.0, -4, -2.0]]])

        outputs_1 = instance_norm_layer_1(inputs)
        outputs_2 = instance_norm_layer_2(inputs)

        self.assertAllClose(outputs_1, outputs_2)

    def test_correctness_instance_norm(self):
        instance_norm_layer = layers.GroupNormalization(
            groups=4, axis=-1, scale=False, center=False
        )

        inputs = np.array([[[-1.0, 1.0, 0, 2.0], [1.0, 3.0, -4, -2.0]]])

        expected_instance_norm_output = np.array(
            [[[-1.0, -1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0]]]
        )

        self.assertAllClose(
            instance_norm_layer(inputs),
            expected_instance_norm_output,
            atol=1e-3,
        )

    def test_correctness_1d(self):
        layer_with_1_group = layers.GroupNormalization(
            groups=1, axis=-1, scale=False, center=False
        )
        layer_with_2_groups = layers.GroupNormalization(
            groups=2, axis=1, scale=False, center=False
        )

        inputs = np.array([[-1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 0, -2.0]])

        expected_output_1_group = np.array(
            [[-0.898, -0.898, 0.539, 0.539, 1.257, 1.257, -0.180, -1.616]],
        )
        self.assertAllClose(
            layer_with_1_group(inputs),
            expected_output_1_group,
            atol=1e-3,
        )

        expected_output_2_groups = np.array(
            [[-1.0, -1.0, 1.0, 1.0, 0.904, 0.904, -0.301, -1.507]]
        )
        self.assertAllClose(
            layer_with_2_groups(inputs),
            expected_output_2_groups,
            atol=1e-3,
        )

    def test_correctness_2d(self):
        layer_with_1_group = layers.GroupNormalization(
            groups=1, axis=-1, scale=False, center=False
        )
        layer_with_2_groups = layers.GroupNormalization(
            groups=2, axis=2, scale=False, center=False
        )

        inputs = np.array([[[-1.0, -1.0, 2.0, 2.0], [1.0, 1.0, 0, -2.0]]])

        expected_output_1_group = np.array(
            [[[-0.898, -0.898, 1.257, 1.257], [0.539, 0.539, -0.180, -1.616]]]
        )

        self.assertAllClose(
            layer_with_1_group(inputs),
            expected_output_1_group,
            atol=1e-3,
        )

        expected_output_2_groups = np.array(
            [[[-1.0, -1.0, 0.904, 0.904], [1.0, 1.0, -0.301, -1.507]]]
        )
        self.assertAllClose(
            layer_with_2_groups(inputs),
            expected_output_2_groups,
            atol=1e-3,
        )

    def test_broadcasting_2d_channels_first(self):
        x = np.arange(16).reshape((1, 4, 2, 2)).astype("float32")
        x = layers.GroupNormalization(groups=2, axis=1)(x)
        self.assertAllClose(
            x,
            np.array(
                [
                    [
                        [[-1.5274, -1.0910], [-0.6546, -0.2182]],
                        [[0.2182, 0.6546], [1.0910, 1.5274]],
                        [[-1.5274, -1.0910], [-0.6546, -0.2182]],
                        [[0.2182, 0.6546], [1.0910, 1.5274]],
                    ]
                ]
            ),
            atol=1e-3,
        )

    def test_large_value_within_autocast_scope(self):
        layer = layers.GroupNormalization(groups=2)
        layer.build((1, 2, 4))
        # Use 70000 to trigger overflow for float16
        large_value = ops.full(layer.gamma.shape, 70000)
        with AutocastScope("float16"):
            layer.gamma.assign(large_value)
            self.assertAllClose(layer.gamma.value, large_value)

    def test_mixed_precision_large_input_no_nan(self):
        # Inputs above the float16 max (~65504) must not overflow to inf/nan
        # before the float32 normalization runs. See issue #22586.
        x = ops.full((2, 4, 8), 70000.0)
        layer = layers.GroupNormalization(groups=2, dtype="mixed_float16")
        out = ops.convert_to_numpy(layer(x))
        self.assertFalse(np.any(np.isnan(out)))

    def test_beta_preserved_for_large_constant_inputs(self):
        x = ops.full((2, 1, 4), 1e10)
        layer = layers.GroupNormalization(
            groups=2,
            scale=False,
            beta_initializer=initializers.Constant(5.0),
        )
        self.assertAllClose(layer(x), ops.full((2, 1, 4), 5.0))

        layer = layers.GroupNormalization(
            groups=2,
            gamma_initializer=initializers.Constant(3.0),
            beta_initializer=initializers.Constant(5.0),
        )
        self.assertAllClose(layer(x), ops.full((2, 1, 4), 5.0))

    def test_numpy_integer_groups(self):
        layer = layers.GroupNormalization(groups=np.int64(2))
        self.assertIs(type(layer.groups), int)
        self.assertEqual(layer.groups, 2)
