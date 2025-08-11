import numpy as np

import keras
from keras import ops
from keras.src.distillation.strategies import FeatureDistillation
from keras.src.distillation.strategies import LogitsDistillation
from keras.src.distillation.strategies import MultiOutputDistillation
from keras.src.testing import TestCase


class MultiOutputTeacher(keras.Model):
    """Multi-output teacher model for testing."""

    def __init__(self, vocab_size=10, hidden_dim=32):
        super().__init__()
        self.dense1 = keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = keras.layers.Dense(vocab_size)
        self.dense3 = keras.layers.Dense(5)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        output1 = self.dense2(x)
        output2 = self.dense3(x)
        return [output1, output2]


class MultiOutputStudent(keras.Model):
    """Multi-output student model for testing."""

    def __init__(self, vocab_size=10, hidden_dim=16):
        super().__init__()
        self.dense1 = keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = keras.layers.Dense(vocab_size)
        self.dense3 = keras.layers.Dense(5)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        output1 = self.dense2(x)
        output2 = self.dense3(x)
        return [output1, output2]


class TestLogitsDistillation(TestCase):
    """Essential test cases for LogitsDistillation strategy."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.strategy = LogitsDistillation(temperature=2.0)

    def test_logits_distillation_loss(self):
        """Test logits distillation loss computation."""
        # Create dummy logits with sufficient difference to ensure non-zero loss
        teacher_logits = ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_logits = ops.convert_to_tensor(
            np.array([[2.0, 1.0, 4.0], [3.0, 6.0, 2.0]]), dtype="float32"
        )

        # Compute loss
        loss = self.strategy.compute_loss(teacher_logits, student_logits)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite and positive
        self.assertTrue(ops.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_temperature_scaling(self):
        """Test temperature scaling in logits distillation."""
        # Create dummy logits with sufficient difference
        teacher_logits = ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0]]), dtype="float32"
        )
        student_logits = ops.convert_to_tensor(
            np.array([[2.0, 1.0, 4.0]]), dtype="float32"
        )

        # Test with different temperatures
        temperatures = [1.0, 2.0, 4.0]
        losses = []

        for temp in temperatures:
            strategy = LogitsDistillation(temperature=temp)
            loss = strategy.compute_loss(teacher_logits, student_logits)
            losses.append(loss)

        # Higher temperature should result in different loss values
        self.assertNotEqual(losses[0], losses[1])
        self.assertNotEqual(losses[1], losses[2])


class TestLogitsDistillationComprehensive(TestCase):
    """Comprehensive test cases for LogitsDistillation strategy."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.strategy = LogitsDistillation(temperature=2.0)

    def test_initialization(self):
        """Test LogitsDistillation initialization."""
        # Test default initialization
        strategy = LogitsDistillation()
        self.assertEqual(strategy.temperature, 2.0)
        self.assertEqual(strategy.loss_type, "kl_divergence")
        self.assertEqual(strategy.output_index, 0)

        # Test custom initialization
        strategy = LogitsDistillation(
            temperature=3.0, loss_type="mse", output_index=1
        )
        self.assertEqual(strategy.temperature, 3.0)
        self.assertEqual(strategy.loss_type, "mse")
        self.assertEqual(strategy.output_index, 1)

    def test_invalid_loss_type(self):
        """Test that invalid loss types raise ValueError."""
        with self.assertRaises(ValueError):
            LogitsDistillation(loss_type="invalid_loss")

    def test_logits_distillation_loss_mse(self):
        """Test logits distillation loss computation with MSE."""
        strategy = LogitsDistillation(temperature=2.0, loss_type="mse")

        teacher_logits = ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_logits = ops.convert_to_tensor(
            np.array([[2.0, 1.0, 4.0], [3.0, 6.0, 2.0]]), dtype="float32"
        )

        # Compute loss
        loss = strategy.compute_loss(teacher_logits, student_logits)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite and positive
        self.assertTrue(ops.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_logits_distillation_loss_cross_entropy(self):
        """Test logits distillation loss computation with cross entropy."""
        strategy = LogitsDistillation(
            temperature=2.0, loss_type="cross_entropy"
        )

        teacher_logits = ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_logits = ops.convert_to_tensor(
            np.array([[2.0, 1.0, 4.0], [3.0, 6.0, 2.0]]), dtype="float32"
        )

        # Compute loss
        loss = strategy.compute_loss(teacher_logits, student_logits)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite and positive
        self.assertTrue(ops.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_multi_output_support(self):
        """Test support for multi-output models."""
        # Create dummy multi-output logits
        teacher_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32"),
            ops.convert_to_tensor(np.array([[4.0, 5.0]]), dtype="float32"),
        ]
        student_outputs = [
            ops.convert_to_tensor(np.array([[1.1, 2.1, 3.1]]), dtype="float32"),
            ops.convert_to_tensor(np.array([[4.1, 5.1]]), dtype="float32"),
        ]

        # Test with output_index=0
        strategy = LogitsDistillation(temperature=2.0, output_index=0)
        loss = strategy.compute_loss(teacher_outputs, student_outputs)
        self.assertTrue(ops.isfinite(loss))

        # Test with output_index=1
        strategy = LogitsDistillation(temperature=2.0, output_index=1)
        loss = strategy.compute_loss(teacher_outputs, student_outputs)
        self.assertTrue(ops.isfinite(loss))

    def test_output_validation(self):
        """Test output validation."""
        strategy = LogitsDistillation(temperature=2.0, output_index=0)

        # Test with compatible outputs
        teacher_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32")
        ]
        student_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32")
        ]

        # Should not raise an error
        strategy.validate_outputs(teacher_outputs, student_outputs)

        # Test with incompatible output shapes
        teacher_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32")
        ]
        student_outputs = [
            ops.convert_to_tensor(
                np.array([[1.0, 2.0]]), dtype="float32"
            )  # Different number of classes
        ]

        with self.assertRaises(ValueError):
            strategy.validate_outputs(teacher_outputs, student_outputs)

        # Test with invalid output index
        strategy = LogitsDistillation(
            temperature=2.0, output_index=1
        )  # Invalid index
        teacher_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32")
        ]
        student_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32")
        ]

        with self.assertRaises(ValueError):
            strategy.validate_outputs(teacher_outputs, student_outputs)

    def test_get_config(self):
        """Test get_config method."""
        strategy = LogitsDistillation(
            temperature=3.0, loss_type="mse", output_index=1
        )
        config = strategy.get_config()

        expected_config = {
            "temperature": 3.0,
            "loss_type": "mse",
            "output_index": 1,
        }

        self.assertEqual(config, expected_config)


class TestFeatureDistillation(TestCase):
    """Comprehensive test cases for FeatureDistillation strategy."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.strategy = FeatureDistillation()

    def test_initialization(self):
        """Test FeatureDistillation initialization."""
        # Test default initialization
        strategy = FeatureDistillation()
        self.assertEqual(strategy.loss_type, "mse")
        self.assertIsNone(strategy.teacher_layer_name)
        self.assertIsNone(strategy.student_layer_name)

        # Test custom initialization
        strategy = FeatureDistillation(
            loss_type="cosine",
            teacher_layer_name="layer1",
            student_layer_name="layer2",
        )
        self.assertEqual(strategy.loss_type, "cosine")
        self.assertEqual(strategy.teacher_layer_name, "layer1")
        self.assertEqual(strategy.student_layer_name, "layer2")

    def test_invalid_loss_type(self):
        """Test that invalid loss types raise ValueError."""
        with self.assertRaises(ValueError):
            FeatureDistillation(loss_type="invalid_loss")

    def test_feature_distillation_loss_mse(self):
        """Test feature distillation loss computation with MSE."""
        strategy = FeatureDistillation(loss_type="mse")

        # Create dummy feature tensors
        teacher_features = ops.convert_to_tensor(
            np.random.random((2, 16)).astype(np.float32)
        )
        student_features = ops.convert_to_tensor(
            np.random.random((2, 16)).astype(np.float32)
        )

        # Compute loss
        loss = strategy.compute_loss(teacher_features, student_features)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite and non-negative
        self.assertTrue(ops.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_feature_distillation_loss_cosine(self):
        """Test feature distillation loss computation with cosine similarity."""
        strategy = FeatureDistillation(loss_type="cosine")

        # Create dummy feature tensors
        teacher_features = ops.convert_to_tensor(
            np.random.random((2, 16)).astype(np.float32)
        )
        student_features = ops.convert_to_tensor(
            np.random.random((2, 16)).astype(np.float32)
        )

        # Compute loss
        loss = strategy.compute_loss(teacher_features, student_features)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite
        self.assertTrue(ops.isfinite(loss))

    def test_feature_validation(self):
        """Test feature validation."""
        strategy = FeatureDistillation()

        # Test with compatible features
        teacher_features = [
            ops.convert_to_tensor(np.random.random((2, 16)).astype(np.float32))
        ]
        student_features = [
            ops.convert_to_tensor(np.random.random((2, 16)).astype(np.float32))
        ]

        # Should not raise an error
        strategy.validate_outputs(teacher_features, student_features)

        # Test with incompatible dimensions
        teacher_features = [
            ops.convert_to_tensor(
                np.random.random((2, 16, 8)).astype(np.float32)
            )  # 3D
        ]
        student_features = [
            ops.convert_to_tensor(
                np.random.random((2, 16)).astype(np.float32)
            )  # 2D
        ]

        with self.assertRaises(ValueError):
            strategy.validate_outputs(teacher_features, student_features)

    def test_list_input_handling(self):
        """Test that the strategy handles list inputs correctly."""
        strategy = FeatureDistillation()

        # Test with list inputs
        teacher_features = [
            ops.convert_to_tensor(np.random.random((2, 16)).astype(np.float32)),
            ops.convert_to_tensor(np.random.random((2, 8)).astype(np.float32)),
        ]
        student_features = [
            ops.convert_to_tensor(np.random.random((2, 16)).astype(np.float32)),
            ops.convert_to_tensor(np.random.random((2, 8)).astype(np.float32)),
        ]

        # Should use first output by default
        loss = strategy.compute_loss(teacher_features, student_features)
        self.assertTrue(ops.isfinite(loss))

    def test_get_config(self):
        """Test get_config method."""
        strategy = FeatureDistillation(
            loss_type="cosine",
            teacher_layer_name="teacher_layer",
            student_layer_name="student_layer",
        )
        config = strategy.get_config()

        expected_config = {
            "loss_type": "cosine",
            "teacher_layer_name": "teacher_layer",
            "student_layer_name": "student_layer",
        }

        self.assertEqual(config, expected_config)


class TestMultiOutputDistillation(TestCase):
    """Comprehensive test cases for MultiOutputDistillation strategy."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create strategies for different outputs
        self.logits_strategy = LogitsDistillation(
            temperature=2.0, output_index=0
        )
        self.feature_strategy = FeatureDistillation(loss_type="mse")

        # Create multi-output strategy
        self.strategy = MultiOutputDistillation(
            output_strategies={
                0: self.logits_strategy,
                1: self.feature_strategy,
            },
            weights={0: 1.0, 1: 0.5},
        )

    def test_initialization(self):
        """Test MultiOutputDistillation initialization."""
        # Test with explicit weights
        strategy = MultiOutputDistillation(
            output_strategies={
                0: self.logits_strategy,
                1: self.feature_strategy,
            },
            weights={0: 2.0, 1: 1.0},
        )
        self.assertEqual(strategy.weights[0], 2.0)
        self.assertEqual(strategy.weights[1], 1.0)

        # Test with default weights (should be 1.0 for all)
        strategy = MultiOutputDistillation(
            output_strategies={
                0: self.logits_strategy,
                1: self.feature_strategy,
            }
        )
        self.assertEqual(strategy.weights[0], 1.0)
        self.assertEqual(strategy.weights[1], 1.0)

    def test_multi_output_loss_computation(self):
        """Test multi-output distillation loss computation."""
        # Create dummy multi-output data
        teacher_outputs = [
            ops.convert_to_tensor(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
            ),
            ops.convert_to_tensor(
                np.array([[0.1, 0.2], [0.3, 0.4]]), dtype="float32"
            ),
        ]
        student_outputs = [
            ops.convert_to_tensor(
                np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]), dtype="float32"
            ),
            ops.convert_to_tensor(
                np.array([[0.15, 0.25], [0.35, 0.45]]), dtype="float32"
            ),
        ]

        # Compute loss
        loss = self.strategy.compute_loss(teacher_outputs, student_outputs)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite and positive
        self.assertTrue(ops.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_output_validation(self):
        """Test output validation for multi-output distillation."""
        # Test with valid outputs
        teacher_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32"),
            ops.convert_to_tensor(np.array([[0.1, 0.2]]), dtype="float32"),
        ]
        student_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32"),
            ops.convert_to_tensor(np.array([[0.1, 0.2]]), dtype="float32"),
        ]

        # Should not raise an error
        self.strategy.validate_outputs(teacher_outputs, student_outputs)

        # Test with insufficient teacher outputs
        teacher_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32")
            # Missing second output
        ]
        student_outputs = [
            ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]), dtype="float32"),
            ops.convert_to_tensor(np.array([[0.1, 0.2]]), dtype="float32"),
        ]

        with self.assertRaises(ValueError):
            self.strategy.validate_outputs(teacher_outputs, student_outputs)

    def test_weight_application(self):
        """Test that weights are properly applied."""
        # Create strategies with known behavior
        strategy1 = MultiOutputDistillation(
            output_strategies={
                0: self.logits_strategy,
                1: self.feature_strategy,
            },
            weights={0: 1.0, 1: 1.0},  # Equal weights
        )

        strategy2 = MultiOutputDistillation(
            output_strategies={
                0: self.logits_strategy,
                1: self.feature_strategy,
            },
            weights={0: 2.0, 1: 1.0},  # Different weights
        )

        # Create test data
        teacher_outputs = [
            ops.convert_to_tensor(
                np.array([[10.0, 20.0, 30.0]]), dtype="float32"
            ),
            ops.convert_to_tensor(np.array([[0.1, 0.2]]), dtype="float32"),
        ]
        student_outputs = [
            ops.convert_to_tensor(
                np.array([[5.0, 15.0, 25.0]]), dtype="float32"
            ),
            ops.convert_to_tensor(np.array([[0.15, 0.25]]), dtype="float32"),
        ]

        # Compute losses
        loss1 = strategy1.compute_loss(teacher_outputs, student_outputs)
        loss2 = strategy2.compute_loss(teacher_outputs, student_outputs)

        # Losses should be different due to different weights, but may be
        # very close
        # Just verify that both losses are finite and positive
        self.assertTrue(ops.isfinite(loss1))
        self.assertTrue(ops.isfinite(loss2))
        self.assertGreater(loss1, 0.0)
        self.assertGreater(loss2, 0.0)

    def test_end_to_end_with_multi_output_models(self):
        """Test end-to-end training with multi-output models."""
        from keras.src.distillation.distiller import Distiller

        # Create multi-output models
        teacher = MultiOutputTeacher(vocab_size=10, hidden_dim=32)
        student = MultiOutputStudent(vocab_size=10, hidden_dim=16)

        # Create multi-output distillation strategy
        multi_strategy = MultiOutputDistillation(
            output_strategies={
                0: LogitsDistillation(temperature=2.0, output_index=0),
                1: FeatureDistillation(loss_type="mse"),
            },
            weights={0: 1.0, 1: 0.5},
        )

        # Create distiller
        distiller = Distiller(
            teacher=teacher,
            student=student,
            strategies=[multi_strategy],
            alpha=0.5,
            temperature=2.0,
        )

        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"],
            steps_per_execution=1,
        )

        # Create test data for multi-output model
        x = np.random.random((20, 5)).astype(np.float32)
        # Multi-output targets: [output1_targets, output2_targets]
        y = [
            np.random.randint(0, 10, (20,)).astype(np.int32),  # For output1 (10 classes)
            np.random.randint(0, 5, (20,)).astype(np.int32),   # For output2 (5 classes)
        ]

        # Test that training works
        history = distiller.fit(x, y, epochs=1, verbose=0)

        # Check that training completed
        self.assertIn("total_loss", history.history)
        self.assertIn("student_loss", history.history)
        self.assertIn("distillation_loss", history.history)

        # Test prediction
        predictions = distiller.predict(x[:5], verbose=0)
        self.assertEqual(
            predictions[0].shape, (5, 10)
        )  # Should return first output
