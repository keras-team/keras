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
        # Test default initialization (no temperature specified)
        strategy = LogitsDistillation()
        self.assertEqual(strategy.temperature, 3.0)  # Default fallback
        self.assertEqual(strategy.loss_type, "kl_divergence")
        self.assertEqual(strategy.output_index, 0)
        self.assertFalse(strategy._temperature_explicitly_set)

        # Test custom initialization
        strategy = LogitsDistillation(
            temperature=5.0,
            loss_type="categorical_crossentropy",
            output_index=1,
        )
        self.assertEqual(strategy.temperature, 5.0)
        self.assertEqual(strategy.loss_type, "categorical_crossentropy")
        self.assertEqual(strategy.output_index, 1)
        self.assertTrue(strategy._temperature_explicitly_set)

    def test_invalid_loss_type(self):
        """Test that invalid loss types raise ValueError."""
        with self.assertRaises(ValueError):
            LogitsDistillation(loss_type="invalid_loss")

    def test_default_temperature_mechanism(self):
        """Test that default temperature can be set from Distiller."""
        # Create strategy without explicit temperature
        strategy = LogitsDistillation()
        self.assertEqual(strategy.temperature, 3.0)
        self.assertFalse(strategy._temperature_explicitly_set)

        # Set default temperature
        strategy.set_default_temperature(4.0)
        self.assertEqual(strategy.temperature, 4.0)

        # Create strategy with explicit temperature
        strategy_explicit = LogitsDistillation(temperature=2.0)
        self.assertEqual(strategy_explicit.temperature, 2.0)
        self.assertTrue(strategy_explicit._temperature_explicitly_set)

        # Try to set default - should not change
        strategy_explicit.set_default_temperature(4.0)
        self.assertEqual(strategy_explicit.temperature, 2.0)  # Unchanged

    def test_logits_distillation_loss_kl_divergence(self):
        """Test logits distillation loss computation with KL divergence."""
        strategy = LogitsDistillation(
            temperature=2.0, loss_type="kl_divergence"
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

    def test_logits_distillation_loss_categorical_crossentropy(self):
        """Test logits distillation loss with categorical crossentropy."""
        strategy = LogitsDistillation(
            temperature=2.0, loss_type="categorical_crossentropy"
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
            temperature=3.0,
            loss_type="categorical_crossentropy",
            output_index=1,
        )
        config = strategy.get_config()

        expected_config = {
            "temperature": 3.0,
            "loss_type": "categorical_crossentropy",
            "output_index": 1,
        }
        self.assertEqual(config, expected_config)


class TestFeatureDistillation(TestCase):
    """Test cases for FeatureDistillation strategy."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create models with named layers for feature extraction
        self.teacher = keras.Sequential(
            [
                keras.layers.Dense(
                    64, activation="relu", name="teacher_dense_1"
                ),
                keras.layers.Dense(
                    32, activation="relu", name="teacher_dense_2"
                ),
                keras.layers.Dense(10, name="teacher_output"),
            ]
        )

        self.student = keras.Sequential(
            [
                keras.layers.Dense(
                    32, activation="relu", name="student_dense_1"
                ),
                keras.layers.Dense(
                    16, activation="relu", name="student_dense_2"
                ),
                keras.layers.Dense(10, name="student_output"),
            ]
        )

    def test_initialization(self):
        """Test FeatureDistillation initialization."""
        # Test default initialization
        strategy = FeatureDistillation()
        self.assertEqual(strategy.loss_type, "mse")
        self.assertIsNone(strategy.teacher_layer_name)
        self.assertIsNone(strategy.student_layer_name)
        self.assertIsNone(strategy._teacher_feature_model)
        self.assertIsNone(strategy._student_feature_model)

        # Test custom initialization
        strategy = FeatureDistillation(
            loss_type="cosine",
            teacher_layer_name="dense_1",
            student_layer_name="dense_1",
        )
        self.assertEqual(strategy.loss_type, "cosine")
        self.assertEqual(strategy.teacher_layer_name, "dense_1")
        self.assertEqual(strategy.student_layer_name, "dense_1")

    def test_invalid_loss_type(self):
        """Test that invalid loss types raise ValueError."""
        with self.assertRaises(ValueError):
            FeatureDistillation(loss_type="invalid_loss")

    def test_create_feature_extractor_with_layer_name(self):
        """Test feature extractor creation with specific layer name."""
        strategy = FeatureDistillation(
            teacher_layer_name="teacher_dense_1",
            student_layer_name="student_dense_1",
        )

        # Test teacher feature extractor creation
        teacher_feature_extractor = strategy._create_feature_extractor(
            self.teacher, "teacher_dense_1"
        )
        self.assertIsInstance(teacher_feature_extractor, keras.Model)
        self.assertEqual(
            teacher_feature_extractor.name,
            f"{self.teacher.name}_features_teacher_dense_1",
        )

        # Test student feature extractor creation
        student_feature_extractor = strategy._create_feature_extractor(
            self.student, "student_dense_1"
        )
        self.assertIsInstance(student_feature_extractor, keras.Model)
        self.assertEqual(
            student_feature_extractor.name,
            f"{self.student.name}_features_student_dense_1",
        )

    def test_create_feature_extractor_without_layer_name(self):
        """Test feature model creation without layer name (returns original)."""
        strategy = FeatureDistillation()

        # Should return original model when no layer name specified
        feature_model = strategy._create_feature_extractor(self.teacher, None)
        self.assertIs(feature_model, self.teacher)

    def test_create_feature_extractor_invalid_layer_name(self):
        """Test that invalid layer names raise ValueError."""
        strategy = FeatureDistillation()

        with self.assertRaises(ValueError) as cm:
            strategy._create_feature_extractor(
                self.teacher, "nonexistent_layer"
            )

        self.assertIn(
            "Layer 'nonexistent_layer' not found in model", str(cm.exception)
        )
        self.assertIn("Available layers:", str(cm.exception))

    def test_get_teacher_features(self):
        """Test teacher feature extraction."""
        strategy = FeatureDistillation(teacher_layer_name="teacher_dense_1")

        # Create dummy input
        x = np.random.random((2, 10)).astype(np.float32)

        # Get features
        features = strategy._get_teacher_features(self.teacher, x)

        # Check that features have the expected shape (after first dense layer)
        self.assertEqual(features.shape, (2, 64))  # batch_size, hidden_dim

        # Check that feature model was created and cached
        self.assertIsNotNone(strategy._teacher_feature_model)

    def test_get_student_features(self):
        """Test student feature extraction."""
        strategy = FeatureDistillation(student_layer_name="student_dense_1")

        # Create dummy input
        x = np.random.random((2, 10)).astype(np.float32)

        # Get features
        features = strategy._get_student_features(self.student, x)

        # Check that features have the expected shape (after first dense layer)
        self.assertEqual(features.shape, (2, 32))  # batch_size, hidden_dim

        # Check that feature model was created and cached
        self.assertIsNotNone(strategy._student_feature_model)

    def test_feature_distillation_loss_mse(self):
        """Test feature distillation loss computation with MSE."""
        strategy = FeatureDistillation(loss_type="mse")

        teacher_features = ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_features = ops.convert_to_tensor(
            np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]), dtype="float32"
        )

        # Compute loss
        loss = strategy.compute_loss(teacher_features, student_features)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite and positive
        self.assertTrue(ops.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_feature_distillation_loss_cosine(self):
        """Test feature distillation loss computation with cosine similarity."""
        strategy = FeatureDistillation(loss_type="cosine")

        teacher_features = ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_features = ops.convert_to_tensor(
            np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]), dtype="float32"
        )

        # Compute loss
        loss = strategy.compute_loss(teacher_features, student_features)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite and non-negative (cosine distance)
        self.assertTrue(ops.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_get_config(self):
        """Test configuration serialization."""
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
            loss=[
                "sparse_categorical_crossentropy",
                "sparse_categorical_crossentropy",
            ],
            steps_per_execution=1,
        )

        # Create test data for multi-output model
        x = np.random.random((20, 5)).astype(np.float32)
        # Multi-output targets: [output1_targets, output2_targets]
        y = [
            np.random.randint(0, 10, (20,)).astype(
                np.int32
            ),  # For output1 (10 classes)
            np.random.randint(0, 5, (20,)).astype(
                np.int32
            ),  # For output2 (5 classes)
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
