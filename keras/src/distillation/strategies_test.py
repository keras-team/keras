import json

import numpy as np
import pytest

import keras
from keras import ops
from keras.src.distillation.distiller import Distiller
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


@pytest.mark.requires_trainable_backend
class TestLogitsDistillation(TestCase):
    """Essential test cases for LogitsDistillation strategy."""

    def test_logits_distillation_end_to_end(self):
        """Test logits distillation loss computation end-to-end."""
        strategy = LogitsDistillation(temperature=2.0)

        # Create dummy logits with sufficient difference to ensure non-zero loss
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

    def test_logits_distillation_with_different_loss_types(self):
        """Test logits distillation with different loss types."""
        # Test KL divergence
        strategy_kl = LogitsDistillation(
            temperature=2.0, loss_type="kl_divergence"
        )
        teacher_logits = ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0]]), dtype="float32"
        )
        student_logits = ops.convert_to_tensor(
            np.array([[2.0, 1.0, 4.0]]), dtype="float32"
        )

        loss_kl = strategy_kl.compute_loss(teacher_logits, student_logits)
        self.assertTrue(ops.isfinite(loss_kl))
        self.assertGreater(loss_kl, 0.0)

        # Test categorical crossentropy
        strategy_ce = LogitsDistillation(
            temperature=2.0, loss_type="categorical_crossentropy"
        )
        loss_ce = strategy_ce.compute_loss(teacher_logits, student_logits)
        self.assertTrue(ops.isfinite(loss_ce))
        self.assertGreater(loss_ce, 0.0)


@pytest.mark.requires_trainable_backend
class TestFeatureDistillation(TestCase):
    """Essential test cases for FeatureDistillation strategy."""

    def test_feature_distillation_end_to_end(self):
        """Test feature distillation end-to-end."""
        # Create models with named layers for feature extraction
        teacher = keras.Sequential(
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

        student = keras.Sequential(
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

        # Build models
        dummy_input = np.random.random((1, 10)).astype(np.float32)
        _ = teacher(dummy_input)
        _ = student(dummy_input)

        # Test MSE loss
        strategy_mse = FeatureDistillation(
            loss_type="mse",
            teacher_layer_name="teacher_dense_1",
            student_layer_name="student_dense_1",
        )

        teacher_features = ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_features = ops.convert_to_tensor(
            np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]), dtype="float32"
        )

        loss_mse = strategy_mse.compute_loss(teacher_features, student_features)
        self.assertEqual(len(loss_mse.shape), 0)
        self.assertTrue(ops.isfinite(loss_mse))
        self.assertGreater(loss_mse, 0.0)

        # Test cosine loss
        strategy_cosine = FeatureDistillation(loss_type="cosine")
        loss_cosine = strategy_cosine.compute_loss(
            teacher_features, student_features
        )
        self.assertEqual(len(loss_cosine.shape), 0)
        self.assertTrue(ops.isfinite(loss_cosine))
        self.assertGreaterEqual(loss_cosine, 0.0)


@pytest.mark.requires_trainable_backend
class TestMultiOutputDistillation(TestCase):
    """Essential test cases for MultiOutputDistillation strategy."""

    def test_multi_output_distillation_end_to_end(self):
        """Test multi-output distillation end-to-end."""
        # Create strategies for different outputs
        logits_strategy = LogitsDistillation(temperature=2.0, output_index=0)
        feature_strategy = FeatureDistillation(loss_type="mse")

        # Create multi-output strategy
        strategy = MultiOutputDistillation(
            output_strategies={
                0: logits_strategy,
                1: feature_strategy,
            },
            weights={0: 1.0, 1: 0.5},
        )

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
        loss = strategy.compute_loss(teacher_outputs, student_outputs)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)

        # Check that loss is finite and positive
        self.assertTrue(ops.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_end_to_end_with_multi_output_models(self):
        """Test end-to-end training with multi-output models."""

        # Create multi-output models
        teacher = MultiOutputTeacher(vocab_size=10, hidden_dim=32)
        student = MultiOutputStudent(vocab_size=10, hidden_dim=16)

        # Build models before creating the distiller
        teacher.build((None, 5))
        student.build((None, 5))

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
            strategy=multi_strategy,
            student_loss_weight=0.5,
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            student_loss=[
                "sparse_categorical_crossentropy",
                "sparse_categorical_crossentropy",
            ],
            metrics=[
                ["accuracy"],  # Metrics for output 0
                ["accuracy"],  # Metrics for output 1
            ],
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

    def test_serialization(self):
        """Test MultiOutputDistillation serialization and deserialization."""

        # Create nested strategies
        strategy1 = LogitsDistillation(temperature=3.0, output_index=0)
        strategy2 = FeatureDistillation(loss_type="mse")

        multi_strategy = MultiOutputDistillation(
            output_strategies={0: strategy1, 1: strategy2},
            weights={0: 1.0, 1: 0.5},
        )

        # Test get_config
        config = multi_strategy.get_config()

        # Verify structure
        self.assertIn("output_strategies", config)
        self.assertIn("weights", config)
        self.assertEqual(config["weights"], {0: 1.0, 1: 0.5})

        # Test JSON serialization
        json_str = json.dumps(config)
        self.assertIsInstance(json_str, str)

        # Test from_config
        reconstructed = MultiOutputDistillation.from_config(config)

        # Verify reconstruction
        self.assertEqual(len(reconstructed.output_strategies), 2)
        self.assertEqual(reconstructed.weights, {0: 1.0, 1: 0.5})

        # Verify nested strategies
        self.assertIsInstance(
            reconstructed.output_strategies[0], LogitsDistillation
        )
        self.assertIsInstance(
            reconstructed.output_strategies[1], FeatureDistillation
        )
        self.assertEqual(reconstructed.output_strategies[0].temperature, 3.0)
        self.assertEqual(reconstructed.output_strategies[1].loss_type, "mse")
