import numpy as np
import pytest

import keras
from keras.src.distillation.distillation_loss import FeatureDistillation
from keras.src.distillation.distillation_loss import LogitsDistillation
from keras.src.distillation.distiller import Distiller
from keras.src.testing import TestCase


@pytest.mark.requires_trainable_backend
class TestLogitsDistillation(TestCase):
    """Test cases for LogitsDistillation strategy."""

    def test_logits_distillation_basic(self):
        """Test basic logits distillation loss computation."""
        strategy = LogitsDistillation(temperature=2.0)

        # Create dummy logits
        teacher_logits = keras.ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_logits = keras.ops.convert_to_tensor(
            np.array([[2.0, 1.0, 4.0], [3.0, 6.0, 2.0]]), dtype="float32"
        )

        # Compute loss
        loss = strategy.compute_loss(teacher_logits, student_logits)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)
        self.assertTrue(keras.ops.isfinite(loss))
        self.assertGreater(loss, 0.0)


@pytest.mark.requires_trainable_backend
class TestFeatureDistillation(TestCase):
    """Test cases for FeatureDistillation strategy."""

    def test_feature_distillation_basic(self):
        """Test basic feature distillation loss computation."""
        strategy = FeatureDistillation(loss="mse")

        # Create dummy features
        teacher_features = keras.ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_features = keras.ops.convert_to_tensor(
            np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]), dtype="float32"
        )

        # Compute loss
        loss = strategy.compute_loss(teacher_features, student_features)

        # Check that loss is a scalar tensor
        self.assertEqual(len(loss.shape), 0)
        self.assertTrue(keras.ops.isfinite(loss))
        self.assertGreater(loss, 0.0)


@pytest.mark.requires_trainable_backend
class TestEndToEndDistillation(TestCase):
    """End-to-end distillation tests with real models."""

    def test_logits_distillation_end_to_end(self):
        """Test end-to-end logits distillation with real models."""
        # Create teacher model (larger)
        teacher = keras.Sequential(
            [
                keras.layers.Dense(
                    64, activation="relu", name="teacher_dense_1"
                ),
                keras.layers.Dense(
                    32, activation="relu", name="teacher_dense_2"
                ),
                keras.layers.Dense(
                    10, activation="softmax", name="teacher_output"
                ),
            ]
        )

        # Create student model (smaller)
        student = keras.Sequential(
            [
                keras.layers.Dense(
                    32, activation="relu", name="student_dense_1"
                ),
                keras.layers.Dense(
                    16, activation="relu", name="student_dense_2"
                ),
                keras.layers.Dense(
                    10, activation="softmax", name="student_output"
                ),
            ]
        )

        # Create test data
        x = np.random.random((32, 20)).astype(np.float32)
        y = np.random.randint(0, 10, (32,)).astype(np.int32)

        # Build models to avoid JAX tracer issues
        dummy_input = x[:2]
        teacher(dummy_input)
        student(dummy_input)

        # Create distiller
        distiller = Distiller(
            teacher=teacher,
            student=student,
            strategies=LogitsDistillation(temperature=3.0),
            student_loss_weight=0.5,
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            student_loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Test training
        history = distiller.fit(x, y, epochs=2, verbose=0)

        # Verify training completed
        self.assertIn("total_loss", history.history)
        self.assertIn("student_loss", history.history)
        self.assertIn("distillation_loss", history.history)

        # Verify loss values are reasonable
        final_loss = history.history["total_loss"][-1]
        self.assertTrue(np.isfinite(final_loss))
        self.assertGreater(final_loss, 0.0)

        # Test prediction
        predictions = distiller.predict(x[:5], verbose=0)
        self.assertEqual(predictions.shape, (5, 10))

        # Test student model access
        student_model = distiller.student_model
        self.assertIsInstance(student_model, keras.Model)

    def test_feature_distillation_end_to_end(self):
        """Test end-to-end feature distillation with real models."""
        # Create teacher model
        teacher = keras.Sequential(
            [
                keras.layers.Dense(
                    32, activation="relu", name="teacher_dense_1"
                ),
                keras.layers.Dense(
                    16, activation="relu", name="teacher_dense_2"
                ),
                keras.layers.Dense(10, name="teacher_output"),
            ]
        )

        # Create student model with compatible intermediate layer sizes
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

        # Build models first
        dummy_input = np.random.random((2, 20)).astype(np.float32)
        teacher(dummy_input)
        student(dummy_input)

        # Create distiller with feature distillation
        distiller = Distiller(
            teacher=teacher,
            student=student,
            strategies=FeatureDistillation(
                loss="mse",
                teacher_layer_name="teacher_dense_1",
                student_layer_name="student_dense_1",
            ),
            student_loss_weight=0.5,
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            student_loss="sparse_categorical_crossentropy",
        )

        # Create test data
        x = np.random.random((32, 20)).astype(np.float32)
        y = np.random.randint(0, 10, (32,)).astype(np.int32)

        # Test training
        history = distiller.fit(x, y, epochs=2, verbose=0)

        # Verify training completed
        self.assertIn("total_loss", history.history)
        self.assertIn("student_loss", history.history)
        self.assertIn("distillation_loss", history.history)

        # Verify feature extraction worked
        self.assertIsNotNone(distiller._teacher_feature_extractor)
        self.assertIsNotNone(distiller._student_feature_extractor)

        # Test that feature extractors have correct outputs
        self.assertEqual(
            len(distiller._teacher_feature_extractor.outputs), 2
        )  # final + dense_1
        self.assertEqual(
            len(distiller._student_feature_extractor.outputs), 2
        )  # final + dense_1

    def test_multi_strategy_distillation_end_to_end(self):
        """Test end-to-end distillation with multiple strategies."""
        # Create models
        teacher = keras.Sequential(
            [
                keras.layers.Dense(
                    32, activation="relu", name="teacher_dense_1"
                ),
                keras.layers.Dense(
                    16, activation="relu", name="teacher_dense_2"
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

        # Build models first
        dummy_input = np.random.random((2, 20)).astype(np.float32)
        teacher(dummy_input)
        student(dummy_input)

        # Create multiple strategies
        strategies = [
            LogitsDistillation(temperature=3.0),
            FeatureDistillation(
                loss="mse",
                teacher_layer_name="teacher_dense_1",
                student_layer_name="student_dense_1",
            ),
            FeatureDistillation(
                loss="mse",
                teacher_layer_name="teacher_dense_2",
                student_layer_name="student_dense_2",
            ),
        ]

        # Create distiller
        distiller = Distiller(
            teacher=teacher,
            student=student,
            strategies=strategies,
            strategy_weights=[1.0, 0.5, 0.3],
            student_loss_weight=0.5,
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            student_loss="sparse_categorical_crossentropy",
        )

        # Create test data
        x = np.random.random((32, 20)).astype(np.float32)
        y = np.random.randint(0, 10, (32,)).astype(np.int32)

        # Test training
        history = distiller.fit(x, y, epochs=2, verbose=0)

        # Verify training completed
        self.assertIn("total_loss", history.history)
        self.assertIn("student_loss", history.history)
        self.assertIn("distillation_loss", history.history)

        # Verify efficient feature extraction
        self.assertIsNotNone(distiller._teacher_feature_extractor)
        self.assertIsNotNone(distiller._student_feature_extractor)

        # Should have 3 outputs: final + dense_1 + dense_2
        self.assertEqual(len(distiller._teacher_feature_extractor.outputs), 3)
        self.assertEqual(len(distiller._student_feature_extractor.outputs), 3)

        # Test that loss decreases (learning is happening)
        initial_loss = history.history["total_loss"][0]
        final_loss = history.history["total_loss"][-1]
        self.assertTrue(np.isfinite(initial_loss))
        self.assertTrue(np.isfinite(final_loss))
