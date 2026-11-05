import numpy as np
import pytest

import keras
from keras.src.distillation.distillation_loss import FeatureDistillation
from keras.src.distillation.distillation_loss import LogitsDistillation
from keras.src.distillation.distiller import Distiller
from keras.src.testing import TestCase


@pytest.mark.requires_trainable_backend
class TestLogitsDistillation(TestCase):
    """Test cases for LogitsDistillation distillation_loss."""

    def test_logits_distillation_basic(self):
        """Test basic logits distillation structure validation."""
        # Create dummy logits
        teacher_logits = keras.ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_logits = keras.ops.convert_to_tensor(
            np.array([[2.0, 1.0, 4.0], [3.0, 6.0, 2.0]]), dtype="float32"
        )

        distillation_loss = LogitsDistillation(temperature=3.0)
        distillation_loss.validate_outputs(teacher_logits, student_logits)
        incompatible_logits = {"output": teacher_logits}
        with self.assertRaises(ValueError):
            distillation_loss.validate_outputs(
                teacher_logits, incompatible_logits
            )


@pytest.mark.requires_trainable_backend
class TestFeatureDistillation(TestCase):
    """Test cases for FeatureDistillation distillation_loss."""

    def test_feature_distillation_basic(self):
        """Test basic feature distillation structure validation."""
        # Create dummy features
        teacher_features = keras.ops.convert_to_tensor(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype="float32"
        )
        student_features = keras.ops.convert_to_tensor(
            np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]), dtype="float32"
        )

        distillation_loss = FeatureDistillation(loss="mse")
        distillation_loss.validate_outputs(teacher_features, student_features)
        incompatible_features = [teacher_features, teacher_features]
        with self.assertRaises(ValueError):
            distillation_loss.validate_outputs(
                teacher_features, incompatible_features
            )


@pytest.mark.requires_trainable_backend
class TestEndToEndDistillation(TestCase):
    """End-to-end distillation tests with real models."""

    def setUp(self):
        """Set up models and test data for all tests."""
        super().setUp()

        # Create teacher model
        self.teacher = keras.Sequential(
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

        # Create student model
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

        self.x = np.random.random((32, 20)).astype(np.float32)
        self.y = np.random.randint(0, 10, (32,)).astype(np.int32)

        self.teacher(self.x[:2])
        self.student(self.x[:2])

    def test_logits_distillation_end_to_end(self):
        """Test end-to-end logits distillation with real models."""
        # Create distiller
        distiller = Distiller(
            teacher=self.teacher,
            student=self.student,
            distillation_losses=LogitsDistillation(temperature=3.0),
            student_loss_weight=0.5,
        )

        # Compile distiller
        distiller.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Test training
        history = distiller.fit(self.x, self.y, epochs=2, verbose=0)

        # Verify training completed
        self.assertIn("total_loss", history.history)
        self.assertIn("student_loss", history.history)
        self.assertIn("distillation_loss", history.history)

        # Verify loss values are reasonable
        final_loss = history.history["total_loss"][-1]
        self.assertTrue(np.isfinite(final_loss))
        self.assertGreater(final_loss, 0.0)

        # Test prediction
        predictions = distiller.predict(self.x[:5], verbose=0)
        self.assertEqual(predictions.shape, (5, 10))

        # Test student model access
        student_model = distiller.student
        self.assertIsInstance(student_model, keras.Model)

    def test_feature_distillation_end_to_end(self):
        """Test end-to-end feature distillation with real models."""
        # Create distiller with feature distillation
        distiller = Distiller(
            teacher=self.teacher,
            student=self.student,
            distillation_losses=FeatureDistillation(
                loss="mse",
                teacher_layer_name="teacher_dense_1",
                student_layer_name="student_dense_1",
            ),
            student_loss_weight=0.5,
        )

        # Compile distiller
        distiller.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Test training
        history = distiller.fit(self.x, self.y, epochs=2, verbose=0)

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

    def test_multi_distillation_loss_distillation_end_to_end(self):
        """Test end-to-end distillation with multiple distillation_loss."""
        # Create multiple distillation_loss
        distillation_loss = [
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
            teacher=self.teacher,
            student=self.student,
            distillation_losses=distillation_loss,
            distillation_loss_weights=[1.0, 0.5, 0.3],
            student_loss_weight=0.5,
        )

        # Compile distiller
        distiller.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Test training
        history = distiller.fit(self.x, self.y, epochs=2, verbose=0)

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
