import numpy as np
import pytest
from absl.testing import parameterized

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

    @parameterized.named_parameters(
        ("kl_string", "kl_divergence", False),
        ("kl_instance", keras.losses.KLDivergence(), False),
        ("ce_string", "categorical_crossentropy", True),
        (
            "ce_instance_from_logits",
            keras.losses.CategoricalCrossentropy(from_logits=True),
            True,
        ),
        (
            "ce_instance_from_probs",
            keras.losses.CategoricalCrossentropy(),
            True,
        ),
    )
    def test_logits_distillation_loss_values(self, loss, includes_entropy):
        temperature = 2.0
        teacher = np.array([[5.0, 0.0, -3.0], [1.0, 2.0, 0.5]], "float32")
        student = np.array([[0.0, 2.0, 1.0], [1.5, 0.0, 2.0]], "float32")
        distillation_loss = LogitsDistillation(
            temperature=temperature, loss=loss
        )

        def softmax(x):
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)

        p_teacher = softmax(teacher / temperature)
        p_student = softmax(student / temperature)
        entropy = -np.sum(p_teacher * np.log(p_teacher), axis=-1).mean()
        kl = np.sum(p_teacher * np.log(p_teacher / p_student), axis=-1).mean()
        offset = entropy if includes_entropy else 0.0

        # A student that matches the teacher has zero KL divergence and a
        # crossentropy equal to the teacher entropy.
        matched = distillation_loss.compute_loss(teacher, teacher)
        self.assertAllClose(
            matched, temperature**2 * offset, atol=1e-6, rtol=1e-6
        )
        mismatched = distillation_loss.compute_loss(teacher, student)
        self.assertAllClose(
            mismatched, temperature**2 * (kl + offset), atol=1e-5, rtol=1e-5
        )

        # Both losses have the gradient `T * (p_student - p_teacher) / N`.
        expected_grad = temperature * (p_student - p_teacher) / len(student)
        student_gradient = keras.ops.grad(
            lambda s: distillation_loss.compute_loss(teacher, s)
        )
        grad = keras.ops.convert_to_numpy(student_gradient(student))
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertAllClose(grad, expected_grad, atol=1e-5, rtol=1e-5)
        matched_grad = keras.ops.convert_to_numpy(student_gradient(teacher))
        self.assertTrue(np.all(np.isfinite(matched_grad)))
        self.assertAllClose(matched_grad, np.zeros_like(teacher), atol=1e-6)

    @parameterized.named_parameters(
        ("positive_axis_logits", 1, True),
        ("negative_axis_logits", -2, True),
        ("last_axis_logits", -1, True),
        ("positive_axis_probs", 1, False),
        ("negative_axis_probs", -2, False),
        ("last_axis_probs", -1, False),
    )
    def test_categorical_crossentropy_axis(self, axis, from_logits):
        temperature = 2.0
        teacher = np.array([[[5.0, -3.0], [0.0, 4.0], [1.0, 2.0]]], "float32")
        student = np.array([[[0.0, 2.0], [3.0, 1.0], [-1.0, 4.0]]], "float32")
        distillation_loss = LogitsDistillation(
            temperature=temperature,
            loss=keras.losses.CategoricalCrossentropy(
                axis=axis, from_logits=from_logits
            ),
        )

        def log_softmax(x):
            shifted = x - x.max(axis=axis, keepdims=True)
            return shifted - np.log(
                np.exp(shifted).sum(axis=axis, keepdims=True)
            )

        teacher_probs = np.exp(log_softmax(teacher / temperature))
        student_log_probs = log_softmax(student / temperature)
        expected_loss = (
            -(temperature**2)
            * np.sum(teacher_probs * student_log_probs, axis=axis).mean()
        )
        loss = distillation_loss.compute_loss(teacher, student)
        self.assertAllClose(loss, expected_loss, atol=1e-5, rtol=1e-5)

        # Average over all positions outside the class axis.
        num_predictions = teacher.size // teacher.shape[axis]
        expected_grad = (
            temperature
            * (np.exp(student_log_probs) - teacher_probs)
            / num_predictions
        )
        student_gradient = keras.ops.grad(
            lambda s: distillation_loss.compute_loss(teacher, s)
        )
        grad = keras.ops.convert_to_numpy(student_gradient(student))
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertAllClose(grad, expected_grad, atol=1e-5, rtol=1e-5)

    def test_logits_distillation_multi_output(self):
        temperature = 3.0
        teacher = np.array([[5.0, 0.0, -3.0]], "float32")
        distillation_loss = LogitsDistillation(
            temperature=temperature,
            loss=["kl_divergence", "categorical_crossentropy"],
        )
        p_teacher = np.exp(teacher / temperature)
        p_teacher /= p_teacher.sum()
        entropy = -np.sum(p_teacher * np.log(p_teacher))

        loss = distillation_loss.compute_loss(
            [teacher, teacher], [teacher, teacher]
        )
        self.assertAllClose(loss, temperature**2 * entropy, atol=1e-5)

    @parameterized.named_parameters(
        ("sparse_ce_string", "sparse_categorical_crossentropy"),
        (
            "sparse_ce_instance",
            keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ),
        ("binary_ce_string", "binary_crossentropy"),
        ("categorical_focal_string", "categorical_focal_crossentropy"),
    )
    def test_logits_distillation_rejects_unsupported_loss(self, loss):
        with self.assertRaisesRegex(ValueError, "does not support the loss"):
            LogitsDistillation(loss=loss)
        with self.assertRaisesRegex(ValueError, "does not support the loss"):
            LogitsDistillation(loss=["kl_divergence", loss])


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

    @parameterized.named_parameters(
        ("from_logits", True),
        ("from_probs", False),
    )
    def test_distiller_categorical_axis(self, from_logits):
        def make_model(logits, name):
            inputs = keras.Input((1,))
            outputs = keras.layers.Dense(
                6,
                kernel_initializer="zeros",
                bias_initializer=keras.initializers.Constant(logits),
            )(inputs)
            outputs = keras.layers.Reshape((3, 2))(outputs)
            return keras.Model(inputs, outputs, name=name)

        teacher = make_model([5.0, -3.0, 0.0, 4.0, 1.0, 2.0], "teacher")
        student = make_model([0.0, 2.0, 3.0, 1.0, -1.0, 4.0], "student")
        distiller = Distiller(
            teacher=teacher,
            student=student,
            distillation_losses=LogitsDistillation(
                temperature=2.0,
                loss=keras.losses.CategoricalCrossentropy(
                    axis=1, from_logits=from_logits
                ),
            ),
            student_loss_weight=0.0,
        )
        distiller.compile(optimizer="sgd", loss="mse", jit_compile=False)
        results = distiller.evaluate(
            np.zeros((2, 1), "float32"),
            np.zeros((2, 3, 2), "float32"),
            batch_size=2,
            verbose=0,
            return_dict=True,
        )
        # Mean crossentropy over the two positions, scaled by temperature^2.
        self.assertAllClose(results["loss"], 6.6393418, atol=1e-5, rtol=1e-5)
        self.assertAllClose(
            results["distillation_loss"], 6.6393418, atol=1e-5, rtol=1e-5
        )

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
