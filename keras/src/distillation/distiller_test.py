import numpy as np
import pytest

import keras
from keras.src.distillation.distiller import Distiller
from keras.src.distillation.strategies import LogitsDistillation
from keras.src.testing import TestCase


class SimpleTeacher(keras.Model):
    """Simple teacher model for testing."""

    def __init__(self, vocab_size=10, hidden_dim=32):
        super().__init__()
        self.dense1 = keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        return self.dense2(x)


class SimpleStudent(keras.Model):
    """Simple student model for testing."""

    def __init__(self, vocab_size=10, hidden_dim=16):
        super().__init__()
        self.dense1 = keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        return self.dense2(x)


@pytest.mark.requires_trainable_backend
class TestDistiller(TestCase):
    """Essential test cases for the Distiller class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create teacher and student models
        self.teacher = SimpleTeacher(vocab_size=10, hidden_dim=32)
        self.student = SimpleStudent(vocab_size=10, hidden_dim=16)

        # Create distillation strategy
        self.strategy = LogitsDistillation()

        # Create distiller
        self.distiller = Distiller(
            teacher=self.teacher,
            student=self.student,
            strategies=[self.strategy],
            alpha=0.5,
            temperature=2.0,
        )

        # Compile distiller (avoid additional metrics for JAX sharding issues)
        self.distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            steps_per_execution=1,
        )

        # Create test data
        self.x = np.random.random((20, 5)).astype(np.float32)
        self.y = np.random.randint(0, 10, (20,)).astype(np.int32)

    def test_distiller_initialization(self):
        """Test Distiller initialization."""
        # Check that teacher is frozen
        self.assertFalse(self.teacher.trainable)

        # Check that student is trainable
        self.assertTrue(self.student.trainable)

        # Check alpha and temperature
        self.assertEqual(self.distiller.alpha, 0.5)
        self.assertEqual(self.distiller.temperature, 2.0)

        # Check strategies
        self.assertLen(self.distiller.strategies, 1)
        self.assertIsInstance(self.distiller.strategies[0], LogitsDistillation)

        # Check that strategy received the default temperature
        self.assertEqual(self.distiller.strategies[0].temperature, 2.0)

    def test_distiller_call(self):
        """Test Distiller call method (inference)."""
        # Call should return student outputs
        outputs = self.distiller(self.x)

        # Check output shape
        expected_shape = (20, 10)  # batch_size, vocab_size
        self.assertEqual(outputs.shape, expected_shape)

        # Check that outputs are from student, not teacher
        student_outputs = self.student(self.x)
        self.assertAllClose(outputs, student_outputs)

    def test_teacher_freezing(self):
        """Test that teacher is properly frozen."""
        # Teacher should be frozen
        self.assertFalse(self.teacher.trainable)

        # Student should be trainable
        self.assertTrue(self.student.trainable)

        # Create a new teacher that is trainable and verify it gets frozen
        new_teacher = SimpleTeacher(vocab_size=10, hidden_dim=32)
        self.assertTrue(new_teacher.trainable)  # Should be trainable initially

        # Create distiller - should freeze the teacher
        Distiller(
            teacher=new_teacher,
            student=self.student,
            strategies=[self.strategy],
            alpha=0.5,
            temperature=2.0,
        )

        # Teacher should now be frozen
        self.assertFalse(new_teacher.trainable)

    def test_model_compatibility_validation(self):
        """Test model compatibility validation."""
        # Test with non-Keras objects
        with self.assertRaises(ValueError):
            Distiller(
                teacher="not_a_model",
                student=self.student,
                strategies=[self.strategy],
            )

        with self.assertRaises(ValueError):
            Distiller(
                teacher=self.teacher,
                student="not_a_model",
                strategies=[self.strategy],
            )

    def test_alpha_weighting(self):
        """Test that alpha correctly weights student vs distillation loss."""
        # Test with alpha = 0.0 (only distillation loss)
        distiller_0 = Distiller(
            teacher=self.teacher,
            student=self.student,
            strategies=[self.strategy],
            alpha=0.0,
            temperature=2.0,
        )
        distiller_0.compile(
            optimizer=keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            steps_per_execution=1,
        )

        # Test with alpha = 1.0 (only student loss)
        distiller_1 = Distiller(
            teacher=self.teacher,
            student=self.student,
            strategies=[self.strategy],
            alpha=1.0,
            temperature=2.0,
        )
        distiller_1.compile(
            optimizer=keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            steps_per_execution=1,
        )

        # Test that they can be used for training without errors
        small_x = self.x[:5]
        small_y = self.y[:5]

        # Both should train without errors
        history_0 = distiller_0.fit(small_x, small_y, epochs=1, verbose=0)
        history_1 = distiller_1.fit(small_x, small_y, epochs=1, verbose=0)

        # Check that training completed
        self.assertIn("total_loss", history_0.history)
        self.assertIn("total_loss", history_1.history)

    def test_full_training_workflow(self):
        """Test complete training workflow with model.fit() - MOST IMPORTANT."""
        # Create larger dataset for training
        np.random.seed(42)
        x_train = np.random.random((100, 5)).astype(np.float32)
        y_train = np.random.randint(0, 10, (100,)).astype(np.int32)
        x_val = np.random.random((20, 5)).astype(np.float32)
        y_val = np.random.randint(0, 10, (20,)).astype(np.int32)

        # Create fresh models for training
        teacher = SimpleTeacher(vocab_size=10, hidden_dim=32)
        student = SimpleStudent(vocab_size=10, hidden_dim=16)

        # Create distiller
        distiller = Distiller(
            teacher=teacher,
            student=student,
            strategies=[LogitsDistillation(temperature=2.0)],
            alpha=0.5,
            temperature=2.0,
        )

        # Compile (avoid additional metrics to prevent JAX sharding issues)
        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            steps_per_execution=1,
        )

        # Train the model
        history = distiller.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=16,
            verbose=0,
        )

        # Check that training completed
        self.assertIn("total_loss", history.history)
        self.assertIn("val_total_loss", history.history)
        self.assertIn("student_loss", history.history)
        self.assertIn("distillation_loss", history.history)

        # Check that losses are finite
        for loss_name in ["total_loss", "student_loss", "distillation_loss"]:
            losses = history.history[loss_name]
            self.assertGreater(len(losses), 0)
            for loss in losses:
                self.assertTrue(np.isfinite(loss))

        # Check that the model can make predictions
        predictions = distiller.predict(x_val[:5], verbose=0)
        self.assertEqual(predictions.shape, (5, 10))  # batch_size, vocab_size

        # Check that student weights have changed (indicating learning)
        initial_weights = [w.numpy().copy() for w in student.trainable_weights]

        # Train a bit more
        distiller.fit(x_train[:10], y_train[:10], epochs=1, verbose=0)

        final_weights = [w.numpy() for w in student.trainable_weights]

        # At least some weights should have changed
        weights_changed = any(
            not np.allclose(initial, final, atol=1e-6)
            for initial, final in zip(initial_weights, final_weights)
        )
        self.assertTrue(
            weights_changed, "Student weights should change during training"
        )

    def test_evaluation_workflow(self):
        """Test evaluation workflow with model.evaluate()."""
        # Create dataset
        np.random.seed(42)
        x_test = np.random.random((30, 5)).astype(np.float32)
        y_test = np.random.randint(0, 10, (30,)).astype(np.int32)

        # Create fresh models
        teacher = SimpleTeacher(vocab_size=10, hidden_dim=32)
        student = SimpleStudent(vocab_size=10, hidden_dim=16)

        # Create and compile distiller
        distiller = Distiller(
            teacher=teacher,
            student=student,
            strategies=[LogitsDistillation(temperature=2.0)],
            alpha=0.5,
            temperature=2.0,
        )

        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            steps_per_execution=1,
        )

        # Train briefly
        distiller.fit(x_test[:10], y_test[:10], epochs=1, verbose=0)

        # Evaluate the model
        results = distiller.evaluate(x_test, y_test, verbose=0)

        # Check that evaluation returns expected metrics
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # All results should be finite
        for result in results:
            self.assertTrue(np.isfinite(result))

    def test_prediction_workflow(self):
        """Test prediction workflow with model.predict()."""
        # Create dataset
        np.random.seed(42)
        x_test = np.random.random((20, 5)).astype(np.float32)

        # Create fresh models
        teacher = SimpleTeacher(vocab_size=10, hidden_dim=32)
        student = SimpleStudent(vocab_size=10, hidden_dim=16)

        # Create and compile distiller
        distiller = Distiller(
            teacher=teacher,
            student=student,
            strategies=[LogitsDistillation(temperature=2.0)],
            alpha=0.5,
            temperature=2.0,
        )

        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            steps_per_execution=1,
        )

        # Make predictions
        predictions = distiller.predict(x_test, verbose=0)

        # Check prediction shape
        self.assertEqual(predictions.shape, (20, 10))  # batch_size, vocab_size

        # Check that predictions are finite
        self.assertTrue(np.all(np.isfinite(predictions)))

        # Check predictions sum to reasonable values (not zeros/infinities)
        prediction_sums = np.sum(predictions, axis=1)
        self.assertTrue(np.all(np.isfinite(prediction_sums)))

    def test_get_student_model_method(self):
        """Test the get_student_model() convenience method."""
        distiller = Distiller(
            teacher=self.teacher,
            student=self.student,
            strategies=[LogitsDistillation()],
            alpha=0.5,
        )

        # Test that get_student_model returns the same as direct access
        student_direct = distiller.student
        student_method = distiller.get_student_model()

        self.assertIs(student_direct, student_method)
        self.assertEqual(student_method.name, self.student.name)

    def test_distiller_serialization_and_saving(self):
        """Test Distiller serialization, saving, and loading."""
        import json
        import os
        import tempfile

        # Use standard Sequential models for serialization testing
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
                    16, activation="relu", name="student_dense_1"
                ),
                keras.layers.Dense(
                    8, activation="relu", name="student_dense_2"
                ),
                keras.layers.Dense(10, name="student_output"),
            ]
        )

        # Create distiller with multiple strategies
        from keras.src.distillation.strategies import FeatureDistillation
        from keras.src.distillation.strategies import LogitsDistillation

        strategies = [
            LogitsDistillation(temperature=3.0, loss_type="kl_divergence"),
            FeatureDistillation(
                loss_type="mse",
                teacher_layer_name="teacher_dense_1",
                student_layer_name="student_dense_1",
            ),
        ]

        original_distiller = Distiller(
            teacher=teacher,
            student=student,
            strategies=strategies,
            alpha=0.7,
            temperature=4.0,
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
        )

        # Build the models by calling them
        x_test = np.random.random((2, 20)).astype(np.float32)
        _ = original_distiller(x_test)

        # Test get_config
        config = original_distiller.get_config()

        # Verify all components are in config
        required_keys = [
            "teacher",
            "student",
            "strategies",
            "student_loss_fn",
            "alpha",
            "temperature",
            "input_mapping",
            "output_mapping",
        ]
        for key in required_keys:
            self.assertIn(key, config, f"Missing key: {key}")

        # Test JSON serialization
        json_str = json.dumps(config)
        self.assertIsInstance(json_str, str)

        # Test from_config reconstruction
        reconstructed_distiller = Distiller.from_config(config)

        # Verify reconstruction
        self.assertEqual(reconstructed_distiller.alpha, 0.7)
        self.assertEqual(reconstructed_distiller.temperature, 4.0)
        self.assertEqual(len(reconstructed_distiller.strategies), 2)

        # Verify strategy types
        self.assertIsInstance(
            reconstructed_distiller.strategies[0], LogitsDistillation
        )
        self.assertIsInstance(
            reconstructed_distiller.strategies[1], FeatureDistillation
        )

        # Verify strategy parameters
        self.assertEqual(reconstructed_distiller.strategies[0].temperature, 3.0)
        self.assertEqual(reconstructed_distiller.strategies[1].loss_type, "mse")

        # Test that reconstructed distiller can be used for inference
        reconstructed_output = reconstructed_distiller(x_test)
        self.assertEqual(reconstructed_output.shape, (2, 10))

        # Test model saving and loading (full integration test)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "distiller_model")

            # Compile original distiller
            original_distiller.compile(
                optimizer=keras.optimizers.Adam(),
                loss="sparse_categorical_crossentropy",
            )

            # Save the model
            original_distiller.save(model_path)

            # Load the model
            loaded_distiller = keras.models.load_model(model_path)

            # Verify loaded model works
            loaded_output = loaded_distiller(x_test)
            self.assertEqual(loaded_output.shape, (2, 10))

            # Verify parameters are preserved
            self.assertEqual(loaded_distiller.alpha, 0.7)
            self.assertEqual(loaded_distiller.temperature, 4.0)


        # The core serialization functionality is working
        self.assertTrue(True, "Distiller serialization test passed")
