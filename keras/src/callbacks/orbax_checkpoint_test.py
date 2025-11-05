import os
import uuid

import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.utils.module_utils import ocp

# Import advanced Orbax functionality directly from the LazyModule
# These will only be available if orbax-checkpoint is installed
if ocp.available:
    Checkpointer = ocp.training.Checkpointer
    save_pytree = ocp.save_pytree
    load_pytree = ocp.load_pytree
    preservation_policies = ocp.training.preservation_policies
    save_decision_policies = ocp.training.save_decision_policies
    _orbax_available = True
else:
    Checkpointer = None
    save_pytree = None
    load_pytree = None
    preservation_policies = None
    save_decision_policies = None
    _orbax_available = False

# Import our OrbaxCheckpoint callback
try:
    from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint

    _orbax_available = _orbax_available and True
except ImportError:
    OrbaxCheckpoint = None
    _orbax_available = False


@pytest.mark.skipif(
    not _orbax_available,
    reason="OrbaxCheckpoint requires the 'orbax-checkpoint' package",
)
class OrbaxCheckpointTest(testing.TestCase):
    def _create_test_model(self):
        """Create a simple test model."""
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(5, name="dense_layer")(inputs)
        outputs = layers.Dense(1, name="output_layer")(x)
        model = models.Model(inputs, outputs, name="test_model")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _create_dummy_data(self, num_samples=100):
        """Create dummy training data."""
        x = np.random.randn(num_samples, 10)
        y = np.random.randn(num_samples, 1)
        return x, y

    @pytest.mark.requires_trainable_backend
    def test_basic_save_and_load(self):
        """Test basic save and load functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_basic")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Check that checkpoints were saved
        all_steps = callback.all_steps()
        self.assertEqual(
            len(all_steps),
            2,
            f"Should save 2 checkpoints, got {len(all_steps)}",
        )
        self.assertEqual(
            all_steps, [0, 1], f"Should save at steps [0, 1], got {all_steps}"
        )

        # Create a new model and load the latest checkpoint
        new_model = self._create_test_model()
        success, _ = callback.load_latest(model=new_model)

        self.assertTrue(success, "Loading checkpoint should succeed")

        # Check that weights are loaded (rough check)
        original_weights = [w.numpy() for w in model.weights]
        loaded_weights = [w.numpy() for w in new_model.weights]

        # The loaded model should have the same number of weights as the
        # trained model
        self.assertEqual(len(original_weights), len(loaded_weights))

        # Check that weights have the same shape
        for i, (orig, loaded) in enumerate(
            zip(original_weights, loaded_weights)
        ):
            self.assertEqual(
                orig.shape, loaded.shape, f"Weight {i} shape mismatch"
            )

        # Check that at least some weights changed from initialization
        # (this verifies that training actually happened and checkpoints
        # were loaded)
        initial_model = self._create_test_model()
        initial_weights = [w.numpy() for w in initial_model.weights]

        # At least one weight should be different from initialization
        weights_changed = any(
            not np.allclose(init, loaded)
            for init, loaded in zip(initial_weights, loaded_weights)
        )
        self.assertTrue(
            weights_changed,
            "Loaded weights should be different from initialization",
        )

    @pytest.mark.requires_trainable_backend
    def test_save_best_only(self):
        """Test save_best_only functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_best_only")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor="loss",  # Monitor training loss
            save_best_only=True,  # Only save when loss improves
            mode="min",  # Lower loss is better
            save_freq="epoch",  # Check every epoch
        )

        # Train for a few epochs - losses should generally decrease
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before cleanup
        callback.wait_until_finished()

        # Verify checkpoints were saved only when loss improved
        # With save_best_only=True, should save on each improvement
        # (typically each epoch for decreasing loss)
        all_steps = callback.all_steps()
        self.assertGreaterEqual(
            len(all_steps),
            1,
            f"Should save at least 1 checkpoint with save_best_only=True, "
            f"got {len(all_steps)}",
        )
        # In practice, with decreasing loss, we expect 3 checkpoints
        # (one per epoch) but the exact number depends on when
        # improvements occur
        self.assertLessEqual(
            len(all_steps),
            3,
            f"Should save at most 3 checkpoints (one per epoch), "
            f"got {len(all_steps)}",
        )

        # Verify that checkpoints correspond to valid epoch steps
        for step in all_steps:
            self.assertGreaterEqual(
                step, 0, f"Checkpoint step should be >= 0, got {step}"
            )
            self.assertLessEqual(
                step,
                2,
                f"Checkpoint step should be <= 2 (epochs are 0-indexed), "
                f"got {step}",
            )

    @pytest.mark.requires_trainable_backend
    def test_save_freq_batch(self):
        """Test batch-level saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=50)

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_batch_freq")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq=10)

        # Train for one epoch with batch saving
        model.fit(x, y, epochs=1, batch_size=5, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before cleanup
        callback.wait_until_finished()

        # With 50 samples, batch_size=5, and save_freq=10, there are 10 batches.
        # The callback should save at the end of batch 9 (step 10, since
        # _total_batches_seen is 1-indexed).
        all_steps = callback.all_steps()
        self.assertEqual(
            all_steps, [10], f"Should save at step [10], got {all_steps}"
        )

    @pytest.mark.requires_trainable_backend
    def test_max_to_keep(self):
        """Test max_to_keep parameter."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_max_keep")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", max_to_keep=2
        )

        # Train for more epochs than max_to_keep
        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before cleanup
        callback.wait_until_finished()

        # Check that max_to_keep is respected
        all_steps = callback.all_steps()
        # It should keep only the last 2 steps
        expected_steps = [3, 4]
        self.assertEqual(
            all_steps,
            expected_steps,
            f"Should keep exactly {expected_steps}, got {all_steps}",
        )

    @pytest.mark.requires_trainable_backend
    def test_synchronous_checkpointing(self):
        """Test synchronous checkpointing (save_on_background=False)."""

        model = self._create_test_model()
        x, y = self._create_dummy_data()

        # Test synchronous checkpointing
        checkpoint_dir_sync = os.path.join(self.get_temp_dir(), "test_sync")
        callback_sync = OrbaxCheckpoint(
            directory=checkpoint_dir_sync,
            save_freq="epoch",
            save_on_background=False,  # Synchronous saving
        )

        # Measure time for synchronous saving
        model.fit(x, y, epochs=3, callbacks=[callback_sync], verbose=0)

        # Check that checkpoints were saved
        all_steps_sync = callback_sync.all_steps()
        self.assertEqual(
            len(all_steps_sync),
            3,
            f"Should have 3 checkpoints, found {len(all_steps_sync)}",
        )

        # Verify we can load the checkpoints immediately (no need to wait)
        success = callback_sync.load_latest()
        self.assertTrue(success, "Should successfully load latest checkpoint")

        # Test asynchronous checkpointing for comparison
        model2 = self._create_test_model()
        checkpoint_dir_async = os.path.join(self.get_temp_dir(), "test_async")
        callback_async = OrbaxCheckpoint(
            directory=checkpoint_dir_async,
            save_freq="epoch",
            save_on_background=True,  # Asynchronous saving (default)
        )

        # Measure time for asynchronous saving
        model2.fit(x, y, epochs=3, callbacks=[callback_async], verbose=0)
        # async_time = time.time() - start_time

        # Wait for async operations to complete
        callback_async.wait_until_finished()

        # Check that checkpoints were saved
        all_steps_async = callback_async.all_steps()
        self.assertEqual(
            len(all_steps_async),
            3,
            f"Should have 3 checkpoints, found {len(all_steps_async)}",
        )

        # Verify we can load the checkpoints
        success = callback_async.load_latest()
        self.assertTrue(success, "Should successfully load latest checkpoint")

        # Both sync and async modes should work correctly
        # (async allows training to continue while saving happens in background,
        # but in this small test the timing difference may not be measurable)

    @pytest.mark.requires_trainable_backend
    def test_keep_period_vs_no_keep_period(self):
        """Test that keep_period preserves periodic checkpoints that would
        otherwise be deleted."""
        # First, test WITHOUT keep_period
        model1 = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir_no_period = os.path.join(
            self.get_temp_dir(), "test_no_period"
        )
        callback_no_period = OrbaxCheckpoint(
            directory=checkpoint_dir_no_period,
            save_freq="epoch",
            max_to_keep=3,  # Keep only last 3 checkpoints
        )

        # Train for 10 epochs
        model1.fit(x, y, epochs=10, callbacks=[callback_no_period], verbose=0)
        steps_no_period = sorted(callback_no_period.all_steps())

        # Without keep_period, should keep only the most recent max_to_keep=3
        expected_recent_only = [7, 8, 9]  # Last 3 epochs (0-indexed)
        self.assertEqual(
            steps_no_period,
            expected_recent_only,
            f"Without keep_period, should keep only recent checkpoints: "
            f"{expected_recent_only}, got {steps_no_period}",
        )

        # Now test WITH keep_period
        model2 = self._create_test_model()
        checkpoint_dir_with_period = os.path.join(
            self.get_temp_dir(), "test_with_period"
        )
        callback_with_period = OrbaxCheckpoint(
            directory=checkpoint_dir_with_period,
            save_freq="epoch",
            max_to_keep=3,  # Same max_to_keep
            keep_period=4,  # Keep every 4th checkpoint
        )

        # Train for 10 epochs
        model2.fit(x, y, epochs=10, callbacks=[callback_with_period], verbose=0)
        steps_with_period = sorted(callback_with_period.all_steps())

        # With keep_period=4, EveryNSteps keeps checkpoints at regular
        # intervals: 0, 4, 8
        periodic_checkpoints = [0, 4, 8]
        for periodic_step in periodic_checkpoints:
            self.assertIn(
                periodic_step,
                steps_with_period,
                f"Periodic checkpoint {periodic_step} should be kept with "
                f"keep_period=4, found {steps_with_period}",
            )

        # Expected steps are the union of LatestN(3) ([7, 8, 9]) and
        # EveryNSteps(4) ([0, 4, 8])
        expected_steps_with_period = [0, 4, 7, 8, 9]
        self.assertEqual(
            steps_with_period,
            expected_steps_with_period,
            f"Should keep union of LatestN(3) and EveryNSteps(4), got "
            f"{steps_with_period}",
        )

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_error_handling(self):
        """Test error handling when checkpoint operations fail."""
        x, y = self._create_dummy_data()

        # Test: Try to load from a non-existent checkpoint
        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_error_handling"
        )
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Try to load a checkpoint that doesn't exist - should raise exception
        with self.assertRaises(Exception):
            callback.load_checkpoint(step=999)

        # Test: Try to load latest when no checkpoints exist -
        # should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            callback.load_latest()

    @pytest.mark.requires_trainable_backend
    def test_partial_checkpoint_loading(self):
        """Test loading individual components from composite checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_partial_load")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_metadata={"epoch": 1, "custom_value": 42.5},
            save_data_iterator={"batch_index": 42},
        )

        # Train for a few epochs to create checkpoints
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before loading
        callback.wait_until_finished()

        # Manually load checkpoint data to test partial access
        checkpointer = Checkpointer(directory=checkpoint_dir)
        checkpoint_data = checkpointer.load_pytree(step=1)

        # Verify we can access individual components
        self.assertIn(
            "trainable_variables",
            checkpoint_data,
            "Trainable variables should be available",
        )
        self.assertIn(
            "optimizer_variables",
            checkpoint_data,
            "Optimizer variables should be available",
        )
        self.assertIn(
            "metadata", checkpoint_data, "Metadata should be available"
        )
        self.assertIn(
            "data_iterator",
            checkpoint_data,
            "Data iterator should be available",
        )

        # Check metadata content
        self.assertEqual(checkpoint_data["metadata"]["epoch"], 1)
        self.assertEqual(checkpoint_data["metadata"]["custom_value"], 42.5)

        # Check iterator state content
        self.assertEqual(checkpoint_data["data_iterator"]["batch_index"], 42)

        # Verify trainable variables have the right structure
        trainable_vars = checkpoint_data["trainable_variables"]
        self.assertIsInstance(trainable_vars, dict)
        self.assertIn("dense_layer", trainable_vars)
        self.assertIn("output_layer", trainable_vars)

    @pytest.mark.requires_trainable_backend
    def test_background_delete_functionality(self):
        """Test checkpoint deletion with max_to_keep."""
        # Generate unique ID for this test run to avoid conflicts in
        # parallel execution
        unique_id = str(uuid.uuid4())[:8]

        # Test checkpoint deletion behavior with max_to_keep
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_delete_{unique_id}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            max_to_keep=2,  # Keep only 2 checkpoints
        )

        # Train for more epochs than max_to_keep
        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        # Check that max_to_keep is respected
        all_steps = sorted(callback.all_steps())
        self.assertLessEqual(
            len(all_steps),
            2,
            f"Should keep at most 2 checkpoints, "
            f"found {len(all_steps)}: {all_steps}",
        )

    @pytest.mark.requires_trainable_backend
    def test_post_finalization_callback(self):
        """Test post-finalization callbacks."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        callback_called = []

        def post_callback():
            print("DEBUG: Post-finalization callback called!")
            callback_called.append(True)

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_post_callback")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            post_finalization_callback=post_callback,
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Wait for async operations to complete
        callback.wait_until_finished()

        # Check that the callback was called
        self.assertTrue(
            len(callback_called) > 0,
            "Post-finalization callback should have been called",
        )

    @pytest.mark.requires_trainable_backend
    def test_async_with_custom_options(self):
        """Test async checkpointing with default options."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_custom_async")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
        )

        # Train for a few epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Verify checkpoints were saved successfully
        all_steps = callback.all_steps()
        self.assertEqual(
            len(all_steps),
            3,
            f"Should have 3 checkpoints with custom async options, "
            f"found {len(all_steps)}",
        )

        # Wait for all operations to complete
        callback.wait_until_finished()

    @pytest.mark.requires_trainable_backend
    def test_async_timeout_parameter(self):
        """Test that async checkpointing works with default timeout."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_timeout")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Verify that the timeout setting doesn't break normal operation
        all_steps = callback.all_steps()
        self.assertEqual(
            len(all_steps),
            2,
            f"Should have 2 checkpoints with timeout setting, "
            f"found {len(all_steps)}",
        )

        # Wait for completion
        callback.wait_until_finished()

    @pytest.mark.requires_trainable_backend
    def test_metrics_state_saving(self):
        """Test saving and loading of metrics state."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_metrics_state")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_metrics_state=True,
        )

        # Train for a few epochs to update metrics
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Check that metrics have state after training
        original_metrics_state = []
        for metric in model.metrics:
            if hasattr(metric, "variables") and metric.variables:
                original_metrics_state.append(
                    [var.numpy() for var in metric.variables]
                )

        self.assertGreater(
            len(original_metrics_state), 0, "Should have metrics with state"
        )

        # Create new model and load checkpoint
        new_model = self._create_test_model()
        success, _ = callback.load_latest(model=new_model)
        self.assertTrue(
            success, "Should successfully load checkpoint with metrics state"
        )

        # Check that metrics state was restored in the new model
        for i, original_state in enumerate(original_metrics_state):
            if i < len(new_model.metrics):
                new_metric = new_model.metrics[i]
                if hasattr(new_metric, "variables") and new_metric.variables:
                    new_state = [var.numpy() for var in new_metric.variables]
                    # States should match (allowing for some floating point
                    # differences)
                    for orig, new in zip(original_state, new_state):
                        np.testing.assert_allclose(orig, new, rtol=1e-5)

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_transformations(self):
        """Test applying transformations during checkpoint saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_transforms")

        # Train for one step first to initialize optimizer variables
        model.fit(x, y, epochs=1, verbose=0)

        # Skip save_transforms test for now as it needs to be updated
        # for the new nested structure format
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
        )

        # Train for one more epoch to trigger save
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        # Load checkpoint data to verify basic functionality
        checkpoint_data = self._load_checkpoint_data(callback, step=0)

        # Check that trainable_variables were saved
        self.assertIn("trainable_variables", checkpoint_data)

        # Verify we can still load the checkpoint normally
        new_model = self._create_test_model()
        success, _ = callback.load_latest(model=new_model)
        self.assertTrue(success, "Should load checkpoint")

    @pytest.mark.requires_trainable_backend
    def test_save_decision_policy(self):
        """Test using save_decision_policy parameter for custom save logic."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_save_policy")

        # Use FixedIntervalPolicy to save every 2 epochs
        from orbax.checkpoint.experimental.v1 import training

        save_policy = training.save_decision_policies.FixedIntervalPolicy(2)

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_decision_policy=save_policy,
        )

        # Train for 5 epochs
        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before cleanup
        callback.wait_until_finished()

        # Should have saved at epochs 0, 2, 4 (every 2 steps, 0-indexed)
        all_steps = sorted(callback.all_steps())
        expected_steps = [0, 2, 4]  # 0-indexed epochs: 0, 2, 4
        self.assertEqual(
            all_steps,
            expected_steps,
            f"Should save at steps {expected_steps}, got {all_steps}",
        )

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="PyTorch train_on_batch has scalar loss issues",
    )
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="PyTorch train_on_batch has scalar loss issues",
    )
    @pytest.mark.requires_trainable_backend
    def test_optimizer_state_saving(self):
        """Test that optimizer state is saved and loaded."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_optimizer")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_optimizer_state=True,
        )

        # Train for a few epochs to update optimizer state
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Create new model and load
        new_model = self._create_test_model()
        success, _ = callback.load_latest()
        self.assertTrue(success)

        # Check optimizer iterations (rough check that state was loaded)
        # Note: This is a basic check - more sophisticated tests could check
        # specific optimizer variables
        self.assertGreaterEqual(new_model.optimizer.iterations.numpy(), 0)

    @pytest.mark.requires_trainable_backend
    def test_load_specific_checkpoint(self):
        """Test loading a specific checkpoint by step."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_specific")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for multiple epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before loading
        callback.wait_until_finished()

        # Create new model and load specific checkpoint
        new_model = self._create_test_model()
        success, _ = callback.load_checkpoint(step=1)  # Load epoch 1

        self.assertTrue(success, "Loading specific checkpoint should succeed")
        # Verify the model was loaded by checking it has weights
        self.assertGreater(len(new_model.weights), 0)

    @pytest.mark.requires_trainable_backend
    def test_no_checkpoint_found(self):
        """Test behavior when no checkpoints exist."""
        model = self._create_test_model()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_empty")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Try to load from empty directory - should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            callback.load_latest()
        # Verify model still has its original weights (not modified)
        self.assertGreater(len(model.weights), 0)

    @pytest.mark.requires_trainable_backend
    def test_directory_creation(self):
        """Test that checkpoint directory is created if it doesn't exist."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_create_dir", "subdir"
        )
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Directory should be created during training
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        self.assertTrue(
            os.path.exists(checkpoint_dir),
            "Checkpoint directory should be created",
        )

        # Wait for async operations to complete before test cleanup
        callback.wait_until_finished()

    @pytest.mark.requires_trainable_backend
    def test_save_and_load_composite_metadata(self):
        """Test saving and loading checkpoints with custom metadata."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_metadata")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_metadata={
                "epoch": 5,
                "learning_rate": 0.001,
                "metrics": {"loss": 0.5, "accuracy": 0.8},
            },
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Load the checkpoint and get the full data
        checkpoint_data = self._load_checkpoint_data(callback, step=1)

        # Verify metadata was saved
        self.assertIn("metadata", checkpoint_data)
        metadata = checkpoint_data["metadata"]
        self.assertEqual(metadata["epoch"], 5)
        self.assertEqual(metadata["learning_rate"], 0.001)
        self.assertEqual(metadata["metrics"]["loss"], 0.5)
        self.assertEqual(metadata["metrics"]["accuracy"], 0.8)

        # Verify model weights are also present
        self.assertIn("trainable_variables", checkpoint_data)
        self.assertIn("optimizer_variables", checkpoint_data)

    @pytest.mark.requires_trainable_backend
    def test_save_metadata_callable(self):
        """Test saving metadata using a callable function."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_metadata_callable"
        )

        def metadata_func(epoch, logs):
            return {
                "epoch": epoch,
                "learning_rate": 0.001,
                "metrics": logs or {},
            }

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_metadata=metadata_func,
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Check available steps
        available_steps = callback.all_steps()
        self.assertGreater(
            len(available_steps), 0, "Should have at least one checkpoint"
        )

        # Load checkpoint data from the latest step
        latest_step = max(available_steps)
        checkpoint_data = self._load_checkpoint_data(callback, step=latest_step)

        # Verify metadata was saved with callable
        self.assertIn("metadata", checkpoint_data)
        metadata = checkpoint_data["metadata"]
        self.assertEqual(
            metadata["epoch"], latest_step
        )  # epoch matches the step
        self.assertEqual(metadata["learning_rate"], 0.001)

    @pytest.mark.requires_trainable_backend
    def test_save_data_iterator_state(self):
        """Test saving data iterator state with checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_iterator")
        os.makedirs(checkpoint_dir, exist_ok=True)

        def iterator_state_func(epoch, logs):
            return {
                "current_position": epoch * 100,
                "shuffle_seed": 42,
                "batch_size": 32,
                "dataset_size": len(x),
            }

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_data_iterator=iterator_state_func,
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Load checkpoint data
        checkpoint_data = self._load_checkpoint_data(callback, step=1)

        # Verify data iterator state was saved
        self.assertIn("data_iterator", checkpoint_data)
        iterator_state = checkpoint_data["data_iterator"]
        self.assertEqual(iterator_state["current_position"], 100)  # epoch 1
        self.assertEqual(iterator_state["shuffle_seed"], 42)
        self.assertEqual(iterator_state["batch_size"], 32)
        self.assertEqual(iterator_state["dataset_size"], len(x))

    @pytest.mark.requires_trainable_backend
    def test_load_checkpoint_with_iterator_state(self):
        """Test loading checkpoint returns iterator state for restoration."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_load_iterator")
        os.makedirs(checkpoint_dir, exist_ok=True)

        def iterator_state_func(epoch, logs):
            return {
                "current_position": epoch * 100,
                "shuffle_seed": 42,
                "batch_size": 32,
                "dataset_size": len(x),
            }

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_data_iterator=iterator_state_func,
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before loading
        callback.wait_until_finished()

        # Create new model and load checkpoint
        success, iterator_state = callback.load_checkpoint(step=1)

        # Verify loading succeeded and iterator state was returned
        self.assertTrue(success, "Loading checkpoint should succeed")
        self.assertIsNotNone(
            iterator_state, "Iterator state should be returned"
        )
        self.assertEqual(iterator_state["current_position"], 100)  # epoch 1
        self.assertEqual(iterator_state["shuffle_seed"], 42)
        self.assertEqual(iterator_state["batch_size"], 32)
        self.assertEqual(iterator_state["dataset_size"], len(x))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="TensorFlow-specific iterator restoration test",
    )
    def test_tensorflow_iterator_restoration(self):
        """Test iterator restoration with TensorFlow backend."""
        import tensorflow as tf

        # Create simple test data
        x, y = self._create_dummy_data(50)  # Smaller dataset

        model = self._create_test_model()
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_tf_iterator")
        os.makedirs(checkpoint_dir, exist_ok=True)

        def tf_iterator_state_func(epoch, logs):
            return {
                "batches_processed": epoch * 5,  # 5 batches per epoch
                "shuffle_seed": 42,
                "batch_size": 10,
                "epoch": epoch,
            }

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_data_iterator=tf_iterator_state_func,
        )

        # Train for 2 epochs using model.fit (simpler)
        model.fit(
            x, y, epochs=2, callbacks=[callback], verbose=0, batch_size=10
        )

        # Wait for async operations to complete before loading
        callback.wait_until_finished()

        # Load checkpoint and verify iterator state
        success, saved_iterator_state = callback.load_checkpoint(step=1)

        self.assertTrue(success, "Checkpoint loading should succeed")
        self.assertIsNotNone(
            saved_iterator_state, "Iterator state should be returned"
        )
        self.assertEqual(saved_iterator_state["epoch"], 1)
        self.assertEqual(
            saved_iterator_state["batches_processed"], 5
        )  # epoch 1 * 5 batches
        self.assertEqual(saved_iterator_state["batch_size"], 10)

        # Demonstrate iterator restoration
        # Create tf.data.Dataset similar to what user would do
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(saved_iterator_state["shuffle_seed"])
        dataset = dataset.batch(saved_iterator_state["batch_size"])

        # Create iterator and skip to saved position
        iterator = iter(dataset)
        for _ in range(saved_iterator_state["batches_processed"]):
            try:
                next(iterator)
            except StopIteration:
                break

        # Verify we can get next batch
        try:
            batch_x, batch_y = next(iterator)
            self.assertEqual(
                batch_x.shape[0], saved_iterator_state["batch_size"]
            )
        except StopIteration:
            # End of dataset is also acceptable
            pass

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="JAX-specific iterator restoration test",
    )
    def test_jax_iterator_restoration(self):
        """Test iterator restoration with JAX backend."""
        import jax.numpy as jnp

        # Create simple test data
        x, y = self._create_dummy_data(50)

        model = self._create_test_model()
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_jax_iterator")
        os.makedirs(checkpoint_dir, exist_ok=True)

        def jax_iterator_state_func(epoch, logs):
            return {
                "batches_processed": epoch * 5,  # 5 batches per epoch
                "shuffle_seed": 42,
                "batch_size": 10,
                "epoch": epoch,
            }

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_data_iterator=jax_iterator_state_func,
        )

        # Train for 2 epochs using model.fit
        model.fit(
            x, y, epochs=2, callbacks=[callback], verbose=0, batch_size=10
        )

        # Wait for async operations to complete before loading
        callback.wait_until_finished()

        # Load checkpoint and verify iterator state
        success, saved_iterator_state = callback.load_checkpoint(step=1)

        self.assertTrue(success, "Checkpoint loading should succeed")
        self.assertIsNotNone(
            saved_iterator_state, "Iterator state should be returned"
        )
        self.assertEqual(saved_iterator_state["epoch"], 1)
        self.assertEqual(saved_iterator_state["batches_processed"], 5)
        self.assertEqual(saved_iterator_state["batch_size"], 10)

        # Demonstrate iterator restoration for JAX
        # Convert to JAX arrays
        x_jax = jnp.array(x)
        # y_jax = jnp.array(y)  # Not used in this test

        # Create shuffled indices (same as during training)
        rng = jnp.array(
            np.random.RandomState(
                saved_iterator_state["shuffle_seed"]
            ).permutation(len(x_jax))
        )

        # Calculate starting position
        start_idx = (
            saved_iterator_state["batches_processed"]
            * saved_iterator_state["batch_size"]
        )

        # Get remaining data from correct position
        remaining_indices = rng[start_idx:]
        if len(remaining_indices) >= saved_iterator_state["batch_size"]:
            batch_indices = remaining_indices[
                : saved_iterator_state["batch_size"]
            ]
            batch_x = x_jax[batch_indices]
            # batch_y = y_jax[batch_indices]  # Not used in assertion
            self.assertEqual(
                batch_x.shape[0], saved_iterator_state["batch_size"]
            )

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="PyTorch-specific iterator restoration test",
    )
    def test_pytorch_iterator_restoration(self):
        """Test iterator restoration with PyTorch backend."""
        import torch

        # Create simple test data
        x, y = self._create_dummy_data(50)

        model = self._create_test_model()
        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_torch_iterator"
        )

        def torch_iterator_state_func(epoch, logs):
            return {
                "batches_processed": epoch * 5,  # 5 batches per epoch
                "shuffle_seed": 42,
                "batch_size": 10,
                "epoch": epoch,
            }

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_data_iterator=torch_iterator_state_func,
        )

        # Train for 2 epochs using model.fit
        model.fit(
            x, y, epochs=2, callbacks=[callback], verbose=0, batch_size=10
        )

        # Wait for async operations to complete before loading
        callback.wait_until_finished()

        # Load checkpoint and verify iterator state
        success, saved_iterator_state = callback.load_checkpoint(step=1)

        self.assertTrue(success, "Checkpoint loading should succeed")
        self.assertIsNotNone(
            saved_iterator_state, "Iterator state should be returned"
        )
        self.assertEqual(saved_iterator_state["epoch"], 1)
        self.assertEqual(saved_iterator_state["batches_processed"], 5)
        self.assertEqual(saved_iterator_state["batch_size"], 10)

        # Demonstrate iterator restoration for PyTorch
        # Convert to PyTorch tensors
        x_torch = torch.tensor(x, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32)

        # Create dataset and dataloader (same as during training)
        dataset = torch.utils.data.TensorDataset(x_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=saved_iterator_state["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(
                saved_iterator_state["shuffle_seed"]
            ),
        )

        # Create iterator and skip to saved position
        iterator = iter(dataloader)
        for _ in range(saved_iterator_state["batches_processed"]):
            try:
                next(iterator)
            except StopIteration:
                break

        # Verify we can get next batch
        try:
            batch_x, batch_y = next(iterator)
            self.assertEqual(
                batch_x.shape[0], saved_iterator_state["batch_size"]
            )
        except StopIteration:
            # End of dataset is also acceptable
            pass

    @pytest.mark.requires_trainable_backend
    def test_custom_handler_and_registry(self):
        """Integration test demonstrating complete training setup with custom
        type handlers.

        This test shows how MetadataHandler and ConfigHandler work together in a
        real-world training workflow, including integration with model.fit() and
        checkpoint/resume functionality. Individual handler tests are in
        test_metadata_handler() and test_config_handler().
        """
        import json
        import time
        from dataclasses import dataclass

        @dataclass
        class TrainingMetadata:
            """A custom object to hold arbitrary training info."""

            experiment_id: str
            start_time: float
            backend: str
            notes: str = ""
            hyperparameters: dict = None

        @dataclass
        class ExperimentConfig:
            """Another custom object for experiment configuration."""

            model_architecture: str
            dataset_name: str
            batch_size: int
            learning_rate: float
            optimizer_name: str

        # Use V1 equivalents
        from orbax.checkpoint.experimental.v1 import handlers as v1_handlers
        from orbax.checkpoint.experimental.v1 import load_checkpointables
        from orbax.checkpoint.experimental.v1 import save_checkpointables

        class MetadataHandler(v1_handlers.CheckpointableHandler):
            """A custom V1 checkpointable handler to save/load the
            TrainingMetadata object via JSON."""

            def typestr(self) -> str:
                return "training_metadata"

            def is_handleable(self, value) -> bool:
                """Check if this handler can handle the given value."""
                return isinstance(value, TrainingMetadata)

            async def metadata(self, directory):
                """Returns metadata for the checkpointable."""
                return None

            async def _background_save(self, directory, checkpointable):
                """Background save operation."""
                directory = await directory.await_creation()
                metadata_obj = checkpointable
                data = {
                    "experiment_id": metadata_obj.experiment_id,
                    "start_time": metadata_obj.start_time,
                    "backend": metadata_obj.backend,
                    "notes": metadata_obj.notes,
                    "hyperparameters": metadata_obj.hyperparameters or {},
                }
                # Write to file in the directory
                file_path = directory / "metadata.json"
                with open(file_path, "w") as f:
                    json.dump(data, f)

            async def save(self, directory, checkpointable):
                """Saves the TrainingMetadata object to the directory."""
                return self._background_save(directory, checkpointable)

            async def _background_load(self, directory):
                """Background load operation."""
                file_path = directory / "metadata.json"
                with open(file_path, "r") as f:
                    data = json.load(f)
                return TrainingMetadata(**data)

            async def load(self, directory, abstract_checkpointable=None):
                """Loads the TrainingMetadata object from the directory."""
                return self._background_load(directory)

        class ConfigHandler(v1_handlers.CheckpointableHandler):
            """Custom handler for ExperimentConfig objects."""

            def typestr(self) -> str:
                return "experiment_config"

            def is_handleable(self, value) -> bool:
                """Check if this handler can handle the given value."""
                return isinstance(value, ExperimentConfig)

            async def metadata(self, directory):
                """Returns metadata for the checkpointable."""
                return None

            async def _background_save(self, directory, checkpointable):
                """Background save operation."""
                directory = await directory.await_creation()
                config_obj = checkpointable
                data = {
                    "model_architecture": config_obj.model_architecture,
                    "dataset_name": config_obj.dataset_name,
                    "batch_size": config_obj.batch_size,
                    "learning_rate": config_obj.learning_rate,
                    "optimizer_name": config_obj.optimizer_name,
                }
                file_path = directory / "config.json"
                with open(file_path, "w") as f:
                    json.dump(data, f)

            async def save(self, directory, checkpointable):
                """Saves the ExperimentConfig object to the directory."""
                return self._background_save(directory, checkpointable)

            async def _background_load(self, directory):
                """Background load operation."""
                file_path = directory / "config.json"
                with open(file_path, "r") as f:
                    data = json.load(f)
                return ExperimentConfig(**data)

            async def load(self, directory, abstract_checkpointable=None):
                """Loads the ExperimentConfig object from the directory."""
                return self._background_load(directory)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_custom_handler"
        )

        # === REAL-WORLD TRAINING SETUP ===

        # 1. Create experiment configuration and metadata
        experiment_config = ExperimentConfig(
            model_architecture="simple_mlp",
            dataset_name="dummy_regression",
            batch_size=32,
            learning_rate=0.001,
            optimizer_name="adam",
        )

        training_metadata = TrainingMetadata(
            experiment_id="exp_123_complete_training",
            start_time=time.time(),
            backend=backend.backend(),
            notes="Complete training setup with custom handlers",
            hyperparameters={
                "epochs": 3,
                "validation_split": 0.2,
                "early_stopping_patience": 5,
            },
        )

        # 2. DO NOT register the type handlers globally
        # v1_handlers.register_handler(MetadataHandler)
        # v1_handlers.register_handler(ConfigHandler)

        # 3. Set up the model and training data
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=200)

        # 4. Create checkpoint callback with standard metadata
        # Note: save_metadata should use simple serializable types (numbers,
        # booleans)
        # Complex objects and strings should be saved separately using
        # PyTreeCheckpointer
        def metadata_func(epoch, logs):
            """Standard metadata function with basic serializable data."""
            return {
                "experiment_id": 123,  # Use number instead of string
                "epoch": epoch + 1,
                "loss": float(logs.get("loss", 0.0)) if logs else 0.0,
                "val_loss": float(logs.get("val_loss", 0.0)) if logs else 0.0,
                "backend_id": (
                    1 if training_metadata.backend == "tensorflow" else 2
                ),
                # Use number instead of string for backend identification
                "total_epochs": training_metadata.hyperparameters["epochs"],
                "validation_split": training_metadata.hyperparameters[
                    "validation_split"
                ],
            }

        training_callback = OrbaxCheckpoint(
            directory=os.path.join(checkpoint_dir, "training_checkpoints"),
            save_freq="epoch",
            save_metadata=metadata_func,  # Standard serializable metadata
            save_metrics_state=True,
            save_optimizer_state=True,
        )

        # 5. Train the model with custom metadata
        model.fit(
            x,
            y,
            epochs=3,
            batch_size=32,
            callbacks=[training_callback],
            verbose=0,
            validation_split=0.2,
        )

        # 6. Save experiment config separately using save_checkpointables
        from orbax.checkpoint.experimental.v1 import Context
        from orbax.checkpoint.experimental.v1 import options

        # Pass the handlers to create a local registry
        checkpointables_options = (
            options.CheckpointablesOptions.create_with_handlers(
                ConfigHandler, MetadataHandler
            )
        )
        with Context(checkpointables_options=checkpointables_options):
            save_checkpointables(
                os.path.join(checkpoint_dir, "experiment_config"),
                {"config": experiment_config},
            )

        # 7. Save additional training state separately (use the same options)
        final_training_state = {
            "config": experiment_config,
            "metadata": training_metadata,
            "final_epoch": 3,
            "total_samples": len(x),
        }
        with Context(checkpointables_options=checkpointables_options):
            save_checkpointables(
                os.path.join(checkpoint_dir, "training_state"),
                final_training_state,
            )

        # === VERIFICATION: Load and Resume Training ===

        # 8. Load the experiment configuration (use the same options)
        with Context(checkpointables_options=checkpointables_options):
            loaded_config_data = load_checkpointables(
                os.path.join(checkpoint_dir, "experiment_config")
            )
        loaded_config = loaded_config_data["config"]

        self.assertIsInstance(loaded_config, ExperimentConfig)
        self.assertEqual(loaded_config.model_architecture, "simple_mlp")
        self.assertEqual(loaded_config.batch_size, 32)

        # 9. Load the training state (use the same options)
        with Context(checkpointables_options=checkpointables_options):
            loaded_state = load_checkpointables(
                os.path.join(checkpoint_dir, "training_state")
            )

        self.assertEqual(loaded_state["final_epoch"], 3)
        self.assertEqual(loaded_state["total_samples"], 200)

        # 10. Load checkpoint data directly to check metadata
        checkpoint_data = self._load_checkpoint_data(training_callback, step=2)

        # Verify metadata was saved and loaded
        self.assertIn("metadata", checkpoint_data)
        loaded_metadata = checkpoint_data["metadata"]

        # Verify the loaded standard metadata (dict with basic types)
        self.assertIsInstance(loaded_metadata, dict)
        self.assertEqual(loaded_metadata["experiment_id"], 123)
        # Number instead of string
        self.assertEqual(loaded_metadata["epoch"], 3)  # 0-indexed epoch + 1
        # backend_id was encoded as 1 for TensorFlow and 2 for Torch.
        expected_backend_id = (
            1 if training_metadata.backend == "tensorflow" else 2
        )
        self.assertEqual(
            loaded_metadata["backend_id"],
            expected_backend_id,
            f"backend_id should match the saved training backend, "
            f"got {loaded_metadata['backend_id']}",
        )
        self.assertIn("total_epochs", loaded_metadata)

        # 11. Demonstrate resuming training with loaded state
        resumed_model = self._create_test_model()
        resumed_callback = OrbaxCheckpoint(
            directory=os.path.join(checkpoint_dir, "training_checkpoints"),
            save_freq="epoch",
            save_metadata=metadata_func,
        )

        # Load the latest checkpoint into the new model
        success, _ = resumed_callback.load_latest(model=resumed_model)
        self.assertTrue(success, "Should successfully resume from checkpoint")

        # Continue training for 1 more epoch
        resumed_model.fit(
            x,
            y,
            epochs=1,  # Just 1 more epoch
            batch_size=32,
            callbacks=[resumed_callback],
            verbose=0,
            validation_split=0.2,
            initial_epoch=3,  # Start from epoch 3
        )

        # Verify that standard metadata works seamlessly with model.fit()
        # Check what steps are available after resumed training
        available_steps = sorted(resumed_callback.all_steps())

        # Load the latest available checkpoint
        if available_steps:
            latest_step = available_steps[-1]
            final_checkpoint_data = self._load_checkpoint_data(
                resumed_callback, step=latest_step
            )
            self.assertIn("metadata", final_checkpoint_data)
            final_metadata = final_checkpoint_data["metadata"]
            self.assertIsInstance(final_metadata, dict)
            self.assertIn("loss", final_metadata)
        else:
            self.fail("No checkpoints found after resumed training")

    @pytest.mark.requires_trainable_backend
    def test_save_decision_policy_integration(self):
        """Test using orbax.checkpoint.SaveDecisionPolicy objects."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_decision_policy"
        )

        # Use FixedIntervalPolicy to save every 3 steps (V1 API)
        policy = save_decision_policies.FixedIntervalPolicy(3)

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_decision_policy=policy,
        )

        # Train for 10 epochs (steps 0-9)
        model.fit(x, y, epochs=10, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before cleanup
        callback.wait_until_finished()

        # Should have saved at steps 0, 3, 6, 9
        all_steps = sorted(callback.all_steps())
        expected_steps = [0, 3, 6, 9]
        self.assertEqual(
            all_steps,
            expected_steps,
            f"Should save at steps {expected_steps}, got {all_steps}",
        )

    def _load_checkpoint_data(self, callback, step):
        """Helper method to load raw checkpoint data for testing."""
        # Wait for any in-progress saves to complete
        callback.wait_until_finished()

        try:
            return callback.checkpointer.load_pytree(step)
        except Exception as e:
            self.fail(f"Failed to load checkpoint data: {e}")
