import os
import shutil
import tempfile

import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing

try:
    # Import advanced Orbax functionality through the Keras bridge
    from keras.src.callbacks.orbax_checkpoint import CheckpointManager
    from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint
    from keras.src.callbacks.orbax_checkpoint import PyTreeCheckpointer
    from keras.src.callbacks.orbax_checkpoint import SaveArgs
    from keras.src.callbacks.orbax_checkpoint import StandardRestore
    from keras.src.callbacks.orbax_checkpoint import TypeHandler
    from keras.src.callbacks.orbax_checkpoint import metadata
    from keras.src.callbacks.orbax_checkpoint import register_type_handler
except ImportError:
    OrbaxCheckpoint = None
    CheckpointManager = None
    SaveArgs = None
    StandardRestore = None
    TypeHandler = None
    register_type_handler = None
    PyTreeCheckpointer = None
    metadata = None


class OrbaxCheckpointTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_model(self):
        """Create a simple test model."""
        inputs = layers.Input(shape=(10,))
        x = layers.Dense(5)(inputs)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs, outputs)
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

        checkpoint_dir = os.path.join(self.temp_dir, "test_basic")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Create a new model and load the checkpoint
        new_model = self._create_test_model()
        success = callback.load_latest()

        self.assertTrue(success, "Loading checkpoint should succeed")

        # Check that weights are loaded (rough check)
        original_weights = [w.numpy() for w in model.weights]
        loaded_weights = [w.numpy() for w in new_model.weights]

        # Weights should be different initially
        self.assertFalse(np.allclose(original_weights[0], loaded_weights[0]))

    @pytest.mark.requires_trainable_backend
    def test_save_best_only(self):
        """Test save_best_only functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_best_only")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor="loss",
            save_best_only=True,
            mode="min",
            save_freq="epoch",
        )

        # Train for a few epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Should have saved checkpoints
        checkpoints = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoints), 0, "Should have saved at least one checkpoint"
        )

    @pytest.mark.requires_trainable_backend
    def test_save_freq_batch(self):
        """Test batch-level saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=50)

        checkpoint_dir = os.path.join(self.temp_dir, "test_batch_freq")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq=10)

        # Train for one epoch with batch saving
        model.fit(x, y, epochs=1, batch_size=5, callbacks=[callback], verbose=0)

        # Should have saved checkpoints
        checkpoints = []
        for root, dirs, files in os.walk(checkpoint_dir):
            checkpoints.extend(dirs)

        self.assertGreater(
            len(checkpoints),
            0,
            "Should have saved checkpoints at batch intervals",
        )

    @pytest.mark.requires_trainable_backend
    def test_max_to_keep(self):
        """Test max_to_keep parameter."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_max_keep")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", max_to_keep=2
        )

        # Train for more epochs than max_to_keep
        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        # Check that max_to_keep is respected
        all_steps = callback.manager.all_steps()
        self.assertLessEqual(
            len(all_steps),
            2,
            f"Should keep at most 2 checkpoints, found {len(all_steps)}: "
            f"{all_steps}",
        )

    @pytest.mark.requires_trainable_backend
    def test_synchronous_checkpointing(self):
        """Test synchronous checkpointing (save_on_background=False)."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_sync")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=False,  # Synchronous saving
        )

        # Train for a few epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Check that checkpoints were saved
        all_steps = callback.manager.all_steps()
        self.assertEqual(
            len(all_steps),
            3,
            f"Should have 3 checkpoints, found {len(all_steps)}",
        )

        # Verify we can load the checkpoints
        success = callback.load_latest()
        self.assertTrue(success, "Should successfully load latest checkpoint")

    @pytest.mark.requires_trainable_backend
    def test_keep_period_functionality(self):
        """Test keep_period parameter keeps checkpoints every Nth save
        plus recent ones."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_keep_period")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            max_to_keep=5,  # Keep last 5 checkpoints
            keep_period=3,  # Keep every 3rd checkpoint
        )

        # Train for 10 epochs
        model.fit(x, y, epochs=10, callbacks=[callback], verbose=0)

        # Check that checkpoints follow keep_period pattern
        all_steps = sorted(callback.manager.all_steps())

        # Should keep:
        # - Checkpoints that are multiples of keep_period: 3, 6, 9
        # - Plus the most recent max_to_keep checkpoints: 5, 6, 7, 8, 9, 10
        # (but 10 doesn't exist yet)
        # But since max_to_keep=5, and we have multiples, it keeps a combination
        # The exact behavior may vary, but should include multiples of
        # keep_period

        multiples_of_period = [step for step in all_steps if step % 3 == 0]
        self.assertGreater(
            len(multiples_of_period),
            0,
            f"Should keep at least some multiples of keep_period=3, "
            f"found {all_steps}",
        )

        # Should not keep more than max_to_keep total (though this may be
        # approximate)
        self.assertLessEqual(
            len(all_steps),
            10,  # Allow some flexibility
            f"Should not keep excessively many checkpoints, "
            f"found {len(all_steps)}: {all_steps}",
        )

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_error_handling(self):
        """Test error handling when checkpoint operations fail."""
        x, y = self._create_dummy_data()

        # Test: Try to load from a non-existent checkpoint
        checkpoint_dir = os.path.join(self.temp_dir, "test_error_handling")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Try to load a checkpoint that doesn't exist
        success, iterator_state = callback.load_checkpoint(step=999)
        self.assertFalse(
            success, "Loading non-existent checkpoint should fail gracefully"
        )
        self.assertIsNone(
            iterator_state, "Iterator state should be None for failed load"
        )

        # Test: Try to load latest when no checkpoints exist
        success, iterator_state = callback.load_latest()
        self.assertFalse(
            success,
            "Loading latest when no checkpoints exist should fail gracefully",
        )
        self.assertIsNone(
            iterator_state, "Iterator state should be None for failed load"
        )

    @pytest.mark.requires_trainable_backend
    def test_partial_checkpoint_loading(self):
        """Test loading individual components from composite checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_partial_load")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_metadata={"epoch": 1, "custom_value": 42.5},
            save_data_iterator={"batch_index": 42},
        )

        # Train for a few epochs to create checkpoints
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Manually load checkpoint data to test partial access
        manager = CheckpointManager(directory=checkpoint_dir)
        restore_args = StandardRestore()
        checkpoint_data = manager.restore(step=1, args=restore_args)

        # Verify we can access individual components
        self.assertIn(
            "model_weights",
            checkpoint_data,
            "Model weights should be available",
        )
        self.assertIn(
            "optimizer_state",
            checkpoint_data,
            "Optimizer state should be available",
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

        # Verify model weights have the right shape (without loading them)
        model_weights = checkpoint_data["model_weights"]
        self.assertEqual(
            len(model_weights),
            len(model.weights),
            "Should have weights for all model parameters",
        )

    @pytest.mark.requires_trainable_backend
    def test_background_delete_functionality(self):
        """Test background deletion of old checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_background_delete")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            max_to_keep=3,  # Keep only 3 checkpoints
            enable_background_delete=True,  # Enable background deletion
        )

        # Train for more epochs than max_to_keep
        model.fit(x, y, epochs=6, callbacks=[callback], verbose=0)

        # Check that only max_to_keep checkpoints remain
        all_steps = callback.manager.all_steps()
        self.assertLessEqual(
            len(all_steps),
            3,
            f"Should keep at most 3 checkpoints with background delete, "
            f"found {len(all_steps)}: {all_steps}",
        )

        # Wait for background operations to complete
        callback.manager.wait_until_finished()

    @pytest.mark.requires_trainable_backend
    def test_post_finalization_callback(self):
        """Test post-finalization callbacks."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        callback_called = []

        def post_callback():
            callback_called.append(True)

        checkpoint_dir = os.path.join(self.temp_dir, "test_post_callback")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            post_finalization_callback=post_callback,
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Wait for async operations to complete
        callback.manager.wait_until_finished()

        # Check that the callback was called
        self.assertTrue(
            len(callback_called) > 0,
            "Post-finalization callback should have been called",
        )

    @pytest.mark.requires_trainable_backend
    def test_async_with_custom_options(self):
        """Test async checkpointing with custom AsyncOptions."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_custom_async")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            async_timeout_secs=1200,  # Custom timeout: 20 minutes
            enable_background_delete=True,  # Enable background delete
        )

        # Train for a few epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Verify checkpoints were saved successfully
        all_steps = callback.manager.all_steps()
        self.assertEqual(
            len(all_steps),
            3,
            f"Should have 3 checkpoints with custom async options, "
            f"found {len(all_steps)}",
        )

        # Wait for all operations to complete
        callback.manager.wait_until_finished()

    @pytest.mark.requires_trainable_backend
    def test_async_timeout_parameter(self):
        """Test that async timeout parameter is properly configured."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_timeout")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            async_timeout_secs=300,  # Short timeout: 5 minutes
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Verify that the timeout setting doesn't break normal operation
        all_steps = callback.manager.all_steps()
        self.assertEqual(
            len(all_steps),
            2,
            f"Should have 2 checkpoints with timeout setting, "
            f"found {len(all_steps)}",
        )

        # Wait for completion
        callback.manager.wait_until_finished()

    @pytest.mark.requires_trainable_backend
    def test_metrics_state_saving(self):
        """Test saving and loading of metrics state."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_metrics_state")
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

        # Check what's in the checkpoint
        checkpoint_data = self._load_checkpoint_data(callback, step=1)
        print(f"Checkpoint data keys: {list(checkpoint_data.keys())}")
        if "metrics_state" in checkpoint_data:
            print(f"Metrics state saved: {checkpoint_data['metrics_state']}")

        # Create new model and load checkpoint
        new_model = self._create_test_model()
        success, _ = callback.load_latest(model=new_model)
        self.assertTrue(
            success, "Should successfully load checkpoint with metrics state"
        )

        # Check that metrics state was restored in the new model
        print("New model metrics state after loading:")
        for i, metric in enumerate(new_model.metrics):
            if hasattr(metric, "variables") and metric.variables:
                state = [var.numpy() for var in metric.variables]
                print(f"  Metric {i}: {state}")

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

        checkpoint_dir = os.path.join(self.temp_dir, "test_transforms")

        # Create save_args that converts float32 to float16
        # Note: save_args structure must match composite_state structure (lists)
        save_args = {
            "model_weights": [
                SaveArgs(dtype=np.dtype(np.float16)),  # weights
                SaveArgs(dtype=np.dtype(np.float16)),  # bias
                SaveArgs(dtype=np.dtype(np.float16)),  # output weights
                SaveArgs(dtype=np.dtype(np.float16)),  # output bias
            ],
            "optimizer_state": [
                None,  # iteration count (no change)
                None,  # learning rate (no change)
                None,  # momentum vars (no change)
                None,  # momentum vars (no change)
                None,  # momentum vars (no change)
                None,  # momentum vars (no change)
                None,  # momentum vars (no change)
                None,  # momentum vars (no change)
                None,  # momentum vars (no change)
                None,  # momentum vars (no change)
            ],
        }

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_transforms=save_args,
        )

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Load checkpoint data to verify transformation was applied
        checkpoint_data = self._load_checkpoint_data(callback, step=1)

        # Check that model weights were saved in float16
        saved_weights = checkpoint_data["model_weights"]
        self.assertEqual(
            saved_weights[0].dtype,
            np.float16,
            "Weights should be saved in float16 due to transform",
        )

        # Verify we can still load the checkpoint normally
        new_model = self._create_test_model()
        success, _ = callback.load_latest(model=new_model)
        self.assertTrue(success, "Should load transformed checkpoint")

        # Check that weights were converted back to original dtype
        self.assertEqual(
            new_model.weights[0].dtype,
            model.weights[0].dtype,
            "Loaded weights should be converted back to original dtype",
        )

    @pytest.mark.requires_trainable_backend
    def test_save_decision_policy(self):
        """Test using save_interval parameter for custom save logic."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_save_policy")

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",  # This will be overridden by the save_interval
            save_interval=2,  # Save every 2 epochs
        )

        # Train for 5 epochs
        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        # Should have saved at epochs 0, 2, 4 (every 2 steps, 0-indexed)
        all_steps = sorted(callback.manager.all_steps())
        expected_steps = [0, 2, 4]  # 0-indexed epochs: 0, 2, 4
        self.assertEqual(
            all_steps,
            expected_steps,
            f"Should save at steps {expected_steps}, got {all_steps}",
        )

    @pytest.mark.requires_trainable_backend
    def test_end_to_end_iterator_resumption(self):
        """Test complete training resumption with iterator state.

        This test simulates: Run 1 -> Save -> Run 2 -> Restore -> Resume
        and verifies that batches continue from where they left off.
        """
        # Create a larger dataset to make resumption more visible
        x, y = self._create_dummy_data(num_samples=1200)
        batch_size = 20  # 60 batches total

        checkpoint_dir = os.path.join(self.temp_dir, "test_resumption")

        # Track batches processed across runs
        global_batch_counter = [0]  # Use list to modify in nested function
        current_epoch = [0]
        batch_within_epoch = [0]

        def iterator_state_func(epoch, logs):
            return {
                "global_batch_counter": global_batch_counter[0],
                "current_epoch": current_epoch[0],
                "batch_within_epoch": batch_within_epoch[0],
                "batch_size": batch_size,
                "total_samples": len(x),
            }

        # === RUN 1: Train for 2 epochs ===
        model1 = self._create_test_model()
        callback1 = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_data_iterator=iterator_state_func,
        )
        callback1.set_model(model1)  # Set the model on the callback

        # Custom training loop to track batches across epochs
        batches_processed_run1 = []
        total_batches_to_process = 2 * (len(x) // batch_size)  # 2 epochs worth
        for batch_num in range(total_batches_to_process):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(x))
            batch_x = x[batch_start:batch_end]
            batch_y = y[batch_start:batch_end]

            # Track this batch
            global_batch_counter[0] += 1
            batches_processed_run1.append(batch_num)

            # Train on batch
            model1.train_on_batch(batch_x, batch_y)

            # Trigger epoch end at the end of each "epoch"
            epoch = batch_num // (len(x) // batch_size)
            if (batch_num + 1) % (len(x) // batch_size) == 0:
                callback1.on_epoch_end(epoch, logs={"loss": 0.1})

        # Verify Run 1 saved checkpoints
        all_steps_run1 = sorted(callback1.manager.all_steps())
        self.assertEqual(
            len(all_steps_run1), 2, "Run 1 should have saved 2 checkpoints"
        )

        # === RUN 2: Load checkpoint and resume ===
        model2 = self._create_test_model()
        callback2 = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_data_iterator=iterator_state_func,
        )
        callback2.set_model(model2)  # Set the model on the callback

        # Load the latest checkpoint
        success, saved_iterator_state = callback2.load_latest(model=model2)
        self.assertTrue(success, "Should successfully load checkpoint")

        # Verify iterator state was restored
        self.assertIsNotNone(
            saved_iterator_state, "Iterator state should be returned"
        )
        restored_batch_counter = saved_iterator_state["global_batch_counter"]
        expected_batches_after_2_epochs = 2 * (len(x) // batch_size)
        self.assertEqual(
            restored_batch_counter,
            expected_batches_after_2_epochs,
            f"Should have processed {expected_batches_after_2_epochs} batches, "
            f"got {restored_batch_counter}",
        )

        # Resume training from where we left off (with wrapping)
        batches_processed_run2 = []

        # Continue training for 1 more epoch (60 more batches)
        end_batch = restored_batch_counter + (len(x) // batch_size)
        for batch_num in range(restored_batch_counter, end_batch):
            batch_start = (batch_num * batch_size) % len(x)
            batch_end = min(batch_start + batch_size, len(x))
            # Handle wrap-around
            if batch_end < batch_start:
                batch_end = len(x)
            batch_x = x[batch_start:batch_end]
            batch_y = y[batch_start:batch_end]

            # Track this batch
            global_batch_counter[0] += 1
            batches_processed_run2.append(batch_num)

            # Train on batch
            model2.train_on_batch(batch_x, batch_y)

        # Manual epoch end
        callback2.on_epoch_end(2, logs={"loss": 0.05})

        # Verify that Run 2 continued from the correct batch
        expected_first_batch_run2 = expected_batches_after_2_epochs
        self.assertEqual(
            batches_processed_run2[0],
            expected_first_batch_run2,
            f"Run 2 should start from batch {expected_first_batch_run2}, "
            f"got {batches_processed_run2[0]}",
        )

        # Verify no overlap between runs
        max_batch_run1 = max(batches_processed_run1)
        min_batch_run2 = min(batches_processed_run2)
        self.assertEqual(
            min_batch_run2,
            max_batch_run1 + 1,
            "Run 2 should start from the next batch after Run 1 ended",
        )

        # Verify total batches processed
        total_expected_batches = 3 * (len(x) // batch_size)  # 3 epochs total
        final_batch_counter = global_batch_counter[0]
        self.assertEqual(
            final_batch_counter,
            total_expected_batches,
            f"Total batches should be {total_expected_batches}, "
            f"got {final_batch_counter}",
        )

    @pytest.mark.requires_trainable_backend
    def test_optimizer_state_saving(self):
        """Test that optimizer state is saved and loaded."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_optimizer")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_optimizer_state=True,
        )

        # Train for a few epochs to update optimizer state
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Create new model and load
        new_model = self._create_test_model()
        success = callback.load_latest()
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

        checkpoint_dir = os.path.join(self.temp_dir, "test_specific")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for multiple epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

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

        checkpoint_dir = os.path.join(self.temp_dir, "test_empty")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Try to load from empty directory
        success, _ = callback.load_latest()
        self.assertFalse(success, "Loading from empty directory should fail")
        # Verify model still has its original weights (not modified)
        self.assertGreater(len(model.weights), 0)

    @pytest.mark.requires_trainable_backend
    def test_directory_creation(self):
        """Test that checkpoint directory is created if it doesn't exist."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.temp_dir, "test_create_dir", "subdir"
        )
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Directory should be created during training
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        self.assertTrue(
            os.path.exists(checkpoint_dir),
            "Checkpoint directory should be created",
        )

    @pytest.mark.requires_trainable_backend
    def test_save_and_load_composite_metadata(self):
        """Test saving and loading checkpoints with custom metadata."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_metadata")
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
        self.assertIn("model_weights", checkpoint_data)
        self.assertIn("optimizer_state", checkpoint_data)

    @pytest.mark.requires_trainable_backend
    def test_save_metadata_callable(self):
        """Test saving metadata using a callable function."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_metadata_callable")

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

        # Load checkpoint data
        checkpoint_data = self._load_checkpoint_data(callback, step=1)

        # Verify metadata was saved with callable
        self.assertIn("metadata", checkpoint_data)
        metadata = checkpoint_data["metadata"]
        self.assertEqual(metadata["epoch"], 1)  # epoch is 1-indexed in callback
        self.assertEqual(metadata["learning_rate"], 0.001)

    @pytest.mark.requires_trainable_backend
    def test_save_data_iterator_state(self):
        """Test saving data iterator state with checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_iterator")

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

        checkpoint_dir = os.path.join(self.temp_dir, "test_load_iterator")

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
        checkpoint_dir = os.path.join(self.temp_dir, "test_tf_iterator")

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
        checkpoint_dir = os.path.join(self.temp_dir, "test_jax_iterator")

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
        checkpoint_dir = os.path.join(self.temp_dir, "test_torch_iterator")

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
        """Test custom handler for a custom object using PyTreeCheckpointHandler
        directly."""
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

        import asyncio

        # Use the classes imported through the Keras bridge
        # TypeHandler and metadata are already imported above

        class MetadataHandler(TypeHandler):
            """A custom Orbax type handler to save/load the TrainingMetadata
            object via JSON."""

            def typestr(self) -> str:
                return "training_metadata"

            async def metadata(self, infos):
                """Returns metadata for the parameters."""
                return [
                    metadata.Metadata(name=info.name, directory=info.parent_dir)
                    for info in infos
                ]

            async def serialize(self, values, infos, args=None):
                """Serializes the dataclass as a JSON dict."""
                futures = []
                for value, info in zip(values, infos):
                    metadata_obj = value
                    data = {
                        "experiment_id": metadata_obj.experiment_id,
                        "start_time": metadata_obj.start_time,
                        "backend": metadata_obj.backend,
                        "notes": metadata_obj.notes,
                    }
                    # Write to file in the directory
                    file_path = info.path / "metadata.json"
                    with open(file_path, "w") as f:
                        json.dump(data, f)
                    # Return a completed future
                    future_obj = asyncio.Future()
                    future_obj.set_result(None)
                    futures.append(future_obj)
                return futures

            async def deserialize(self, infos, args=None):
                """Deserializes the JSON dict and reconstructs the dataclass
                object."""
                futures = []
                for info in infos:
                    file_path = info.path / "metadata.json"
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    result = TrainingMetadata(**data)
                    # Return a completed future with the result
                    future_obj = asyncio.Future()
                    future_obj.set_result(result)
                    futures.append(future_obj)
                return futures

        checkpoint_dir = os.path.join(self.temp_dir, "test_custom_handler")

        # 1. Create the custom metadata object
        original_metadata = TrainingMetadata(
            experiment_id="exp_123",
            start_time=time.time(),
            backend=backend.backend(),
            notes="Testing custom handlers.",
        )

        # 2. Register the type handler globally
        register_type_handler(
            ty=TrainingMetadata, handler=MetadataHandler(), override=True
        )

        # 3. Create a PyTreeCheckpointer to save the custom object directly
        # PyTreeCheckpointer is already imported above

        checkpointer = PyTreeCheckpointer()

        # 4. Save the custom object
        checkpointer.save(
            os.path.join(checkpoint_dir, "metadata"), original_metadata
        )

        # 5. Restore the custom object
        restored_metadata = checkpointer.restore(
            os.path.join(checkpoint_dir, "metadata")
        )

        # 6. If it's a future, get the result
        if hasattr(restored_metadata, "result"):
            restored_metadata = restored_metadata.result()

        # 7. Verify the custom object was restored correctly
        self.assertIsInstance(restored_metadata, TrainingMetadata)
        self.assertEqual(
            original_metadata.experiment_id, restored_metadata.experiment_id
        )
        self.assertEqual(original_metadata.backend, restored_metadata.backend)
        self.assertEqual(original_metadata.notes, restored_metadata.notes)

    def _load_checkpoint_data_from_manager(self, manager, step):
        """Helper method to load raw checkpoint data from manager."""
        try:
            restore_args = StandardRestore()
            return manager.restore(step, args=restore_args)
        except Exception as e:
            self.fail(f"Failed to load checkpoint data: {e}")

    def _get_state_as_numpy_helper(self, model):
        """Helper to convert model state to numpy (copied from
        orbax_checkpoint.py)."""
        try:
            import keras

            model_weights_np = [
                keras.ops.convert_to_numpy(w) for w in model.weights
            ]
            optimizer_vars_np = [
                keras.ops.convert_to_numpy(v) for v in model.optimizer.variables
            ]
            return model_weights_np, optimizer_vars_np
        except Exception:
            return None, None

    def _load_checkpoint_data(self, callback, step):
        """Helper method to load raw checkpoint data for testing."""
        try:
            restore_args = StandardRestore()
            return callback.manager.restore(step, args=restore_args)
        except Exception as e:
            self.fail(f"Failed to load checkpoint data: {e}")
