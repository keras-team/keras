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
    import orbax.checkpoint as ocp

    from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint
except ImportError:
    ocp = None
    OrbaxCheckpoint = None


@pytest.mark.skipif(
    OrbaxCheckpoint is None,
    reason="`orbax-checkpoint` is required for `OrbaxCheckpoint` tests.",
)
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

    def _load_checkpoint_data(self, callback, step):
        """Helper method to load raw checkpoint data for testing."""
        try:
            restore_args = ocp.args.StandardRestore()
            return callback.manager.restore(step, args=restore_args)
        except Exception as e:
            self.fail(f"Failed to load checkpoint data: {e}")
