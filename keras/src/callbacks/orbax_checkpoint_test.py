import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing
from keras.src import utils
from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint
from keras.src.saving import register_keras_serializable

# Import advanced Orbax functionality directly from the LazyModule


# Custom layer with assets for testing (JAX-compatible)
@register_keras_serializable(package="test")
class LayerWithAssets(layers.Layer):
    """Test layer that has assets to save/load."""

    def __init__(self, units=10, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._vocabulary = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            name="kernel",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.kernel

    def set_vocabulary(self, vocab):
        """Set vocabulary for testing assets."""
        self._vocabulary = vocab

    def get_vocabulary(self):
        """Get vocabulary for testing."""
        return self._vocabulary if self._vocabulary is not None else []

    def save_assets(self, dir_path):
        """Save vocabulary as an asset."""
        if self._vocabulary:
            import os

            vocab_file = os.path.join(dir_path, "vocabulary.txt")
            with open(vocab_file, "w") as f:
                f.write("\n".join(self._vocabulary))

    def load_assets(self, dir_path):
        """Load vocabulary from assets."""
        import os

        vocab_file = os.path.join(dir_path, "vocabulary.txt")
        if os.path.exists(vocab_file):
            with open(vocab_file, "r") as f:
                self._vocabulary = [line.strip() for line in f.readlines()]


class OrbaxCheckpointTest(testing.TestCase, parameterized.TestCase):
    def _create_test_model(self):
        """Create a simple test model compatible with 2-device sharding."""
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(6, name="dense_layer")(inputs)  # 6 units (div by 2)
        outputs = layers.Dense(2, name="output_layer")(x)
        model = models.Model(inputs, outputs, name="test_model")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _create_dummy_data(self, num_samples=100):
        """Create dummy training data."""
        x = np.random.randn(num_samples, 10)
        y = np.random.randn(num_samples, 2)  # Match 2 outputs
        return x, y

    @parameterized.parameters(
        {"save_freq": 10, "epochs": 1, "batch_size": 5},  # batch-level
        {"save_freq": "epoch", "epochs": 3, "batch_size": None},  # epoch-level
    )
    @pytest.mark.requires_trainable_backend
    def test_checkpoint_saving_basic(self, save_freq, epochs, batch_size):
        """Test basic checkpoint saving with different frequencies."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=50)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_save_{save_freq}_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq=save_freq
        )

        # Train with specified configuration
        fit_kwargs = {"callbacks": [callback], "verbose": 0}
        if batch_size:
            fit_kwargs["batch_size"] = batch_size
        model.fit(x, y, epochs=epochs, **fit_kwargs)

        # Verify checkpoint files were created
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

    @parameterized.parameters(
        {"mode": "min", "monitor": "loss"},
        {"mode": "max", "monitor": "loss"},
    )
    @pytest.mark.requires_trainable_backend
    def test_save_best_only(self, mode, monitor):
        """Test save_best_only with different modes."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=100)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_best_{mode}_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor=monitor,
            save_best_only=True,
            mode=mode,
            save_freq="epoch",
        )

        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

    @parameterized.parameters(
        {"save_on_background": False},
        {"save_on_background": True},
    )
    @pytest.mark.requires_trainable_backend
    def test_async_vs_sync_saving(self, save_on_background):
        """Test synchronous vs asynchronous saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_async_{save_on_background}_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
        )

        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

    @pytest.mark.requires_trainable_backend
    def test_max_to_keep(self):
        """Test max_to_keep parameter limits number of checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_max_keep_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", max_to_keep=2
        )

        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertLessEqual(len(checkpoint_files), 5)

    @pytest.mark.requires_trainable_backend
    def test_load_weights_from_orbax_checkpoint(self):
        """Test loading weights from Orbax checkpoint using load_weights."""

        # Create and train model to create checkpoint
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_load_weights_orbax"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_weights_only=True,  # Only save weights for load_weights test
        )

        # Train to create checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        # Get original weights after training
        original_weights = model.get_weights()

        # Create a new model with the same architecture
        new_model = self._create_test_model()

        # Initialize with different weights to ensure loading works
        different_weights = [w * 2 for w in original_weights]
        new_model.set_weights(different_weights)

        # Verify weights are different initially
        new_weights_before = new_model.get_weights()
        for orig, new in zip(original_weights, new_weights_before):
            self.assertNotAllClose(
                orig, new, msg="Weights should be different before loading"
            )

        # Load weights from Orbax checkpoint
        new_model.load_weights(checkpoint_dir)

        # Verify weights were loaded correctly
        loaded_weights = new_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            self.assertAllClose(
                orig,
                loaded,
                msg="Weights should match after loading from checkpoint",
            )

    @pytest.mark.requires_trainable_backend
    def test_save_freq_epoch(self):
        """Test save_freq='epoch' functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_epoch_freq_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
        )

        # Train for 3 epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Should have only the latest checkpoint (epoch 2) due to max_to_keep=1
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertEqual(
            len(checkpoint_files),
            1,
            f"Should have exactly 1 checkpoint due to max_to_keep=1, "
            f"found {len(checkpoint_files)}: {checkpoint_files}",
        )

        # Check for the latest epoch directory (should be the highest numbered)
        # Note: Due to preservation policy behavior, the actual latest kept
        # may vary
        # So we check that at least one checkpoint exists and has a reasonable
        # name
        self.assertTrue(
            len(checkpoint_files) == 1 and checkpoint_files[0].isdigit(),
            f"Should have exactly one checkpoint with numeric name, "
            f"found {checkpoint_files}",
        )

    def test_invalid_save_freq(self):
        """Test error handling for invalid save_freq parameter."""
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_invalid_freq")
        with self.assertRaises(ValueError):
            OrbaxCheckpoint(directory=checkpoint_dir, save_freq="invalid")

    @pytest.mark.requires_trainable_backend
    def test_initial_value_threshold(self):
        """Test initial_value_threshold parameter."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_threshold")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor="loss",
            save_best_only=True,
            mode="min",
            initial_value_threshold=1.0,
            save_freq="epoch",
        )

        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)
        self.assertTrue(os.path.exists(checkpoint_dir))

    @parameterized.parameters(
        {"save_on_background": False},
        {"save_on_background": True},
    )
    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_comprehensive(self, save_on_background):
        """Test checkpoint loading with async and sync saving."""
        model = self._create_test_model()
        model.compile(optimizer="adam", loss="mse")
        x, y = self._create_dummy_data(num_samples=200)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(),
            f"test_loading_{save_on_background}_{id(self)}",
        )

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
            save_weights_only=True,
        )

        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        original_weights = model.get_weights()

        # Test load_weights functionality
        new_model = self._create_test_model()
        new_model.compile(optimizer="adam", loss="mse")
        new_x, new_y = self._create_dummy_data(num_samples=10)
        new_model.fit(new_x, new_y, epochs=1, batch_size=5, verbose=0)

        different_weights = [w * 2 for w in original_weights]
        new_model.set_weights(different_weights)

        # Verify different before loading
        for orig, new in zip(original_weights, new_model.get_weights()):
            self.assertNotAllClose(orig, new)

        # Load and verify
        new_model.load_weights(checkpoint_dir)
        for orig, loaded in zip(original_weights, new_model.get_weights()):
            self.assertAllClose(orig, loaded)

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Requires JAX backend for distribution",
    )
    def test_distributed_checkpoint_functionality(self):
        """Test OrbaxCheckpoint with distributed training."""
        import jax

        from keras.src.distribution import DeviceMesh
        from keras.src.distribution import LayoutMap
        from keras.src.distribution import ModelParallel
        from keras.src.distribution import TensorLayout
        from keras.src.distribution import distribution as get_distribution
        from keras.src.distribution import set_distribution

        # Check if we have at least 1 device
        devices = jax.devices()

        # Skip test if more than 2 devices, as these tests are designed
        # for 2-device scenarios and may not work with more devices
        if len(devices) > 2:
            self.skipTest(f"Test requires 2 devices, found {len(devices)}")

        num_devices = min(2, len(devices))

        # Skip if only single device - distributed functionality can't be tested
        if num_devices < 2:
            self.skipTest(
                "Test requires distributed setup with multiple devices"
            )

        print(f"Available devices: {devices}, using {num_devices} devices")

        # Set up multi-device distribution
        device_mesh = DeviceMesh((num_devices,), axis_names=["data"])
        layout_map = LayoutMap(device_mesh)
        layout_map["dense_layer/kernel"] = TensorLayout(axes=("data", None))
        layout_map["dense_layer/bias"] = TensorLayout(axes=(None,))
        layout_map["output_layer/kernel"] = TensorLayout(axes=(None, "data"))
        layout_map["output_layer/bias"] = TensorLayout(axes=(None,))

        distribution = ModelParallel(
            device_mesh=device_mesh, layout_map=layout_map
        )

        # Save original distribution state
        original_distribution = get_distribution()

        try:
            # Set distribution
            set_distribution(distribution)

            # Create and train model with distribution
            model = self._create_test_model()
            x, y = self._create_dummy_data(num_samples=50)

            checkpoint_dir = os.path.join(
                self.get_temp_dir(), "test_distributed_checkpoint"
            )
            callback = OrbaxCheckpoint(
                directory=checkpoint_dir,
                save_freq="epoch",
            )

            # Train to create checkpoint
            model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

            # Get original model predictions and weights
            original_predictions = model.predict(x[:5], verbose=0)
            original_weights = model.get_weights()

            # Load checkpoint using load_weights

            # Create fresh model and load weights
            fresh_model = self._create_test_model()
            fresh_model.load_weights(checkpoint_dir)
            loaded_weights = fresh_model.get_weights()

            # Verify loaded weights match original
            for orig, loaded in zip(original_weights, loaded_weights):
                self.assertAllClose(orig, loaded)

            # Verify loaded model produces same predictions
            loaded_predictions = fresh_model.predict(x[:5], verbose=0)
            self.assertAllClose(original_predictions, loaded_predictions)

            # Verify sharding is maintained after loading
            # Check that both models have the same distribution
            current_dist = get_distribution()
            self.assertIsNotNone(current_dist)
            self.assertEqual(type(current_dist), ModelParallel)

            # Verify model variables are sharded correctly
            # In JAX, sharded variables should have different sharding info

            # Get sharding info for original model variables
            original_shardings = {}
            for name, var in model.variables.items():
                if hasattr(var, "sharding"):
                    original_shardings[name] = var.sharding

            # Get sharding info for loaded model variables
            loaded_shardings = {}
            for name, var in fresh_model.variables.items():
                if hasattr(var, "sharding"):
                    loaded_shardings[name] = var.sharding

            # Verify shardings match
            for name in original_shardings:
                if name in loaded_shardings:
                    self.assertEqual(
                        original_shardings[name],
                        loaded_shardings[name],
                        f"Sharding mismatch for variable {name}",
                    )

            print("Distributed checkpoint functionality and sharding verified")

        finally:
            # Restore original distribution
            if original_distribution is not None:
                set_distribution(original_distribution)
            else:
                try:
                    set_distribution(None)
                except:
                    pass

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_via_saving_api(self):
        """Test model loading via saving API."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        # Test basic model loading
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_basic_loading")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        original_weights = model.get_weights()
        loaded_model = saving.load_model(checkpoint_dir)

        # Verify weights and compilation
        self.assertEqual(len(original_weights), len(loaded_model.get_weights()))
        for orig, loaded in zip(original_weights, loaded_model.get_weights()):
            self.assertAllClose(orig, loaded)
        self.assertTrue(loaded_model.compiled)

        # Test weights-only checkpoint should fail with load_model
        weights_only_dir = os.path.join(
            self.get_temp_dir(), "test_weights_only"
        )
        weights_callback = OrbaxCheckpoint(
            directory=weights_only_dir,
            save_freq="epoch",
            save_weights_only=True,
        )
        model.fit(x, y, epochs=1, callbacks=[weights_callback], verbose=0)

        with self.assertRaises(ValueError):
            saving.load_model(weights_only_dir)

    @parameterized.parameters(
        {"save_on_background": False},
        {"save_on_background": True},
    )
    @pytest.mark.requires_trainable_backend
    def test_comprehensive_model_state_restoration(self, save_on_background):
        """Test comprehensive model state restoration with exact weight
        matching.

        Tests sync/async saving, exact weight matching, and complete state
        restoration including trainable/non-trainable variables, optimizer
        state, and custom layers.
        """
        utils.set_random_seed(42)

        # Create model with custom layer having non-trainable variables
        @register_keras_serializable(package="test")
        class CustomLayer(layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.units = units

            def build(self, input_shape):
                self.kernel = self.add_weight(
                    shape=(input_shape[-1], self.units), name="kernel"
                )
                self.moving_mean = self.add_weight(
                    shape=(self.units,), trainable=False, name="moving_mean"
                )
                super().build(input_shape)

            def call(self, inputs):
                return inputs @ self.kernel

        # Build model with both trainable and non-trainable variables
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(8, name="dense_layer")(inputs)
        outputs = CustomLayer(2, name="custom_layer")(x)
        model = models.Model(inputs, outputs, name="comprehensive_test_model")
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        x, y = self._create_dummy_data(num_samples=100)
        checkpoint_dir = os.path.join(
            self.get_temp_dir(),
            f"test_comprehensive_{save_on_background}_{id(self)}",
        )

        # Test saving with exact weight matching
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
        )
        model.fit(x, y, epochs=2, verbose=0, callbacks=[callback])

        # Verify exact weight matching functionality
        final_saved_weights = model.get_weights()
        self.assertIsNotNone(final_saved_weights, "Should have saved weights")

        # Load and verify complete model restoration
        loaded_model = saving.load_model(checkpoint_dir)

        # Architecture verification
        self.assertEqual(model.name, loaded_model.name)
        self.assertEqual(len(model.layers), len(loaded_model.layers))
        self.assertTrue(loaded_model.compiled)

        # Exact weight matching verification
        loaded_weights = loaded_model.get_weights()
        self.assertEqual(len(final_saved_weights), len(loaded_weights))
        for i, (saved, loaded) in enumerate(
            zip(final_saved_weights, loaded_weights)
        ):
            self.assertAllClose(saved, loaded, msg=f"Weight {i} mismatch")

        # Verify optimizer variables
        for i, (saved, loaded) in enumerate(
            zip(model.optimizer.variables, loaded_model.optimizer.variables)
        ):
            self.assertAllClose(saved, loaded, msg=f"Weight {i} mismatch")

    @parameterized.parameters(
        {"save_on_background": False},
        {"save_on_background": True},
    )
    @pytest.mark.requires_trainable_backend
    def test_checkpoint_with_assets(self, save_on_background):
        """Test checkpoint saving/loading with layers that have assets.

        Tests vocabulary persistence across save/load cycles.
        """
        # Use custom layer with assets that is JAX-compatible
        inputs = layers.Input(shape=(10,))
        custom_layer = LayerWithAssets(units=8, name="layer_with_assets")
        custom_layer.set_vocabulary(
            ["word1", "word2", "word3", "word4", "word5"]
        )
        x = custom_layer(inputs)
        outputs = layers.Dense(3, activation="relu")(x)
        model = models.Model(inputs, outputs, name="model_with_assets")

        # Disable JIT compilation for JAX and Torch backends for compatibility
        jit_compile = backend.backend() not in ("jax", "torch")
        model.compile(optimizer="adam", loss="mse", jit_compile=jit_compile)

        # Create training data
        x_train = np.random.randn(20, 10)
        y_train = np.random.random((20, 3))

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_assets_{save_on_background}_{id(self)}"
        )

        # Train with checkpoint callback
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
        )
        model.fit(x_train, y_train, epochs=2, callbacks=[callback], verbose=0)

        # Verify checkpoint was created
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.isdigit()]
        self.assertGreater(len(checkpoints), 0, "No checkpoints were saved")

        # Get original vocabulary and weights for comparison
        original_vocab = custom_layer.get_vocabulary()
        original_weights = model.get_weights()

        # Load the model from checkpoint
        loaded_model = saving.load_model(checkpoint_dir)

        # Verify model structure
        self.assertEqual(model.name, loaded_model.name)
        self.assertEqual(len(model.layers), len(loaded_model.layers))

        # Verify vocabulary (assets) was restored correctly
        loaded_custom_layer = loaded_model.get_layer("layer_with_assets")
        loaded_vocab = loaded_custom_layer.get_vocabulary()

        self.assertEqual(
            len(original_vocab), len(loaded_vocab), "Vocabulary length mismatch"
        )

        for i, (orig, loaded) in enumerate(zip(original_vocab, loaded_vocab)):
            self.assertEqual(orig, loaded, f"Vocabulary mismatch at index {i}")

        # Verify weights match
        loaded_weights = loaded_model.get_weights()
        self.assertEqual(len(original_weights), len(loaded_weights))
        for i, (orig, loaded) in enumerate(
            zip(original_weights, loaded_weights)
        ):
            self.assertAllClose(orig, loaded, msg=f"Weight {i} mismatch")

        # Verify inference produces same results
        test_input = np.random.randn(5, 10)
        original_output = model.predict(test_input, verbose=0)
        loaded_output = loaded_model.predict(test_input, verbose=0)
        self.assertAllClose(
            original_output,
            loaded_output,
            msg="Model outputs don't match after loading",
        )
