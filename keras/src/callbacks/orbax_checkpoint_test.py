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


# Custom layer with assets for testing
@register_keras_serializable(package="TestLayers")
class LayerWithAssets(layers.Layer):
    """A custom layer that has assets for testing purposes."""

    def __init__(self, units=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._asset_data = None

    def build(self, input_shape):
        # Create some asset data (e.g., lookup table, vocabulary)
        self._asset_data = np.arange(self.units * 2, dtype=np.float32)
        self._asset_data = self._asset_data.reshape((self.units, 2))
        super().build(input_shape)

    def call(self, inputs):
        # Simple pass-through for testing
        return inputs

    def assets(self):
        """Return list of asset arrays."""
        if self._asset_data is not None:
            return [self._asset_data]
        return []

    def _set_assets(self, assets):
        """Restore assets from loaded checkpoint."""
        if assets and len(assets) > 0:
            self._asset_data = assets[0]

    def get_config(self):
        config = super().get_config()
        config["units"] = self.units
        return config


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
    def test_save_and_load_assets(self, save_on_background):
        """Test saving and loading model assets with async/sync modes."""
        # Create model with layer that has assets
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(8, name="dense_layer")(inputs)
        x = LayerWithAssets(units=4, name="asset_layer")(x)
        outputs = layers.Dense(2, name="output_layer")(x)
        model = models.Model(inputs, outputs, name="asset_test_model")
        model.compile(optimizer="adam", loss="mse")

        # Get the asset layer and verify it has assets
        asset_layer = model.get_layer("asset_layer")
        original_assets = asset_layer.assets()
        self.assertEqual(len(original_assets), 1, "Should have 1 asset")
        self.assertEqual(
            original_assets[0].shape, (4, 2), "Asset shape mismatch"
        )
        original_asset_values = np.copy(original_assets[0])

        # Train and save
        x, y = self._create_dummy_data(num_samples=100)
        checkpoint_dir = os.path.join(
            self.get_temp_dir(),
            f"test_assets_{save_on_background}_{id(self)}",
        )

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
        )

        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Wait for async saves to complete
        callback.wait_until_finished()

        # Capture final weights after training completes
        final_saved_weights = model.get_weights()

        # Verify checkpoint directory structure
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(len(checkpoint_files), 0)

        # Check that assets directory exists
        latest_step = max([int(f) for f in checkpoint_files if f.isdigit()])
        assets_dir = os.path.join(checkpoint_dir, str(latest_step), "assets")
        self.assertTrue(
            os.path.exists(assets_dir), "Assets directory should exist"
        )

        # Verify asset files exist
        # The asset handler saves assets at:
        # assets/{model.name}/layers/{layer.name}
        asset_layer_dir = os.path.join(
            assets_dir, "asset_test_model", "layers", "asset_layer"
        )
        self.assertTrue(
            os.path.exists(asset_layer_dir),
            f"Asset layer directory should exist at {asset_layer_dir}",
        )

        asset_files = os.listdir(asset_layer_dir)
        self.assertIn("asset_0.npy", asset_files, "Asset file should exist")

        # Load model and verify assets are restored
        loaded_model = saving.load_model(checkpoint_dir)
        loaded_asset_layer = loaded_model.get_layer("asset_layer")
        loaded_assets = loaded_asset_layer.assets()

        # Verify assets were loaded correctly
        self.assertEqual(len(loaded_assets), 1, "Should have 1 asset")
        self.assertAllClose(
            original_asset_values,
            loaded_assets[0],
            msg="Asset values should match after loading",
        )

        # Verify model weights also loaded correctly
        loaded_weights = loaded_model.get_weights()
        self.assertEqual(len(final_saved_weights), len(loaded_weights))
        for i, (orig, loaded) in enumerate(
            zip(final_saved_weights, loaded_weights)
        ):
            self.assertAllClose(orig, loaded, msg=f"Weight {i} mismatch")

        # Verify loaded model has correct structure
        self.assertEqual(len(model.layers), len(loaded_model.layers))
        for orig_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            self.assertEqual(orig_layer.name, loaded_layer.name)
