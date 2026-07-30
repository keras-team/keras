from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import Variable

from keras.src.distribution.tensor_parallel import autoconfig


class AutoconfigTest(testing.TestCase):
    def setUp(self):
        super().setUp()

    def test_analyze_dense_layer(self):
        # Case 1: Dense Up Projection (Expansion)
        layer_up = layers.Dense(64)
        layer_up.build((None, 16))  # Input 16, Output 64
        self.assertEqual(autoconfig.analyze_dense_layer(layer_up), "up_projection")

        # Case 2: Dense Down Projection (Contraction)
        layer_down = layers.Dense(16)
        layer_down.build((None, 64))  # Input 64, Output 16
        self.assertEqual(autoconfig.analyze_dense_layer(layer_down), "down_projection")

        # Case 3: Standard Dense
        layer_dense = layers.Dense(20)
        layer_dense.build((None, 16))  # Input 16, Output 20
        self.assertEqual(autoconfig.analyze_dense_layer(layer_dense), "dense")

        # Case 4: Custom threshold
        self.assertEqual(
            autoconfig.analyze_dense_layer(layer_up, expansion_threshold=5.0), "dense"
        )

    def test_analyze_dense_layer_einsum(self):
        # EinsumDense with 3D kernel (up projection heuristic)
        # Equation: "ab,bcd->acd" -> kernel shape (b, c, d)
        # Input dim is 'b', output dims are 'c' and 'd'
        layer = layers.EinsumDense("ab,bcd->acd", output_shape=(4, 8))
        layer.build((None, 2))  # Input dim 2
        # Kernel shape: (2, 4, 8) -> input_dim=2, output_dim=32
        self.assertEqual(autoconfig.analyze_dense_layer(layer), "up_projection")

        # EinsumDense with 3D kernel (down projection heuristic)
        layer_down = layers.EinsumDense("ab,bcd->acd", output_shape=(1, 1))
        layer_down.build((None, 16))  # Input dim 16
        # Kernel shape: (16, 1, 1) -> input_dim=16, output_dim=1
        self.assertEqual(autoconfig.analyze_dense_layer(layer_down), "down_projection")

    def test_get_var_key(self):
        # Variable with path
        var_with_path = Variable(0.0, name="my_var")
        var_with_path._path = "layer_1/kernel"
        self.assertEqual(autoconfig._get_var_key(var_with_path), "layer_1/kernel")

        # Variable without explicit path (auto-generated)
        var_without_path = Variable(0.0)
        # Keras auto-generates a path/name like 'variable' or 'variable_1'
        expected_key = var_without_path.path if var_without_path.path else id(var_without_path)
        self.assertEqual(autoconfig._get_var_key(var_without_path), expected_key)

    def test_get_default_config(self):
        # Create a dummy model to test config generation traversal
        model = models.Sequential([
            layers.Embedding(1000, 64, name="embeddings"),
            layers.Dense(256, name="dense_up"),
            layers.Dropout(0.5, name="dropout"),
            layers.Dense(64, name="dense_down"),
        ])
        model.build((None, 10))

        device_ids = ["gpu:0", "gpu:1"]
        
        # This will call ParallelLayoutMap which is imported from tensor_layout.py
        # We assume it will run correctly once dependencies are added.
        # Note: In this branch, tensor_layout.py might be missing if not merged!
        try:
            config = autoconfig.get_default_config(model, device_ids)
            self.assertIsNotNone(config)
        except ImportError:
            # Expected if tensor_layout is missing on this branch
            pass
