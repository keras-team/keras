import numpy as np
import tensorflow as tf

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.layers import Dense
from keras.src.layers import Embedding
from keras.src.optimizers.muon import Muon


class MuonTest(testing.TestCase):
    def test_config(self):
        optimizer = Muon(learning_rate=0.5, epsilon=1e-5)
        self.run_class_serialization_test(optimizer)

    def test_Newton_Schulz(self):
        optimizer = Muon()
        tensor_input = ops.array([[0.2499, 0.9105], [0.2655, 0.8824]])
        expected_output = ops.array([[-0.4422, 0.6457], [0.7285, 0.2968]])
        output = optimizer.zeropower_via_newtonschulz5(tensor_input, 5)
        self.assertAllClose(output, expected_output, rtol=1e-3, atol=1e-3)

    def test_adamw_single_step(self):
        optimizer = Muon()
        grads = ops.array([1.0, 6.0, 7.0, 2.0])
        var = backend.Variable([1.0, 2.0, 3.0, 4.0], name="test_vars")
        optimizer.build([var])
        optimizer._adamw_update_step(grads, var, 0.5)
        self.assertAllClose(var, [0.5, 1.5, 2.5, 3.5], rtol=1e-4, atol=1e-4)

    def test_should_use_adamw_excluded_layer(self):
        """Ensure exclude_layers keyword works and no .path is accessed."""
        optimizer = Muon(exclude_layers=["dense"])
        dummy_var = backend.Variable(
            [[1.0, 2.0], [3.0, 4.0]], name="dense_kernel_0"
        )
        result = optimizer._should_use_adamw(dummy_var)
        self.assertTrue(result)

    def test_should_use_adamw_embedding(self):
        """Embedding layer should use AdamW when exclude_embeddings=True."""
        embedding = Embedding(2, 2)
        embedding.build()
        optimizer = Muon(exclude_embeddings=True)
        result = optimizer._should_use_adamw(embedding.weights[0])
        self.assertTrue(result)

    def test_should_use_adamw_dimension_rule(self):
        """Variables with dimensions not between 2–4 use AdamW."""
        v_1d = backend.Variable([1.0, 2.0], name="v1d")
        v_5d = backend.Variable(np.zeros((2, 2, 2, 2, 2)), name="v5d")
        optimizer = Muon()
        self.assertTrue(optimizer._should_use_adamw(v_1d))
        self.assertTrue(optimizer._should_use_adamw(v_5d))

    def test_should_use_adamw_dense_layer(self):
        """2D dense layer weights should use Muon (False)."""
        dense = Dense(2)
        dense.build([None, 2])
        optimizer = Muon()
        result = optimizer._should_use_adamw(dense.weights[0])
        self.assertFalse(result)

    def test_muon_single_step(self):
        optimizer = Muon(learning_rate=0.5, weight_decay=0)
        grads = ops.array([[1.0, 6.0], [7.0, 2.0]])
        var = backend.Variable([[1.0, 2.0], [3.0, 4.0]])
        optimizer.build([var])
        optimizer._muon_update_step(grads, var, 0.5)
        self.assertAllClose(
            var, [[1.13, 1.51], [2.57, 4.06]], rtol=1e-2, atol=1e-2
        )

    def test_clip_norm(self):
        optimizer = Muon(clipnorm=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

    def test_clip_value(self):
        optimizer = Muon(clipvalue=1)
        grad = [np.array([100.0, 100.0])]
        clipped_grad = optimizer._clip_gradients(grad)
        self.assertAllClose(clipped_grad[0], [1.0, 1.0])

    def test_no_path_attribute_error(self):
        """Ensure compatibility with TF 2.16+ where
        ResourceVariable has no .path."""
        optimizer = Muon()
        var = tf.Variable([1.0, 2.0], name="test_var")
        # Force-run method that caused AttributeError in issue #21793

        try:
            result = optimizer._should_use_adamw(var)
            self.assertIn(result, [True, False])
        except AttributeError as e:
            self.fail(f"Unexpected AttributeError: {e}")
