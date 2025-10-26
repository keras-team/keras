"""Test custom_gradient with JAX backend when Variables are passed."""
import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np

import keras
from keras import layers
from keras import ops


def test_custom_gradient_with_variable():
    """Test that custom_gradient works with Variables in JAX backend."""
    
    @ops.custom_gradient
    def roundpass(x, log_scaling):
        """Custom gradient function that uses a Variable."""
        scaling = ops.exp(log_scaling)
        rounded = ops.round(x * scaling) / scaling
        
        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            # Straight-through estimator: gradient passes through
            return upstream, ops.zeros_like(log_scaling)
        
        return rounded, grad
    
    # Create a simple layer that uses custom_gradient with a Variable
    class QuantizedLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.log_scaling = self.add_weight(
                name="log_scaling",
                shape=(),
                initializer="zeros",
                trainable=True,
            )
        
        def call(self, x):
            # This should work without needing to manually add .value
            return roundpass(x, self.log_scaling)
    
    # Build a simple model
    inputs = layers.Input(shape=(4,))
    x = QuantizedLayer()(inputs)
    outputs = layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer="adam",
        loss="mse",
    )
    
    # Create dummy data
    x_train = np.random.randn(32, 4).astype("float32")
    y_train = np.random.randn(32, 2).astype("float32")
    
    # Train for one step - this should not raise TypeError
    history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
    
    assert history is not None
    print(
        "✓ Test passed: custom_gradient works with "
        "Variables in JAX backend"
    )


def test_custom_gradient_with_variable_value_property():
    """Test that custom_gradient also works when .value is explicitly used."""
    
    @ops.custom_gradient
    def roundpass(x, log_scaling):
        """Custom gradient function that uses a Variable value."""
        scaling = ops.exp(log_scaling)
        rounded = ops.round(x * scaling) / scaling
        
        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            return upstream, ops.zeros_like(log_scaling)
        
        return rounded, grad
    
    class QuantizedLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.log_scaling = self.add_weight(
                name="log_scaling",
                shape=(),
                initializer="zeros",
                trainable=True,
            )
        
        def call(self, x):
            # Explicitly use .value (workaround mentioned in the issue)
            return roundpass(x, self.log_scaling.value)
    
    # Build a simple model
    inputs = layers.Input(shape=(4,))
    x = QuantizedLayer()(inputs)
    outputs = layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer="adam", loss="mse")
    
    x_train = np.random.randn(32, 4).astype("float32")
    y_train = np.random.randn(32, 2).astype("float32")
    
    history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
    
    assert history is not None
    print(
        "✓ Test passed: custom_gradient works with "
        "Variable.value in JAX backend"
    )
if __name__ == "__main__":
    print("Testing custom_gradient with JAX backend and Variables...")
    print()
    
    test_custom_gradient_with_variable()
    test_custom_gradient_with_variable_value_property()
    
    print()
    print("All tests passed! ✓")
