import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ["KERAS_BACKEND"] = "tensorflow"
tf.random.set_seed(1234)

def create_toy_model():
    inputs = keras.Input(shape=(1,))
    x = keras.layers.Dense(100, activation="tanh", use_bias=True)(inputs)
    x = keras.layers.Dense(1000, activation="tanh", use_bias=True)(x)
    x = keras.layers.Dense(10, activation="tanh", use_bias=True)(x)
    outputs = keras.layers.Dense(1, activation=None, use_bias=False)(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def test_gradient_tape_issue():
    model = create_toy_model()
    
    x = np.expand_dims(np.linspace(0, 10, num=20), axis=1)
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    print(f"Number of layers: {len(model.layers)}")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.__class__.__name__}, Weights: {len(layer.weights)}")

    with tf.GradientTape(watch_accessed_variables=True) as tape:
        last_layer_weights = model.layers[-1].weights[0]
        #tape.watch(last_layer_weights)
        out = model(x)
    
    dout = tape.gradient(out, last_layer_weights)
    
    assert dout is not None, "Gradient should not be None"
    print("Gradient successfully computed!")

if __name__ == "__main__":
    test_gradient_tape_issue()