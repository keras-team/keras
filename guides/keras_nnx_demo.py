"""
Title: How to use Keras with NNX backend
Author: [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2025/08/07
Last modified: 2025/08/07
Description: How to use Keras with NNX backend
Accelerator: CPU
"""

# -*- coding: utf-8 -*-
"""

# A Guide to the Keras & Flax NNX Integration

This tutorial will guide you through the integration of Keras with Flax's NNX (Neural Networks JAX) module system, demonstrating how it significantly enhances variable handling and opens up advanced training capabilities within the JAX ecosystem. Whether you love the simplicity of model.fit() or the fine-grained control of a custom training loop, this integration lets you have the best of both worlds. Let's dive in!

# Why Keras and NNX Integration?

Keras is known for its user-friendliness and high-level API, making deep learning accessible. JAX, on the other hand, provides high-performance numerical computation, especially suited for machine learning research due to its JIT compilation and automatic differentiation capabilities. NNX is Flax's functional module system built on JAX, offering explicit state management and powerful functional programming paradigms

NNX is designed for simplicity. It is characterized by its Pythonic approach, where modules are standard Python classes, promoting ease of use and familiarity. NNX prioritizes user-friendliness and offers fine-grained control over JAX transformations through typed Variable collections

The integration of Keras with NNX allows you to leverage the best of both worlds: the simplicity and modularity of Keras for model construction, combined with the power and explicit control of NNX and JAX for variable management and sophisticated training loops.

# Getting Started: Setting Up Your Environment
"""

!pip install -q git+https://github.com/hertschuh/keras.git@saving_op
!pip uninstall -y flax
!pip install -q flax==0.11.0

"""# Enabling NNX Mode

To activate the integration, we must set two environment variables before importing Keras. This tells Keras to use the JAX backend and switch to NNX as an opt in feature.
"""

import os
os.environ["KERAS_BACKEND"]="jax"
os.environ["KERAS_NNX_ENABLED"]="true"
from flax import nnx
import keras
import jax.numpy as jnp
print("✅ Keras is now running on JAX with NNX enabled!")

"""# The Core Integration: Keras Variables in NNX

The heart of this integration is the new keras.Variable, which is designed to be a native citizen of the Flax NNX ecosystem. This means you can mix Keras and NNX components freely, and NNX's tracing and state management tools will understand your Keras variables.
Let's prove it. We'll create an nnx.Module that contains both a standard nnx.Linear layer and a keras.Variable.
"""

from keras import Variable as KerasVariable

class MyNnxModel(nnx.Module):
  def __init__(self, rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.custom_variable = KerasVariable(jnp.ones((1, 3)))

  def __call__(self, x):
    return self.linear(x) + self.custom_variable

# Instantiate the model
model = MyNnxModel(rngs=nnx.Rngs(0))

# --- Verification ---
# 1. Is the KerasVariable traced by NNX?
print(f"✅ Traced: {hasattr(model.custom_variable, '_trace_state')}")

# 2. Does NNX see the KerasVariable in the model's state?
print("✅ Variables:", nnx.variables(model))

# 3. Can we access its value directly?
print("✅ Value:", model.custom_variable.value)

"""What this shows:
The KerasVariable is successfully traced by NNX, just like any native nnx.Variable.
The nnx.variables() function correctly identifies and lists our custom_variable as part of the model's state.
This confirms that Keras state and NNX state can live together in perfect harmony.

# The Best of Both Worlds: Training Workflows

Now for the exciting part: training models. This integration unlocks two powerful workflows.

## Workflow 1: The Classic Keras Experience (model.fit)
"""

import numpy as np

# --- 1. Create a Keras Model ---
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=(10,), name="my_dense_layer")
])

print("--- Initial Model Weights ---")
initial_weights = model.get_weights()
print(f"Initial Kernel: {initial_weights[0].T}") # .T for better display
print(f"Initial Bias: {initial_weights[1]}")

# --- 2. Create Dummy Data ---
X_dummy = np.random.rand(100, 10)
y_dummy = np.random.rand(100, 1)

# --- 3. Compile and Fit ---
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss='mean_squared_error')

print("\n--- Training with model.fit() ---")
history = model.fit(X_dummy, y_dummy, epochs=5, batch_size=32, verbose=1)

# --- 4. Verify a change ---
print("\n--- Weights After Training ---")
updated_weights = model.get_weights()
print(f"Updated Kernel: {updated_weights[0].T}")
print(f"Updated Bias: {updated_weights[1]}")

# Verification
if not np.array_equal(initial_weights[1], updated_weights[1]):
    print("\n✅ SUCCESS: Model variables were updated during training.")
else:
    print("\n❌ FAILURE: Model variables were not updated.")

"""As you can see, your existing Keras code works out-of-the-box, giving you a high-level, productive experience powered by JAX and NNX under the hood.

## Workflow 2: The Power of NNX: Custom Training Loops

For maximum flexibility, you can treat any Keras layer or model as an nnx.Module and write your own training loop using libraries like Optax.
This is perfect when you need fine-grained control over the gradient and update process.
"""

import numpy as np
import optax

X = np.linspace(-jnp.pi, jnp.pi, 100)[:, None]
Y = 0.8 * X + 0.1 + np.random.normal(0, 0.1, size=X.shape)

class MySimpleKerasModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define the layers of your model
        self.dense_layer = keras.layers.Dense(1)

    def call(self, inputs):
        # Define the forward pass
        # The 'inputs' argument will receive the input tensor when the model is called
        return self.dense_layer(inputs)

model = MySimpleKerasModel()
model(X)

tx = optax.sgd(1e-3)
trainable_var = nnx.All(keras.Variable, lambda path, x: getattr(x, '_trainable', False))
optimizer = nnx.Optimizer(model, tx, wrt=trainable_var)

@nnx.jit
def train_step(model, optimizer, batch):
  x, y = batch

  def loss_fn(model_):
    y_pred = model_(x)
    return jnp.mean((y - y_pred) ** 2)

  diff_state = nnx.DiffState(0, trainable_var)
  grads = nnx.grad(loss_fn, argnums=diff_state)(model)
  optimizer.update(model, grads)

@nnx.jit
def test_step(model, batch):
  x, y = batch
  y_pred = model(x)
  loss = jnp.mean((y - y_pred) ** 2)
  return {'loss': loss}


def dataset(batch_size=10):
  while True:
    idx = np.random.choice(len(X), size=batch_size)
    yield X[idx], Y[idx]

for step, batch in enumerate(dataset()):
  train_step(model, optimizer, batch)

  if step % 100 == 0:
    logs = test_step(model, (X, Y))
    print(f"step: {step}, loss: {logs['loss']}")

  if step >= 500:
    break

"""This example shows how a keras model object is seamlessly passed to nnx.Optimizer and differentiated by nnx.grad. This composition allows you to integrate Keras components into sophisticated JAX/NNX workflows. This approach also works perfectly with sequential, functional, subclassed keras models are even just layers.

#  Saving and Loading

Your investment in the Keras ecosystem is safe. Standard features like model serialization work exactly as you'd expect.
"""

# Create a simple model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=(10,))])
dummy_input = np.random.rand(1, 10)

# Test call
print("Original model output:", model(dummy_input))

# Save and load
model.save('my_nnx_model.keras')
restored_model = keras.models.load_model('my_nnx_model.keras')

print("Restored model output:", restored_model(dummy_input))

"""# Real-World Application: Training Gemma

Before trying out this KerasHub model, please make sure you have set up your Kaggle credentials in colab secrets. The colab pulls in `KAGGLE_KEY` and `KAGGLE_USERNAME` to authenticate and download the models.
"""

import keras_hub

# Set a float16 policy for memory efficiency
keras.config.set_dtype_policy("float16")

# Load Gemma from KerasHub
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_1.1_instruct_2b_en")

# --- 1. Inference / Generation ---
print("--- Gemma Generation ---")
output = gemma_lm.generate("Keras is a", max_length=30)
print(output)

# --- 2. Fine-tuning ---
print("\n--- Gemma Fine-tuning ---")
# Dummy data for demonstration
features = np.array(["The quick brown fox jumped.", "I forgot my homework."])
# The model.fit() API works seamlessly!
gemma_lm.fit(x=features, batch_size=2)
print("\n✅ Gemma fine-tuning step completed successfully!")

"""# Conclusion

The Keras-NNX integration represents a significant step forward, offering a unified framework for both rapid prototyping and high-performance, customizable research. You can now:
Use familiar Keras APIs (Sequential, Model, fit, save) on a JAX backend.
Integrate Keras layers and models directly into Flax NNX modules and training loops.Integrate keras code/model with NNX ecosytem like Qwix, Tunix, etc.
Leverage the entire JAX ecosystem (e.g., nnx.jit, optax) with your Keras models.
Seamlessly work with large models from KerasHub.
"""
