"""
Title: Writing a training loop from scratch in JAX
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/25
Last modified: 2023/06/25
Description: Writing low-level training & evaluation loops in JAX.
Accelerator: None
"""

"""
## Setup
"""

import os

# This guide can only be run with the jax backend.
os.environ["KERAS_BACKEND"] = "jax"

import jax

# We import TF so we can use tf.data.
import tensorflow as tf
import keras
import numpy as np

"""
## Introduction

Keras provides default training and evaluation loops, `fit()` and `evaluate()`.
Their usage is covered in the guide
[Training & evaluation with the built-in methods](https://keras.io/guides/training_with_built_in_methods/).

If you want to customize the learning algorithm of your model while still leveraging
the convenience of `fit()`
(for instance, to train a GAN using `fit()`), you can subclass the `Model` class and
implement your own `train_step()` method, which
is called repeatedly during `fit()`.

Now, if you want very low-level control over training & evaluation, you should write
your own training & evaluation loops from scratch. This is what this guide is about.
"""

"""
## A first end-to-end example

To write a custom training loop, we need the following ingredients:

- A model to train, of course.
- An optimizer. You could either use an optimizer from `keras.optimizers`, or
one from the `optax` package.
- A loss function.
- A dataset. The standard in the JAX ecosystem is to load data via `tf.data`,
so that's what we'll use.

Let's line them up.

First, let's get the model and the MNIST dataset:
"""


def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model()

# Prepare the training dataset.
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)).astype("float32")
x_test = np.reshape(x_test, (-1, 784)).astype("float32")
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

"""
Next, here's the loss function and the optimizer.
We'll use a Keras optimizer in this case.
"""

# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

"""
### Getting gradients in JAX 

Let's train our model using mini-batch gradient with a custom training loop.

In JAX, gradients are computed via *metaprogramming*: you call the `jax.grad` (or
`jax.value_and_grad` on a function in order to create a gradient-computing function
for that first function.

So the first thing we need is a function that returns the loss value.
That's the function we'll use to generate the gradient function. Something like this:

```python
def compute_loss(x, y):
    ...
    return loss
```

Once you have such a function, you can compute gradients via metaprogramming as such:

```python
grad_fn = jax.grad(compute_loss)
grads = grad_fn(x, y)
```

Typically, you don't just want to get the gradient values, you also want to get
the loss value. You can do this by using `jax.value_and_grad` instead of `jax.grad`:

```python
grad_fn = jax.value_and_grad(compute_loss)
loss, grads = grad_fn(x, y)
```

### JAX computation is purely stateless

In JAX, everything must be a stateless function -- so our loss computation function
must be stateless as well. That means that all Keras variables (e.g. weight tensors)
must be passed as function inputs, and any variable that has been updated during the
forward pass must be returned as function output. The function have no side effect.

During the forward pass, the non-trainable variables of a Keras model might get
updated. These variables could be, for instance, RNG seed state variables or
BatchNormalization statistics. We're going to need to return those. So we need
something like this:

```python
def compute_loss_and_updates(trainable_variables, non_trainable_variables, x, y):
    ...
    return loss, non_trainable_variables
```

Once you have such a function, you can get the gradient function by
specifying `hax_aux` in `value_and_grad`: it tells JAX that the loss
computation function returns more outputs than just the loss. Note that the loss
should always be the first output.

```python
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
(loss, non_trainable_variables), grads = grad_fn(
    trainable_variables, non_trainable_variables, x, y
)
```

Now that we have established the basics,
let's implement this `compute_loss_and_updates` function.
Keras models have a `stateless_call` method which will come in handy here.
It works just like `model.__call__`, but it requires you to explicitly
pass the value of all the variables in the model, and it returns not just
the `__call__` outputs but also the (potentially updated) non-trainable
variables.
"""


def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, x, y
):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    return loss, non_trainable_variables


"""
Let's get the gradient function:
"""

grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)

"""
### The training step function

Next, let's implement the end-to-end training step, the function
that will both run the forward pass, compute the loss, compute the gradients,
but also use the optimizer to update the trainable variables. This function
also needs to be stateless, so it will get as input a `state` tuple that
includes every state element we're going to use:

- `trainable_variables` and `non_trainable_variables`: the model's variables.
- `optimizer_variables`: the optimizer's state variables,
such as momentum accumulators.

To update the trainable variables, we use the optimizer's stateless method
`stateless_apply`. It's equivalent to `optimizer.apply()`, but it requires
always passing `trainable_variables` and `optimizer_variables`. It returns
both the updated trainable variables and the updated optimizer_variables.
"""


def train_step(state, data):
    trainable_variables, non_trainable_variables, optimizer_variables = state
    x, y = data
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # Return updated state
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )


"""
### Make it fast with `jax.jit`

By default, JAX operations run eagerly,
just like in TensorFlow eager mode and PyTorch eager mode.
And just like TensorFlow eager mode and PyTorch eager mode, it's pretty slow
-- eager mode is better used as a debugging environment, not as a way to do
any actual work. So let's make our `train_step` fast by compiling it.

When you have a stateless JAX function, you can compile it to XLA via the 
`@jax.jit` decorator. It will get traced during its first execution, and in
subsequent executions you will be executing the traced graph (this is just
like `@tf.function(jit_compile=True)`. Let's try it:
"""


@jax.jit
def train_step(state, data):
    trainable_variables, non_trainable_variables, optimizer_variables = state
    x, y = data
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # Return updated state
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )


"""
We're now ready to train our model. The training loop itself
is trivial: we just repeatedly call `loss, state = train_step(state, data)`.

Note:

- We convert the TF tensors yielded by the `tf.data.Dataset` to NumPy
before passing them to our JAX function.
- All variables must be built beforehand:
the model must be built and the optimizer must be built. Since we're using a
Functional API model, it's already built, but if it were a subclassed model
you'd need to call it on a batch of data to build it.
"""

# Build optimizer variables.
optimizer.build(model.trainable_variables)

trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
state = trainable_variables, non_trainable_variables, optimizer_variables

# Training loop
for step, data in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = train_step(state, data)
    # Log every 100 batches.
    if step % 100 == 0:
        print(f"Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")

"""
A key thing to notice here is that the loop is entirely stateless -- the variables
attached to the model (`model.weights`) are never getting updated during the loop.
Their new values are only stored in the `state` tuple. That means that at some point,
before saving the model, you should be attaching the new variable values back to the model.

Just call `variable.assign(new_value)` on each model variable you want to update:
"""

trainable_variables, non_trainable_variables, optimizer_variables = state
for variable, value in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for variable, value in zip(
    model.non_trainable_variables, non_trainable_variables
):
    variable.assign(value)

"""
## Low-level handling of metrics

Let's add metrics monitoring to this basic training loop.

You can readily reuse built-in Keras metrics (or custom ones you wrote) in such training
loops written from scratch. Here's the flow:

- Instantiate the metric at the start of the loop
- Include `metric_variables` in the `train_step` arguments
and `compute_loss_and_updates` arguments.
- Call `metric.stateless_update_state()` in the `compute_loss_and_updates` function.
It's equivalent to `update_state()` -- only stateless.
- When you need to display the current value of the metric, outside the `train_step`
(in the eager scope), attach the new metric variable values to the metric object
and vall `metric.result()`.
- Call `metric.reset_state()` when you need to clear the state of the metric
(typically at the end of an epoch)

Let's use this knowledge to compute `CategoricalAccuracy` on training and
validation data at the end of training:
"""

# Get a fresh model
model = get_model()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()


def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, metric_variables, x, y
):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metric_variables = train_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, (non_trainable_variables, metric_variables)


grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)


@jax.jit
def train_step(state, data):
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    ) = state
    x, y = data
    (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(
        trainable_variables, non_trainable_variables, metric_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # Return updated state
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    )


"""
We'll also prepare an evaluation step function:
"""


@jax.jit
def eval_step(state, data):
    trainable_variables, non_trainable_variables, metric_variables = state
    x, y = data
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metric_variables = val_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, (
        trainable_variables,
        non_trainable_variables,
        metric_variables,
    )


"""
Here are our loops:
"""

# Build optimizer variables.
optimizer.build(model.trainable_variables)

trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
metric_variables = train_acc_metric.variables
state = (
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metric_variables,
)

# Training loop
for step, data in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = train_step(state, data)
    # Log every 100 batches.
    if step % 100 == 0:
        print(f"Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
        _, _, _, metric_variables = state
        for variable, value in zip(
            train_acc_metric.variables, metric_variables
        ):
            variable.assign(value)
        print(f"Training accuracy: {train_acc_metric.result()}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")

metric_variables = val_acc_metric.variables
(
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metric_variables,
) = state
state = trainable_variables, non_trainable_variables, metric_variables

# Eval loop
for step, data in enumerate(val_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = eval_step(state, data)
    # Log every 100 batches.
    if step % 100 == 0:
        print(
            f"Validation loss (for 1 batch) at step {step}: {float(loss):.4f}"
        )
        _, _, metric_variables = state
        for variable, value in zip(val_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f"Validation accuracy: {val_acc_metric.result()}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")

"""
## Low-level handling of losses tracked by the model

Layers & models recursively track any losses created during the forward pass
by layers that call `self.add_loss(value)`. The resulting list of scalar loss
values are available via the property `model.losses`
at the end of the forward pass.

If you want to be using these loss components, you should sum them
and add them to the main loss in your training step.

Consider this layer, that creates an activity regularization loss:
"""


class ActivityRegularizationLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * jax.numpy.sum(inputs))
        return inputs


"""
Let's build a really simple model that uses it:
"""

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu")(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

"""
Here's what our `compute_loss_and_updates` function should look like now:

- Pass `return_losses=True` to `model.stateless_call()`.
- Sum the resulting `losses` and add them to the main loss.
"""


def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, metric_variables, x, y
):
    y_pred, non_trainable_variables, losses = model.stateless_call(
        trainable_variables, non_trainable_variables, x, return_losses=True
    )
    loss = loss_fn(y, y_pred)
    if losses:
        loss += jax.numpy.sum(losses)
    metric_variables = train_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, non_trainable_variables, metric_variables


"""
That's it!
"""
