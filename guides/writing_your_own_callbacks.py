"""
Title: Writing your own callbacks
Authors: Rick Chao, Francois Chollet
Date created: 2019/03/20
Last modified: 2023/06/25
Description: Complete guide to writing new Keras callbacks.
Accelerator: GPU
"""

"""
## Introduction

A callback is a powerful tool to customize the behavior of a Keras model during
training, evaluation, or inference. Examples include `keras.callbacks.TensorBoard`
to visualize training progress and results with TensorBoard, or
`keras.callbacks.ModelCheckpoint` to periodically save your model during training.

In this guide, you will learn what a Keras callback is, what it can do, and how you can
build your own. We provide a few demos of simple callback applications to get you
started.
"""

"""
## Setup
"""

import numpy as np
import keras

"""
## Keras callbacks overview

All callbacks subclass the `keras.callbacks.Callback` class, and
override a set of methods called at various stages of training, testing, and
predicting. Callbacks are useful to get a view on internal states and statistics of
the model during training.

You can pass a list of callbacks (as the keyword argument `callbacks`) to the following
model methods:

- `keras.Model.fit()`
- `keras.Model.evaluate()`
- `keras.Model.predict()`
"""

"""
## An overview of callback methods

### Global methods

#### `on_(train|test|predict)_begin(self, logs=None)`

Called at the beginning of `fit`/`evaluate`/`predict`.

#### `on_(train|test|predict)_end(self, logs=None)`

Called at the end of `fit`/`evaluate`/`predict`.

### Batch-level methods for training/testing/predicting

#### `on_(train|test|predict)_batch_begin(self, batch, logs=None)`

Called right before processing a batch during training/testing/predicting.

#### `on_(train|test|predict)_batch_end(self, batch, logs=None)`

Called at the end of training/testing/predicting a batch. Within this method, `logs` is
a dict containing the metrics results.

### Epoch-level methods (training only)

#### `on_epoch_begin(self, epoch, logs=None)`

Called at the beginning of an epoch during training.

#### `on_epoch_end(self, epoch, logs=None)`

Called at the end of an epoch during training.
"""

"""
## A basic example

Let's take a look at a concrete example. To get started, let's import tensorflow and
define a simple Sequential Keras model:
"""


# Define the Keras model to add callbacks to
def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model


"""
Then, load the MNIST data for training and testing from Keras datasets API:
"""

# Load example MNIST data and pre-process it
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Limit the data to 1000 samples
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]

"""
Now, define a simple custom callback that logs:

- When `fit`/`evaluate`/`predict` starts & ends
- When each epoch starts & ends
- When each training batch starts & ends
- When each evaluation (test) batch starts & ends
- When each inference (prediction) batch starts & ends
"""


class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print(
            "Start epoch {} of training; got log keys: {}".format(epoch, keys)
        )

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(
            "...Training: start of batch {}; got log keys: {}".format(
                batch, keys
            )
        )

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(
            "...Training: end of batch {}; got log keys: {}".format(batch, keys)
        )

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(
            "...Evaluating: start of batch {}; got log keys: {}".format(
                batch, keys
            )
        )

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(
            "...Evaluating: end of batch {}; got log keys: {}".format(
                batch, keys
            )
        )

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(
            "...Predicting: start of batch {}; got log keys: {}".format(
                batch, keys
            )
        )

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(
            "...Predicting: end of batch {}; got log keys: {}".format(
                batch, keys
            )
        )


"""
Let's try it out:
"""

model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=1,
    verbose=0,
    validation_split=0.5,
    callbacks=[CustomCallback()],
)

res = model.evaluate(
    x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomCallback()]
)

res = model.predict(x_test, batch_size=128, callbacks=[CustomCallback()])

"""
### Usage of `logs` dict

The `logs` dict contains the loss value, and all the metrics at the end of a batch or
epoch. Example includes the loss and mean absolute error.
"""


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(
                batch, logs["loss"]
            )
        )

    def on_test_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(
                batch, logs["loss"]
            )
        )

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()],
)

res = model.evaluate(
    x_test,
    y_test,
    batch_size=128,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()],
)

"""
## Usage of `self.model` attribute

In addition to receiving log information when one of their methods is called,
callbacks have access to the model associated with the current round of
training/evaluation/inference: `self.model`.

Here are a few of the things you can do with `self.model` in a callback:

- Set `self.model.stop_training = True` to immediately interrupt training.
- Mutate hyperparameters of the optimizer (available as `self.model.optimizer`),
such as `self.model.optimizer.learning_rate`.
- Save the model at period intervals.
- Record the output of `model.predict()` on a few test samples at the end of each
epoch, to use as a sanity check during training.
- Extract visualizations of intermediate features at the end of each epoch, to monitor
what the model is learning over time.
- etc.

Let's see this in action in a couple of examples.
"""

"""
## Examples of Keras callback applications

### Early stopping at minimum loss

This first example shows the creation of a `Callback` that stops training when the
minimum of loss has been reached, by setting the attribute `self.model.stop_training`
(boolean). Optionally, you can provide an argument `patience` to specify how many
epochs we should wait before stopping after having reached a local minimum.

`keras.callbacks.EarlyStopping` provides a more complete and general implementation.
"""


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()],
)

"""
### Learning rate scheduling

In this example, we show how a custom Callback can be used to dynamically change the
learning rate of the optimizer during the course of training.

See `callbacks.LearningRateScheduler` for a more general implementations.
"""


class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = self.model.optimizer.learning_rate
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        self.model.optimizer.learning_rate = scheduled_lr
        print(
            f"\nEpoch {epoch}: Learning rate is {float(np.array(scheduled_lr))}."
        )


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05),
    (6, 0.01),
    (9, 0.005),
    (12, 0.001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=15,
    verbose=0,
    callbacks=[
        LossAndErrorPrintingCallback(),
        CustomLearningRateScheduler(lr_schedule),
    ],
)

"""
### Built-in Keras callbacks

Be sure to check out the existing Keras callbacks by
reading the [API docs](https://keras.io/api/callbacks/).
Applications include logging to CSV, saving
the model, visualizing metrics in TensorBoard, and a lot more!
"""
