"""Model where almost everything is implemented from scratch.

- Custom layers
- Custom model subclass
- Custom train_step and test_step
- Custom compile()
- Custom learning rate schedule
- Custom metrics
"""

import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

INPUT_DIM = 32
NUM_CLASSES = 5


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size, INPUT_DIM)),
        InputSpec((batch_size, NUM_CLASSES)),
    )


def get_input_preprocessor():
    return None


class Linear(keras.layers.Layer):
    def __init__(self, units=32, name=None):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="w",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
            name="b",
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class BinaryTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name="binary_true_positives", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        self.true_positives.assign(0)


class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.btp_metric = BinaryTruePositives(name="mae")

        self.linear_1 = Linear(32, name="linear_1")
        self.linear_2 = Linear(NUM_CLASSES, name="linear_2")

    def call(self, inputs, training=False):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = keras.losses.mean_squared_error(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.btp_metric.update_state(y, y_pred)
        return {
            "loss": self.loss_tracker.result(),
            "btp": self.btp_metric.result(),
        }

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=True)
        loss = keras.losses.mean_squared_error(y, y_pred)
        self.loss_tracker.update_state(loss)
        self.btp_metric.update_state(y, y_pred)
        return {
            "loss": self.loss_tracker.result(),
            "btp": self.btp_metric.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.btp_metric]


class CustomLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.initial_learning_rate / tf.cast(step + 1, "float32")

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
        }


def custom_loss(y_true, y_pred):
    return keras.losses.mse(y_true, y_pred)


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    model = CustomModel()
    if build:
        model(tf.zeros((1, INPUT_DIM)))
    if compile:
        model.compile(
            optimizer=keras.optimizers.Adam(CustomLRSchedule(0.1)),
            loss=custom_loss,
            jit_compile=jit_compile,
        )
    return model


def get_custom_objects():
    return {
        "Linear": Linear,
        "CustomLRSchedule": CustomLRSchedule,
        "CustomModel": CustomModel,
        "BinaryTruePositives": BinaryTruePositives,
        "custom_loss": custom_loss,
    }
