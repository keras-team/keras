import numpy as np
import tensorflow as tf

import keras
from keras.src import layers
from keras.src import losses
from keras.src import metrics
from keras.src import models
from keras.src import optimizers
from keras.src.callbacks import LearningRateScheduler


def test_model_fit():
    cpus = tf.config.list_physical_devices("CPU")
    tf.config.set_logical_device_configuration(
        cpus[0],
        [
            tf.config.LogicalDeviceConfiguration(),
            tf.config.LogicalDeviceConfiguration(),
        ],
    )

    keras.utils.set_random_seed(1337)

    strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
    with strategy.scope():
        inputs = layers.Input((100,), batch_size=32)
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(16)(x)
        model = models.Model(inputs, outputs)

    callbacks = [LearningRateScheduler(lambda _: 0.1)]

    model.summary()

    x = np.random.random((5000, 100))
    y = np.random.random((5000, 16))
    batch_size = 32
    epochs = 2

    # Fit from numpy arrays:
    with strategy.scope():
        model.compile(
            optimizer=optimizers.LossScaleOptimizer(
                optimizers.SGD(learning_rate=0.001, momentum=0.01)
            ),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        history = model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
        )

    print("History:")
    print(history.history)

    # Fit again from distributed dataset:
    with strategy.scope():
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        dataset = strategy.experimental_distribute_dataset(dataset)
        history = model.fit(dataset, epochs=epochs, callbacks=callbacks)

    print("History:")
    print(history.history)


if __name__ == "__main__":
    test_model_fit()
