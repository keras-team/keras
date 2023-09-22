import keras  # isort: skip, keep it on top for torch test

import numpy as np
from tensorflow import keras as tf_keras

NUM_CLASSES = 10


def build_mnist_data(num_classes):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    return x_train, y_train, x_test, y_test


def build_keras_model(keras_module, num_classes):
    input_shape = (28, 28, 1)

    model = keras_module.Sequential(
        [
            keras_module.Input(shape=input_shape),
            keras_module.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu"
            ),
            keras_module.layers.MaxPooling2D(pool_size=(2, 2)),
            keras_module.layers.Conv2D(
                64, kernel_size=(3, 3), activation="relu"
            ),
            keras_module.layers.MaxPooling2D(pool_size=(2, 2)),
            keras_module.layers.Flatten(),
            keras_module.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    return model


def train_model(model, x, y):
    batch_size = 256
    epochs = 1

    model.compile(
        loss="mse", optimizer="adam", metrics=["accuracy"], jit_compile=False
    )

    return model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        shuffle=False,
    )


def eval_model(model, x, y):
    score = model.evaluate(x, y, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score


def numerical_test():
    x_train, y_train, x_test, y_test = build_mnist_data(NUM_CLASSES)
    keras_model = build_keras_model(keras, NUM_CLASSES)
    tf_keras_model = build_keras_model(tf_keras, NUM_CLASSES)

    # Make sure both model have same weights before training
    weights = [weight.numpy() for weight in keras_model.weights]
    tf_keras_model.set_weights(weights)

    for kw, kcw in zip(keras_model.weights, tf_keras_model.weights):
        np.testing.assert_allclose(kw.numpy(), kcw.numpy())

    keras_history = train_model(keras_model, x_train, y_train)
    tf_keras_history = train_model(tf_keras_model, x_train, y_train)

    for key in keras_history.history.keys():
        np.testing.assert_allclose(
            keras_history.history[key],
            tf_keras_history.history[key],
            atol=1e-3,
        )


if __name__ == "__main__":
    keras.utils.set_random_seed(1337)
    tf_keras.utils.set_random_seed(1337)
    numerical_test()
