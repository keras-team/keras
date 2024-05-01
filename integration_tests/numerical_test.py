import keras  # isort: skip, keep it on top for torch test

import numpy as np
import tf_keras

keras.backend.set_image_data_format("channels_last")
tf_keras.backend.set_image_data_format("channels_last")

DEBUGGING = True
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 1


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

    return x_train[:100], y_train[:100]


def build_keras_model(keras_module, num_classes):
    input_shape = (28, 28, 1)

    model = keras_module.Sequential(
        [
            keras_module.Input(shape=input_shape),
            keras_module.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu"
            ),
            keras_module.layers.BatchNormalization(),
            keras_module.layers.MaxPooling2D(pool_size=(2, 2)),
            keras_module.layers.Conv2D(
                64, kernel_size=(3, 3), activation="relu"
            ),
            keras_module.layers.BatchNormalization(scale=False, center=True),
            keras_module.layers.MaxPooling2D(pool_size=(2, 2)),
            keras_module.layers.Flatten(),
            keras_module.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def compile_model(model):
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["mae", "accuracy"],
        jit_compile=False,
        run_eagerly=True,
    )


def train_model(model, x, y):
    return model.fit(
        x,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=False,
        verbose=0,
    )


def eval_model(model, x, y):
    score = model.evaluate(x, y, verbose=0, batch_size=BATCH_SIZE)
    print(score)
    return score


def check_history(h1, h2):
    for key in h1.history.keys():
        print(f"{key}:")
        print(h1.history[key])
        print(h2.history[key])
        np.testing.assert_allclose(
            h1.history[key],
            h2.history[key],
            atol=1e-3,
        )


def predict_model(model, x):
    return model.predict(x, batch_size=BATCH_SIZE, verbose=0)


def numerical_test():
    print("Building data and creating models:")
    try:
        x_train, y_train = build_mnist_data(NUM_CLASSES)
        keras_model = build_keras_model(keras, NUM_CLASSES)
        tf_keras_model = build_keras_model(tf_keras, NUM_CLASSES)
        print("Data building and model creation passed.")
    except Exception as e:
        print("Data building and model creation failed with error:", e)
        if not DEBUGGING:
            raise

    print("Setting and checking weights:")
    try:
        weights = [weight.numpy() for weight in keras_model.weights]
        tf_keras_model.set_weights(weights)
        for kw, kcw in zip(keras_model.weights, tf_keras_model.weights):
            np.testing.assert_allclose(kw.numpy(), kcw.numpy())
        print("Weight setting and checking passed.")
    except Exception as e:
        print("Weight setting and checking failed with error:", e)
        if not DEBUGGING:
            raise

    print("Compiling models:")
    try:
        compile_model(keras_model)
        compile_model(tf_keras_model)
        print("Model compilation passed.")
    except Exception as e:
        print("Model compilation failed with error:", e)
        if not DEBUGGING:
            raise

    print("Training models and checking histories:")
    try:
        keras_history = train_model(keras_model, x_train, y_train)
        tf_keras_history = train_model(tf_keras_model, x_train, y_train)
        check_history(keras_history, tf_keras_history)
        print("Training and history checking passed.")
    except Exception as e:
        print("Training and history checking failed with error:", e)
        if not DEBUGGING:
            raise

    print("Checking trained weights:")
    try:
        for kw, kcw in zip(keras_model.weights, tf_keras_model.weights):
            np.testing.assert_allclose(kw.numpy(), kcw.numpy(), atol=1e-3)
        print("Trained weights checking passed.")
    except Exception as e:
        print("Trained weights checking failed with error:", e)
        if not DEBUGGING:
            raise

    print("Predicting with models:")
    try:
        outputs1 = predict_model(keras_model, x_train)
        outputs2 = predict_model(tf_keras_model, x_train)
        np.testing.assert_allclose(outputs1, outputs2, atol=1e-3)
        print("Prediction passed.")
    except Exception as e:
        print("Prediction failed with error:", e)
        if not DEBUGGING:
            raise

    print("Evaluating models:")
    try:
        score1 = eval_model(keras_model, x_train, y_train)
        score2 = eval_model(tf_keras_model, x_train, y_train)
        np.testing.assert_allclose(score1, score2, atol=1e-3)
        print("Evaluation passed.")
    except Exception as e:
        print("Evaluation failed with error:", e)
        if not DEBUGGING:
            raise


if __name__ == "__main__":
    keras.utils.set_random_seed(1337)
    tf_keras.utils.set_random_seed(1337)
    numerical_test()
