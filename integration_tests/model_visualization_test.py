from pathlib import Path

import keras
from keras.src.utils import plot_model


def assert_file_exists(path):
    assert Path(path).is_file(), "File does not exist"


def test_plot_sequential_model():
    model = keras.Sequential(
        [
            keras.Input((3,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    file_name = "sequential.png"
    plot_model(model, file_name)
    assert_file_exists(file_name)

    file_name = "sequential-show_shapes.png"
    plot_model(model, file_name, show_shapes=True)
    assert_file_exists(file_name)

    file_name = "sequential-show_shapes-show_dtype.png"
    plot_model(
        model,
        file_name,
        show_shapes=True,
        show_dtype=True,
    )
    assert_file_exists(file_name)

    file_name = "sequential-show_shapes-show_dtype-show_layer_names.png"
    plot_model(
        model,
        file_name,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
    )
    assert_file_exists(file_name)

    file_name = "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations.png"  # noqa: E501
    plot_model(
        model,
        file_name,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
    assert_file_exists(file_name)

    file_name = "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png"  # noqa: E501
    plot_model(
        model,
        file_name,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
    )
    assert_file_exists(file_name)

    file_name = "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png"  # noqa: E501
    plot_model(
        model,
        file_name,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
        rankdir="LR",
    )
    assert_file_exists(file_name)

    file_name = "sequential-show_layer_activations-show_trainable.png"
    plot_model(
        model,
        file_name,
        show_layer_activations=True,
        show_trainable=True,
    )
    assert_file_exists(file_name)


def plot_functional_model():
    inputs = keras.Input((3,))
    x = keras.layers.Dense(4, activation="relu", trainable=False)(inputs)
    residual = x
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x += residual
    residual = x
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x += residual
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    plot_model(model, "functional.png")
    plot_model(model, "functional-show_shapes.png", show_shapes=True)
    plot_model(
        model,
        "functional-show_shapes-show_dtype.png",
        show_shapes=True,
        show_dtype=True,
    )
    plot_model(
        model,
        "functional-show_shapes-show_dtype-show_layer_names.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
    )
    plot_model(
        model,
        "functional-show_shapes-show_dtype-show_layer_names-show_layer_activations.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
    plot_model(
        model,
        "functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
    )
    plot_model(
        model,
        "functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
        rankdir="LR",
    )
    plot_model(
        model,
        "functional-show_layer_activations-show_trainable.png",
        show_layer_activations=True,
        show_trainable=True,
    )
    plot_model(
        model,
        "functional-show_shapes-show_layer_activations-show_trainable.png",
        show_shapes=True,
        show_layer_activations=True,
        show_trainable=True,
    )


def plot_subclassed_model():
    class MyModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.dense_1 = keras.layers.Dense(3, activation="relu")
            self.dense_2 = keras.layers.Dense(1, activation="sigmoid")

        def call(self, x):
            return self.dense_2(self.dense_1(x))

    model = MyModel()
    model.build((None, 3))

    plot_model(model, "subclassed.png")
    plot_model(model, "subclassed-show_shapes.png", show_shapes=True)
    plot_model(
        model,
        "subclassed-show_shapes-show_dtype.png",
        show_shapes=True,
        show_dtype=True,
    )
    plot_model(
        model,
        "subclassed-show_shapes-show_dtype-show_layer_names.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
    )
    plot_model(
        model,
        "subclassed-show_shapes-show_dtype-show_layer_names-show_layer_activations.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
    plot_model(
        model,
        "subclassed-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
    )
    plot_model(
        model,
        "subclassed-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
        rankdir="LR",
    )
    plot_model(
        model,
        "subclassed-show_layer_activations-show_trainable.png",
        show_layer_activations=True,
        show_trainable=True,
    )
    plot_model(
        model,
        "subclassed-show_shapes-show_layer_activations-show_trainable.png",
        show_shapes=True,
        show_layer_activations=True,
        show_trainable=True,
    )


def plot_nested_functional_model():
    inputs = keras.Input((3,))
    x = keras.layers.Dense(4, activation="relu")(inputs)
    x = keras.layers.Dense(4, activation="relu")(x)
    outputs = keras.layers.Dense(3, activation="relu")(x)
    inner_model = keras.Model(inputs, outputs)

    inputs = keras.Input((3,))
    x = keras.layers.Dense(3, activation="relu", trainable=False)(inputs)
    residual = x
    x = inner_model(x)
    x += residual
    residual = x
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.Dense(3, activation="relu")(x)
    x += residual
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    plot_model(model, "nested-functional.png", expand_nested=True)
    plot_model(
        model,
        "nested-functional-show_shapes.png",
        show_shapes=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_shapes-show_dtype.png",
        show_shapes=True,
        show_dtype=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_shapes-show_dtype-show_layer_names.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png",  # noqa: E501
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
        rankdir="LR",
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_layer_activations-show_trainable.png",
        show_layer_activations=True,
        show_trainable=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_shapes-show_layer_activations-show_trainable.png",  # noqa: E501
        show_shapes=True,
        show_layer_activations=True,
        show_trainable=True,
        expand_nested=True,
    )


def plot_functional_model_with_splits_and_merges():
    class SplitLayer(keras.Layer):
        def call(self, x):
            return list(keras.ops.split(x, 2, axis=1))

    class ConcatLayer(keras.Layer):
        def call(self, xs):
            return keras.ops.concatenate(xs, axis=1)

    inputs = keras.Input((2,))
    a, b = SplitLayer()(inputs)

    a = keras.layers.Dense(2)(a)
    b = keras.layers.Dense(2)(b)

    outputs = ConcatLayer()([a, b])
    model = keras.Model(inputs, outputs)

    plot_model(model, "split-functional.png", expand_nested=True)
    plot_model(
        model,
        "split-functional-show_shapes.png",
        show_shapes=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "split-functional-show_shapes-show_dtype.png",
        show_shapes=True,
        show_dtype=True,
        expand_nested=True,
    )


if __name__ == "__main__":
    test_plot_sequential_model()
    plot_functional_model()
    plot_subclassed_model()
    plot_nested_functional_model()
    plot_functional_model_with_splits_and_merges()
