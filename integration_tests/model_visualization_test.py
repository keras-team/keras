import keras_core
from keras_core.utils.model_visualization import plot_model


def plot_sequential_model():
    model = keras_core.Sequential(
        [
            keras_core.Input((3,)),
            keras_core.layers.Dense(4, activation="relu"),
            keras_core.layers.Dense(1, activation="sigmoid"),
        ]
    )
    plot_model(model, "sequential.png")
    plot_model(model, "sequential-show_shapes.png", show_shapes=True)
    plot_model(
        model,
        "sequential-show_shapes-show_dtype.png",
        show_shapes=True,
        show_dtype=True,
    )
    plot_model(
        model,
        "sequential-show_shapes-show_dtype-show_layer_names.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
    )
    plot_model(
        model,
        "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
    plot_model(
        model,
        "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
    )
    plot_model(
        model,
        "sequential-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
        rankdir="LR",
    )
    plot_model(
        model,
        "sequential-show_layer_activations-show_trainable.png",
        show_layer_activations=True,
        show_trainable=True,
    )


def plot_functional_model():
    inputs = keras_core.Input((3,))
    x = keras_core.layers.Dense(4, activation="relu", trainable=False)(inputs)
    residual = x
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x += residual
    residual = x
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x += residual
    x = keras_core.layers.Dropout(0.5)(x)
    outputs = keras_core.layers.Dense(1, activation="sigmoid")(x)

    model = keras_core.Model(inputs, outputs)
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
        "functional-show_shapes-show_dtype-show_layer_names-show_layer_activations.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
    plot_model(
        model,
        "functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
    )
    plot_model(
        model,
        "functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png",
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
    class MyModel(keras_core.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.dense_1 = keras_core.layers.Dense(3, activation="relu")
            self.dense_2 = keras_core.layers.Dense(1, activation="sigmoid")

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
        "subclassed-show_shapes-show_dtype-show_layer_names-show_layer_activations.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
    plot_model(
        model,
        "subclassed-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
    )
    plot_model(
        model,
        "subclassed-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png",
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
    inputs = keras_core.Input((3,))
    x = keras_core.layers.Dense(4, activation="relu")(inputs)
    x = keras_core.layers.Dense(4, activation="relu")(x)
    outputs = keras_core.layers.Dense(4, activation="relu")(x)
    inner_model = keras_core.Model(inputs, outputs)

    inputs = keras_core.Input((3,))
    x = keras_core.layers.Dense(4, activation="relu", trainable=False)(inputs)
    residual = x
    x = inner_model(x)
    x += residual
    residual = x
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x = keras_core.layers.Dense(4, activation="relu")(x)
    x += residual
    x = keras_core.layers.Dropout(0.5)(x)
    outputs = keras_core.layers.Dense(1, activation="sigmoid")(x)
    model = keras_core.Model(inputs, outputs)

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
        "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
        expand_nested=True,
    )
    plot_model(
        model,
        "nested-functional-show_shapes-show_dtype-show_layer_names-show_layer_activations-show_trainable-LR.png",
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
        "nested-functional-show_shapes-show_layer_activations-show_trainable.png",
        show_shapes=True,
        show_layer_activations=True,
        show_trainable=True,
        expand_nested=True,
    )


if __name__ == "__main__":
    plot_sequential_model()
    plot_functional_model()
    plot_subclassed_model()
    plot_nested_functional_model()
