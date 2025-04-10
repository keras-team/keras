from collections import namedtuple

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state


@keras_export("keras.RematScope")
class RematScope:
    """A context manager for enabling rematerialization in Keras.

    Rematerialization (gradient checkpointing) trades memory for computation by
    recomputing intermediate activations during the backward pass. This is
    particularly useful for training large models or large batch sizes within
    limited memory constraints.

    This should be used when initializing the layer (e.g., `layer(input)`).
    Rematerialization applies at execution time, not at creation time.

    Args:
        mode: Rematerialization mode to apply.
            Options:
            - `"full"`: Apply rematerialization globally to all supported
              operations.
            - `"activations"`: Apply rematerialization to activations on any
              layers that contain `keras.activations` (e.g., `Dense(...,
              activation=relu)`).
            - `"larger_than"`: Apply rematerialization to layers with output
              sizes larger than `output_size_threshold`.
            - `"list_of_layers"`: Apply rematerialization to a specific list of
              layer names.
            - `None`: Disable rematerialization.
        output_size_threshold: Output size threshold for the
            `"larger_than"` mode. Layers producing outputs larger than this
            threshold will be rematerialized. Default is `1024`.
        layer_names: List of layer names for the
            `"list_of_layers"` mode. Default is an empty list.

    Examples:
    Using "list_of_layers" mode:

    ```python
    from keras import RematScope
    input_tensor = tf.random.normal((1, 32, 32, 3))
    with RematScope(mode="list_of_layers", layer_names=["dense_1",
    "conv2d_1"]):
        layer1 = keras.layers.Dense(128, name="dense_1")
        layer2 = keras.layers.Conv2D(64, (3, 3), name="conv2d_1")
        layer3 = keras.layers.Dense(64, name="dense_2")
        # Only layer1 and layer2 will apply rematerialization
        output1 = layer1(input_tensor)
        output2 = layer2(output1)
        output3 = layer3(output2)
    ```

    Using "larger_than" mode with a specific output size threshold:

    ```python
    with RematScope(mode="larger_than", output_size_threshold=2048):
        layer = keras.layers.Conv2D(64, (3, 3))
        output = layer(input_tensor)  # Conv2D outputs larger than 2048
    ```

    Nested scopes for fine-grained control:

    ```python
    with RematScope(mode="full"):
        # Create layers
        layer1 = keras.layers.Dense(128, activation='relu')
        output1 = layer1(input_tensor)  # layer1 is fully rematerialized
        with RematScope(mode="larger_than", output_size_threshold=512):
            layer2 = keras.layers.Conv2D(32, (3, 3))
            output2 = layer2(output1) # layer2 is conditionally rematerialized
            # if output > 512
    ```
    """

    def __init__(
        self, mode="full", output_size_threshold=1024, layer_names=None
    ):
        if mode not in {
            "full",
            "activations",
            "larger_than",
            "list_of_layers",
            None,
        }:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes are: "
                "'full', 'activations', 'larger_than', 'list_of_layers', or "
                " None."
            )
        self.mode = mode
        self.output_size_threshold = output_size_threshold
        self.layer_names = layer_names or []
        self._pop_on_exit = False

    def __enter__(self):
        remat_scope_stack = global_state.get_global_attribute(
            "remat_scope_stack", default=[], set_to_default=True
        )
        remat_scope_stack.append(self)
        self._pop_on_exit = True
        return self

    def __exit__(self, *args, **kwargs):
        if self._pop_on_exit:
            remat_scope_stack = global_state.get_global_attribute(
                "remat_scope_stack"
            )
            remat_scope_stack.pop()


RematMode = namedtuple(
    "RematMode", ["mode", "output_size_threshold", "layer_names"]
)


def get_current_remat_mode():
    """Get the current rematerialization mode and associated settings.

    Returns:
        RematMode or None: The current rematerialization mode, or None if not
        set.
    """
    remat_scope_stack = global_state.get_global_attribute("remat_scope_stack")
    if not remat_scope_stack:
        return None
    active_scope = remat_scope_stack[-1]
    return RematMode(
        active_scope.mode,
        active_scope.output_size_threshold,
        active_scope.layer_names,
    )


@keras_export("keras.remat")
def remat(f):
    """Applies rematerialization to a function or layer for memory optimization.

    Rematerialization is a memory optimization technique that trades off
    computation for memory. Instead of storing intermediate results
    (e.g. activations) for backpropagation, they are recomputed during the
    backward pass. This reduces peak memory usage at the cost of increased
    computation time, allowing the training of larger models or using larger
    batch sizes within the same memory constraints.

    Args:
        f: A callable function, to which rematerialization is
           applied. This is typically a computationally expensive operation
           where intermediate states can be recomputed instead of stored.

    Returns:
        A wrapped function that applies rematerialization. The returned
        function defines a custom gradient, ensuring that during the backward
        pass, the forward computation is recomputed as needed.

    Example:

    ```python
    from keras import Model
    class CustomRematLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.remat_function = remat(self.intermediate_function)

        def intermediate_function(self, x):
            for _ in range(2):
                x = x + x * 0.1  # Simple scaled transformation
            return x

        def call(self, inputs):
            return self.remat_function(inputs)

    # Define a simple model using the custom layer
    inputs = layers.Input(shape=(4,))
    x = layers.Dense(4, activation="relu")(inputs)
    x = CustomRematLayer()(x)  # Custom layer with rematerialization
    outputs = layers.Dense(1)(x)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="mse")
    ```
    """
    return backend.core.remat(f)
