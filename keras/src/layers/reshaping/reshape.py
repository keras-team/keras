from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.layers.layer import Layer
from keras.src.ops import operation_utils


@keras_export("keras.layers.Reshape")
class Reshape(Layer):
    """Layer that reshapes inputs into the given shape.

    Args:
        target_shape: Target shape. Tuple of integers, does not include the
            samples dimension (batch size). One element of the `target_shape`
            can be -1 in which case the missing value is inferred from the
            size of the array and remaining dimensions.

    Input shape:
        Arbitrary, but required to be compatible with `target_shape`.

    Output shape:
        `(batch_size, *target_shape)`

    Example:

    >>> x = keras.Input(shape=(12,))
    >>> y = keras.layers.Reshape((3, 4))(x)
    >>> y.shape
    (None, 3, 4)

    >>> # another example with shape inference using `-1` as dimension
    >>> y = keras.layers.Reshape((-1, 2, 2))(x)
    >>> y.shape
    (None, 3, 2, 2)
    """

    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        target_shape = tuple(target_shape)
        # test validity of target_shape
        if target_shape.count(-1) > 1:
            raise ValueError(
                "The `target_shape` argument must not contain more than one "
                f"`-1` value. Received: target_shape={target_shape}"
            )
        self.target_shape = target_shape
        self.built = True

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            *operation_utils.compute_reshape_output_shape(
                input_shape[1:], self.target_shape, "target_shape"
            ),
        )

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        return KerasTensor(
            shape=output_shape, dtype=inputs.dtype, sparse=inputs.sparse
        )

    def call(self, inputs):
        from keras.src import backend

        # Use PyTorch operations during tracing for better compatibility
        if backend.backend() == "torch" and self._is_tracing():
            import torch

            batch_size = inputs.shape[0]

            if -1 in self.target_shape:
                # Calculate unknown dimension
                total_elements = 1
                for dim in inputs.shape[1:]:
                    total_elements *= dim

                known_elements = 1
                for dim in self.target_shape:
                    if dim != -1:
                        known_elements *= dim

                unknown_dim = total_elements // known_elements
                resolved_shape = [
                    unknown_dim if dim == -1 else dim
                    for dim in self.target_shape
                ]
            else:
                resolved_shape = list(self.target_shape)

            return torch.reshape(inputs, [batch_size] + resolved_shape)

        # Fall back to original Keras implementation
        potentially_resolved_target_shape = (
            operation_utils.compute_reshape_output_shape(
                tuple(inputs.shape)[1:], self.target_shape, "target_shape"
            )
        )
        potentially_resolved_target_shape = tuple(
            -1 if d is None else d for d in potentially_resolved_target_shape
        )
        return ops.reshape(
            inputs, (ops.shape(inputs)[0],) + potentially_resolved_target_shape
        )

    def _is_tracing(self):
        """Check if we're in PyTorch tracing/export mode."""
        try:
            import inspect

            import torch

            # Check PyTorch tracing states
            if (
                hasattr(torch.jit, "is_tracing")
                and torch.jit.is_tracing()
                or hasattr(torch, "compiler")
                and hasattr(torch.compiler, "is_compiling")
                and torch.compiler.is_compiling()
                or hasattr(torch, "_C")
                and hasattr(torch._C, "_get_tracing_state")
                and torch._C._get_tracing_state()
            ):
                return True

            # Check call stack for export functions
            frame_names = [frame.function for frame in inspect.stack()]
            export_functions = [
                "export",
                "_export",
                "export_compat",
                "_capture",
                "trace",
            ]
            return any(func in frame_names for func in export_functions)
        except:
            return False

    def get_config(self):
        config = {"target_shape": self.target_shape}
        base_config = super().get_config()
        return {**base_config, **config}
