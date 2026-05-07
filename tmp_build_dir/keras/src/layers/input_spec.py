from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export


@keras_export(["keras.InputSpec", "keras.layers.InputSpec"])
class InputSpec:
    """Specifies the rank, dtype and shape of every input to a layer.

    Layers can expose (if appropriate) an `input_spec` attribute:
    an instance of `InputSpec`, or a nested structure of `InputSpec` instances
    (one per input tensor). These objects enable the layer to run input
    compatibility checks for input structure, input rank, input shape, and
    input dtype for the first argument of `Layer.__call__`.

    A `None` entry in a shape is compatible with any dimension.

    Args:
        dtype: Expected dtype of the input.
        shape: Shape tuple, expected shape of the input
            (may include `None` for dynamic axes).
            Includes the batch size.
        ndim: Integer, expected rank of the input.
        max_ndim: Integer, maximum rank of the input.
        min_ndim: Integer, minimum rank of the input.
        axes: Dictionary mapping integer axes to
            a specific dimension value.
        allow_last_axis_squeeze: If `True`, allow inputs of rank N+1 as long
            as the last axis of the input is 1, as well as inputs of rank N-1
            as long as the last axis of the spec is 1.
        name: Expected key corresponding to this input when passing data as
            a dictionary.
        optional: Boolean, whether the input is optional or not.
            An optional input can accept `None` values.

    Example:

    ```python
    class MyLayer(Layer):
        def __init__(self):
            super().__init__()
            # The layer will accept inputs with
            # shape (*, 28, 28) & (*, 28, 28, 1)
            # and raise an appropriate error message otherwise.
            self.input_spec = InputSpec(
                shape=(None, 28, 28, 1),
                allow_last_axis_squeeze=True)
    ```
    """

    def __init__(
        self,
        dtype=None,
        shape=None,
        ndim=None,
        max_ndim=None,
        min_ndim=None,
        axes=None,
        allow_last_axis_squeeze=False,
        name=None,
        optional=False,
    ):
        self.dtype = (
            backend.standardize_dtype(dtype) if dtype is not None else None
        )
        if shape is not None:
            self.shape = backend.standardize_shape(shape)
            self.ndim = len(shape)
        else:
            self.ndim = ndim
            self.shape = None
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.name = name
        self.optional = optional
        self.allow_last_axis_squeeze = allow_last_axis_squeeze
        try:
            axes = axes or {}
            self.axes = {int(k): axes[k] for k in axes}
        except (ValueError, TypeError):
            raise TypeError(
                "Argument `axes` must be a dict with integer keys. "
                f"Received: axes={axes}"
            )

        if self.axes and (self.ndim is not None or self.max_ndim is not None):
            max_dim = (self.ndim if self.ndim else self.max_ndim) - 1
            max_axis = max(self.axes)
            if max_axis > max_dim:
                raise ValueError(
                    "Axis {} is greater than the maximum "
                    "allowed value: {}".format(max_axis, max_dim)
                )

    def __repr__(self):
        spec = [
            (f"dtype={str(self.dtype)}") if self.dtype else "",
            (f"shape={str(self.shape)}") if self.shape else "",
            (f"ndim={str(self.ndim)}") if self.ndim else "",
            (f"max_ndim={str(self.max_ndim)}") if self.max_ndim else "",
            (f"min_ndim={str(self.min_ndim)}") if self.min_ndim else "",
            (f"axes={str(self.axes)}") if self.axes else "",
        ]
        return f"InputSpec({', '.join(x for x in spec if x)})"

    def get_config(self):
        return {
            "dtype": self.dtype,
            "shape": self.shape,
            "ndim": self.ndim,
            "max_ndim": self.max_ndim,
            "min_ndim": self.min_ndim,
            "axes": self.axes,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def assert_input_compatibility(input_spec, inputs, layer_name):
    """Checks compatibility between the layer and provided inputs.

    This checks that the tensor(s) `inputs` verify the input assumptions
    of a layer (if any). If not, a clear and actional exception gets raised.

    Args:
        input_spec: An InputSpec instance, list of InputSpec instances, a nested
            structure of InputSpec instances, or None.
        inputs: Input tensor, list of input tensors, or a nested structure of
            input tensors.
        layer_name: String, name of the layer (for error message formatting).

    Raises:
        ValueError: in case of mismatch between
            the provided inputs and the expectations of the layer.
    """
    if not input_spec:
        return

    input_spec = tree.flatten(input_spec)
    if isinstance(inputs, dict):
        # Flatten `inputs` by reference order if input spec names are provided
        names = [spec.name for spec in input_spec]
        if all(names):
            list_inputs = []
            for name in names:
                if name not in inputs:
                    raise ValueError(
                        f'Missing data for input "{name}". '
                        "You passed a data dictionary with keys "
                        f"{list(inputs.keys())}. "
                        f"Expected the following keys: {names}"
                    )
                list_inputs.append(inputs[name])
            inputs = list_inputs

    inputs = tree.flatten(inputs)
    if len(inputs) != len(input_spec):
        raise ValueError(
            f'Layer "{layer_name}" expects {len(input_spec)} input(s),'
            f" but it received {len(inputs)} input tensors. "
            f"Inputs received: {inputs}"
        )
    for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
        if spec is None:
            continue
        if x is None and spec.optional:
            continue

        # Having a shape/dtype is the only commonality of the various
        # tensor-like objects that may be passed. The most common kind of
        # invalid type we are guarding for is a Layer instance (Functional API),
        # which does not have a `shape` attribute.
        if not hasattr(x, "shape"):
            raise ValueError(
                f"Inputs to a layer should be tensors. Got '{x}' "
                f"(of type {type(x)}) as input for layer '{layer_name}'."
            )

        shape = backend.standardize_shape(x.shape)
        ndim = len(shape)
        # Check ndim.
        if spec.ndim is not None and not spec.allow_last_axis_squeeze:
            if ndim != spec.ndim:
                raise ValueError(
                    f'Input {input_index} of layer "{layer_name}" '
                    "is incompatible with the layer: "
                    f"expected ndim={spec.ndim}, found ndim={ndim}. "
                    f"Full shape received: {shape}"
                )
        if spec.max_ndim is not None:
            if ndim is not None and ndim > spec.max_ndim:
                raise ValueError(
                    f'Input {input_index} of layer "{layer_name}" '
                    "is incompatible with the layer: "
                    f"expected max_ndim={spec.max_ndim}, "
                    f"found ndim={ndim}"
                )
        if spec.min_ndim is not None:
            if ndim is not None and ndim < spec.min_ndim:
                raise ValueError(
                    f'Input {input_index} of layer "{layer_name}" '
                    "is incompatible with the layer: "
                    f"expected min_ndim={spec.min_ndim}, "
                    f"found ndim={ndim}. "
                    f"Full shape received: {shape}"
                )
        # Check dtype.
        if spec.dtype is not None:
            dtype = backend.standardize_dtype(x.dtype)
            if dtype != spec.dtype:
                raise ValueError(
                    f'Input {input_index} of layer "{layer_name}" '
                    "is incompatible with the layer: "
                    f"expected dtype={spec.dtype}, "
                    f"found dtype={dtype}"
                )

        # Check specific shape axes.
        if spec.axes:
            for axis, value in spec.axes.items():
                if value is not None and shape[axis] not in {
                    value,
                    None,
                }:
                    raise ValueError(
                        f'Input {input_index} of layer "{layer_name}" is '
                        f"incompatible with the layer: expected axis {axis} "
                        f"of input shape to have value {value}, "
                        "but received input with "
                        f"shape {shape}"
                    )
        # Check shape.
        if spec.shape is not None:
            spec_shape = spec.shape
            if spec.allow_last_axis_squeeze:
                if shape and shape[-1] == 1:
                    shape = shape[:-1]
                if spec_shape and spec_shape[-1] == 1:
                    spec_shape = spec_shape[:-1]
            for spec_dim, dim in zip(spec_shape, shape):
                if spec_dim is not None and dim is not None:
                    if spec_dim != dim:
                        raise ValueError(
                            f'Input {input_index} of layer "{layer_name}" is '
                            "incompatible with the layer: "
                            f"expected shape={spec.shape}, "
                            f"found shape={shape}"
                        )
