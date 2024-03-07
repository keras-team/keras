import re
import string

import numpy as np

from keras import activations
from keras import backend
from keras import constraints
from keras import dtype_policies
from keras import initializers
from keras import ops
from keras import quantizers
from keras import regularizers
from keras.api_export import keras_export
from keras.layers.input_spec import InputSpec
from keras.layers.layer import Layer


@keras_export("keras.layers.EinsumDense")
class EinsumDense(Layer):
    """A layer that uses `einsum` as the backing computation.

    This layer can perform einsum calculations of arbitrary dimensionality.

    Args:
        equation: An equation describing the einsum to perform.
            This equation must be a valid einsum string of the form
            `ab,bc->ac`, `...ab,bc->...ac`, or
            `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
            axis expression sequence.
        output_shape: The expected shape of the output tensor
            (excluding the batch dimension and any dimensions
            represented by ellipses). You can specify `None` for any dimension
            that is unknown or can be inferred from the input shape.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied
            (that is, a "linear" activation: `a(x) = x`).
        bias_axes: A string containing the output dimension(s)
            to apply a bias to. Each character in the `bias_axes` string
            should correspond to a character in the output portion
            of the `equation` string.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix.
        bias_constraint: Constraint function applied to the bias vector.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices
            (the factorization happens on the last dimension).
            This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `EinsumDense` layer by calling `layer.enable_lora(rank)`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Examples:

    **Biased dense layer with einsums**

    This example shows how to instantiate a standard Keras dense layer using
    einsum operations. This example is equivalent to
    `keras.layers.Dense(64, use_bias=True)`.

    >>> layer = keras.layers.EinsumDense("ab,bc->ac",
    ...                                       output_shape=64,
    ...                                       bias_axes="c")
    >>> input_tensor = keras.Input(shape=[32])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 64)

    **Applying a dense layer to a sequence**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence. Here, the `output_shape` has two
    values (since there are two non-batch dimensions in the output); the first
    dimension in the `output_shape` is `None`, because the sequence dimension
    `b` has an unknown shape.

    >>> layer = keras.layers.EinsumDense("abc,cd->abd",
    ...                                       output_shape=(None, 64),
    ...                                       bias_axes="d")
    >>> input_tensor = keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 32, 64)

    **Applying a dense layer to a sequence using ellipses**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence, but uses the ellipsis notation
    instead of specifying the batch and sequence dimensions.

    Because we are using ellipsis notation and have specified only one axis, the
    `output_shape` arg is a single value. When instantiated in this way, the
    layer can handle any number of sequence dimensions - including the case
    where no sequence dimension exists.

    >>> layer = keras.layers.EinsumDense("...x,xy->...y",
    ...                                       output_shape=64,
    ...                                       bias_axes="y")
    >>> input_tensor = keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 32, 64)
    """

    def __init__(
        self,
        equation,
        output_shape,
        activation=None,
        bias_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.equation = equation
        if isinstance(output_shape, int):
            self.partial_output_shape = (output_shape,)
        else:
            self.partial_output_shape = tuple(output_shape)
        self.bias_axes = bias_axes
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_enabled = False

    def build(self, input_shape):
        shape_data = _analyze_einsum_string(
            self.equation,
            self.bias_axes,
            input_shape,
            self.partial_output_shape,
        )
        kernel_shape, bias_shape, full_output_shape = shape_data
        self.full_output_shape = tuple(full_output_shape)
        self._kernel = self.add_weight(
            name="kernel",
            shape=tuple(kernel_shape),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        if bias_shape is not None:
            self.bias = self.add_weight(
                name="bias",
                shape=tuple(bias_shape),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=len(input_shape))
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)
        if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
            self.quantize(self.dtype_policy.quantization_mode)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        if self.lora_enabled:
            return self._kernel + ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )
        return self._kernel

    def compute_output_shape(self, _):
        return self.full_output_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "output_shape": self.partial_output_shape,
            "equation": self.equation,
            "activation": activations.serialize(self.activation),
            "bias_axes": self.bias_axes,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
        return {**base_config, **config}

    def call(self, inputs):
        x = ops.einsum(self.equation, inputs, self.kernel)
        if self.bias is not None:
            x += self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x

    def quantized_call(self, inputs):
        if self.lora_enabled:
            raise ValueError("`quantized_call` doesn't support lora weights")
        inputs, inputs_scale = self.inputs_quantizer(inputs)
        x = ops.einsum(self.equation, inputs, self.kernel)
        # Deal with `inputs_scale`
        inputs_scale = ops.transpose(inputs_scale, self._input_transpose_axes)
        if self._input_expand_axes:
            inputs_scale = ops.expand_dims(
                inputs_scale, axis=self._input_expand_axes
            )
        if self._input_squeeze_axes:
            inputs_scale = ops.squeeze(
                inputs_scale, axis=self._input_squeeze_axes
            )
        # De-scale outputs
        x = ops.cast(x, self.compute_dtype)
        x = ops.divide(x, ops.multiply(inputs_scale, self.kernel_scale))
        if self.bias is not None:
            x += self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        if self.kernel_constraint:
            raise ValueError(
                "Lora is incompatible with kernel constraints. "
                "In order to enable lora on this layer, remove the "
                "`kernel_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. "
                "This can only be done once per layer."
            )
        self._tracker.unlock()
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(self.kernel.shape[:-1] + (rank,)),
            initializer=initializers.get(a_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.kernel.shape[-1]),
            initializer=initializers.get(b_initializer),
            regularizer=self.kernel_regularizer,
        )
        self._kernel.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank

    def quantize(self, mode):
        self._check_quantize_args(mode, self.compute_dtype)
        if mode == "int8":
            if backend.standardize_dtype(self._kernel.dtype) == "int8":
                raise ValueError("`quantize` can only be done once per layer.")
            # Merge lora-related parameters to make use of fully int8 kernel
            self._merge_lora_into_kernel()

            if self.input_spec is None:
                raise ValueError(
                    f"Cannot quantize {self.name} that isn't yet built."
                )
            (
                self._input_reduced_axes,
                self._kernel_reduced_axes,
                self._input_transpose_axes,
                self._kernel_transpose_axes,
                self._input_expand_axes,
                self._kernel_expand_axes,
                self._input_squeeze_axes,
                self._kernel_squeeze_axes,
            ) = _analyze_quantization_info(self.equation, self.input_spec.ndim)
            # Configure `self.inputs_quantizer`
            self.inputs_quantizer = quantizers.AbsMaxQuantizer(
                axis=self._input_reduced_axes
            )
            # Quantize `self._kernel` to int8 and compute corresponding scale
            kernel_value, kernel_scale = quantizers.abs_max_quantize(
                self._kernel, axis=self._kernel_reduced_axes
            )
            kernel_scale = ops.cast(kernel_scale, self.compute_dtype)
            kernel_scale = ops.transpose(
                kernel_scale, self._kernel_transpose_axes
            )
            if self._kernel_expand_axes:
                kernel_scale = ops.expand_dims(
                    kernel_scale, axis=self._kernel_expand_axes
                )
            if self._kernel_squeeze_axes:
                kernel_scale = ops.squeeze(
                    kernel_scale, axis=self._kernel_squeeze_axes
                )
            self._tracker.unlock()
            self._untrack_variable(self._kernel)
            self._kernel = self.add_weight(
                name="kernel",
                shape=self._kernel.shape,
                # Prevent adding a large constant to the computation graph
                initializer=lambda shape, dtype: kernel_value,
                dtype="int8",
                trainable=False,
            )
            self.kernel_scale = self.add_weight(
                name="kernel_scale",
                shape=kernel_scale.shape,
                # Prevent adding a large constant to the computation graph
                initializer=lambda shape, dtype: kernel_scale,
                dtype=self.compute_dtype,
                trainable=False,
            )
            if self.bias is not None:
                self.bias.trainable = False
            self._tracker.lock()
        else:
            NotImplementedError()

        # Set new dtype policy
        if not isinstance(
            self.dtype_policy, dtype_policies.QuantizedDTypePolicy
        ):
            quantized_dtype = f"{mode}_from_{self.dtype_policy.name}"
            self.dtype_policy = dtype_policies.get(quantized_dtype)

    def _merge_lora_into_kernel(self, untrack=False):
        if not self.lora_enabled:
            return
        # Merge lora-enabled kernel into kernel
        self._kernel.assign(self.kernel)
        self.lora_enabled = False
        if untrack:
            self._tracker.unlock()
            self.lora_kernel_a = self._untrack_variable(self.lora_kernel_a)
            self.lora_kernel_b = self._untrack_variable(self.lora_kernel_b)
            self._tracker.lock()
            self.lora_rank = None

    def save_own_variables(self, store):
        if not self.lora_enabled:
            return super().save_own_variables(store)

        kernel_value = self.kernel
        store["0"] = kernel_value
        if self.bias is not None:
            store["1"] = self.bias

    def load_own_variables(self, store):
        if not self.lora_enabled:
            return super().load_own_variables(store)
        self._kernel.assign(store["0"])
        if self.bias is not None:
            self.bias.assign(store["1"])
        self.lora_kernel_a.assign(np.zeros(self.lora_kernel_a.shape))
        self.lora_kernel_b.assign(np.zeros(self.lora_kernel_b.shape))


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
    """Analyzes an einsum string to determine the required weight shape."""

    dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

    # This is the case where no ellipses are present in the string.
    split_string = re.match(
        "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    # This is the case where ellipses are present on the left.
    split_string = re.match(
        "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape, left_elided=True
        )

    # This is the case where ellipses are present on the right.
    split_string = re.match(
        "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    raise ValueError(
        f"Invalid einsum equation '{equation}'. Equations must be in the form "
        "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
    )


def _analyze_split_string(
    split_string, bias_axes, input_shape, output_shape, left_elided=False
):
    """Analyze an pre-split einsum string to find the weight shape."""
    input_spec = split_string.group(1)
    weight_spec = split_string.group(2)
    output_spec = split_string.group(3)
    elided = len(input_shape) - len(input_spec)

    if isinstance(output_shape, int):
        output_shape = [output_shape]
    else:
        output_shape = list(output_shape)

    output_shape.insert(0, input_shape[0])

    if elided > 0 and left_elided:
        for i in range(1, elided):
            # We already inserted the 0th input dimension at dim 0, so we need
            # to start at location 1 here.
            output_shape.insert(1, input_shape[i])
    elif elided > 0 and not left_elided:
        for i in range(len(input_shape) - elided, len(input_shape)):
            output_shape.append(input_shape[i])

    if left_elided:
        # If we have beginning dimensions elided, we need to use negative
        # indexing to determine where in the input dimension our values are.
        input_dim_map = {
            dim: (i + elided) - len(input_shape)
            for i, dim in enumerate(input_spec)
        }
        # Because we've constructed the full output shape already, we don't need
        # to do negative indexing.
        output_dim_map = {
            dim: (i + elided) for i, dim in enumerate(output_spec)
        }
    else:
        input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
        output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

    for dim in input_spec:
        input_shape_at_dim = input_shape[input_dim_map[dim]]
        if dim in output_dim_map:
            output_shape_at_dim = output_shape[output_dim_map[dim]]
            if (
                output_shape_at_dim is not None
                and output_shape_at_dim != input_shape_at_dim
            ):
                raise ValueError(
                    "Input shape and output shape do not match at shared "
                    f"dimension '{dim}'. Input shape is {input_shape_at_dim}, "
                    "and output shape "
                    f"is {output_shape[output_dim_map[dim]]}."
                )

    for dim in output_spec:
        if dim not in input_spec and dim not in weight_spec:
            raise ValueError(
                f"Dimension '{dim}' was specified in the output "
                f"'{output_spec}' but has no corresponding dim in the input "
                f"spec '{input_spec}' or weight spec '{output_spec}'"
            )

    weight_shape = []
    for dim in weight_spec:
        if dim in input_dim_map:
            weight_shape.append(input_shape[input_dim_map[dim]])
        elif dim in output_dim_map:
            weight_shape.append(output_shape[output_dim_map[dim]])
        else:
            raise ValueError(
                f"Weight dimension '{dim}' did not have a match in either "
                f"the input spec '{input_spec}' or the output "
                f"spec '{output_spec}'. For this layer, the weight must "
                "be fully specified."
            )

    if bias_axes is not None:
        num_left_elided = elided if left_elided else 0
        idx_map = {
            char: output_shape[i + num_left_elided]
            for i, char in enumerate(output_spec)
        }

        for char in bias_axes:
            if char not in output_spec:
                raise ValueError(
                    f"Bias dimension '{char}' was requested, but is not part "
                    f"of the output spec '{output_spec}'"
                )

        first_bias_location = min(
            [output_spec.find(char) for char in bias_axes]
        )
        bias_output_spec = output_spec[first_bias_location:]

        bias_shape = [
            idx_map[char] if char in bias_axes else 1
            for char in bias_output_spec
        ]

        if not left_elided:
            for _ in range(elided):
                bias_shape.append(1)
    else:
        bias_shape = None

    return weight_shape, bias_shape, output_shape


def _analyze_quantization_info(equation, input_shape):

    def get_specs(equation, input_shape):
        possible_labels = string.ascii_letters
        dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

        # This is the case where no ellipses are present in the string.
        split_string = re.match(
            "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", dot_replaced_string
        )
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            return input_spec, weight_spec, output_spec

        # This is the case where ellipses are present on the left.
        split_string = re.match(
            "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", dot_replaced_string
        )
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            elided = len(input_shape) - len(input_spec)
            possible_labels = sorted(
                set(possible_labels)
                - set(input_spec)
                - set(weight_spec)
                - set(output_spec)
            )
            # Pad labels on the left to `input_spec` and `output_spec`
            for i in range(elided):
                input_spec = possible_labels[i] + input_spec
                output_spec = possible_labels[i] + output_spec
            return input_spec, weight_spec, output_spec

        # This is the case where ellipses are present on the right.
        split_string = re.match(
            "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", dot_replaced_string
        )
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            elided = len(input_shape) - len(input_spec)
            possible_labels = sorted(
                set(possible_labels)
                - set(input_spec)
                - set(weight_spec)
                - set(output_spec)
            )
            # Pad labels on the right to `input_spec` and `output_spec`
            for i in range(elided):
                input_spec = input_spec + possible_labels[i]
                output_spec = output_spec + possible_labels[i]
            return input_spec, weight_spec, output_spec

        raise ValueError(
            f"Invalid einsum equation '{equation}'. Equations must be in the "
            "form [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
        )

    input_spec, weight_spec, output_spec = get_specs(equation, input_shape)

    # Determine the axes that should be reduced by the quantizer
    input_reduced_axes = []
    weight_reduced_axes = []
    for i, label in enumerate(input_spec):
        index = output_spec.find(label)
        if index == -1:
            input_reduced_axes.append(i)
    for i, label in enumerate(weight_spec):
        index = output_spec.find(label)
        if index == -1:
            weight_reduced_axes.append(i)

    # Determine the axes of `ops.expand_dims`
    input_expand_axes = []
    weight_expand_axes = []
    for i, label in enumerate(output_spec):
        index_input = input_spec.find(label)
        index_weight = weight_spec.find(label)
        if index_input == -1:
            input_expand_axes.append(i)
        if index_weight == -1:
            weight_expand_axes.append(i)

    # Determine the axes of `ops.transpose`
    input_transpose_axes = []
    weight_transpose_axes = []
    for i, label in enumerate(output_spec):
        index_input = input_spec.find(label)
        index_weight = weight_spec.find(label)
        if index_input != -1:
            input_transpose_axes.append(index_input)
        if index_weight != -1:
            weight_transpose_axes.append(index_weight)
    # Postprocess the information:
    # 1. Add dummy axes (1) to transpose_axes
    # 2. Add axis to squeeze_axes if 1. failed
    input_squeeze_axes = []
    weight_squeeze_axes = []
    for ori_index in input_reduced_axes:
        try:
            index = input_expand_axes.pop(0)
        except IndexError:
            input_squeeze_axes.append(ori_index)
        input_transpose_axes.insert(index, ori_index)
    for ori_index in weight_reduced_axes:
        try:
            index = weight_expand_axes.pop(0)
        except IndexError:
            weight_squeeze_axes.append(ori_index)
        weight_transpose_axes.insert(index, ori_index)
    return (
        input_reduced_axes,
        weight_reduced_axes,
        input_transpose_axes,
        weight_transpose_axes,
        input_expand_axes,
        weight_expand_axes,
        input_squeeze_axes,
        weight_squeeze_axes,
    )
