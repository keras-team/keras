"""Quantization geometry: the layer-side protocol behind the strategy registry.

A layer exposes its quantizable structure through
`Layer._quantization_geometry()`, which returns one of the geometry classes
below (the base `Layer` implementation returns `None`, meaning the layer has
no generic quantization support). The strategies in
`keras.src.quantizers.modes` consume the geometry to build variables, compute
quantized values, and run quantized forward passes, so layer classes hold no
per-mode methods.

Two geometry families exist today:

- Projection: a float kernel contracted against the inputs. A strategy writes
  one projection implementation and the geometry supplies what differs per
  layer: how to contract (a plain matmul for `Dense`,
  `ProjectionGeometry`; an einsum for `EinsumDense`,
  `EinsumProjectionGeometry`, whose axis analysis lives on the layer
  itself and is reached through the geometry's hooks), which axes the
  quantizers reduce over, how a scale lines up with the kernel and with
  the outputs, and the 2D `(rows, columns)` view of an N-D kernel.
- Lookup: a float embeddings table indexed by the inputs. `Embedding` is the
  plain case (`LookupGeometry`); `ReversibleEmbedding` adds a reverse
  projection (`ReversibleLookupGeometry`).

Making a layer quantizable
--------------------------

Return a geometry, and list the modes the layer supports:

```python
class MyProjection(Layer):
    def _quantization_geometry(self):
        return ProjectionGeometry(self)

    @property
    def variable_serialization_spec(self):
        # Doubles as the capability declaration: a mode absent from this
        # mapping is rejected for this layer.
        return {
            None: ["kernel", "bias"],
            "int8": ["kernel", "bias", "kernel_scale"],
        }
```

The geometry is a thin adapter, so the strategies still read
state directly off the layer. Beyond what `Layer` already provides, a
quantizable layer must define:

- Projections: `_kernel` (the float kernel variable), `units`, `bias` and
  `activation` (either may be `None`), and, while LoRA is enabled,
  `lora_enabled`, `lora_kernel_a`, `lora_kernel_b`, `lora_alpha` and
  `lora_rank`. `EinsumProjectionGeometry` additionally relies on the
  equation analysis `EinsumDense` prepares in `_set_quantization_info()`.
- Lookups: `_embeddings`, `input_dim`, `output_dim`, and the
  `lora_embeddings_a` / `lora_embeddings_b` equivalents. A reversible
  lookup adds `tie_weights`, `logit_soft_cap`, and, when untied, the
  `reverse_embeddings` variables.

The rest comes from `Layer` itself: strategies read `compute_dtype`,
`dtype_policy` and `path`, create their quantized variables through
`add_weight`, and re-enter through `Layer.quantized_build`, which routes
straight back to the strategy. A layer never needs to know which mode is
running, and implements none of these itself.

Customizing what a strategy does to a layer
-------------------------------------------

Override a geometry hook rather than a strategy method: the hooks on the
classes below are the only points at which strategies vary per
layer. A layer that owns its own ternarization rule, for example, supplies
it through `ternary_values`, and the ternary strategy needs no knowledge of
the layer.

Two things this protocol deliberately does not offer. A layer cannot
override one strategy's math for itself alone, because that surface lives
on the strategy; a layer that contracts its kernel differently overrides
the geometry hooks, and anything beyond that means replacing the strategy (by
subclassing it, overriding the one handler, and registering it under a
new mode name). A new geometry family, on the other hand, needs no dispatcher
change at all: declare its `family` and implement the strategy's
`_build_<family>`, `_call_<family>` and `_quantize_<family>` methods.
"""

import string

import numpy as np

from keras.src import ops


class QuantizationGeometry:
    """Base class for a layer's quantization geometry.

    A geometry names the *family* it belongs to. Strategies
    implement one `_build_<family>`, `_call_<family>` and
    `_quantize_<family>` method per family they support, so introducing a
    family is a declaration plus those methods, with no dispatch chain to
    edit anywhere.
    """

    # Dispatch key for building and quantizing, and for the forward pass
    # unless `call_family` overrides it.
    family = None
    # Forward-pass dispatch key, when the forward pass needs a different
    # implementation from build/quantize (a reversible lookup does).
    call_family = None
    # Whether the layer also projects back through its weight.
    reversible = False

    def __init__(self, layer):
        self.layer = layer

    @property
    def weight_shape(self):
        """Shape of the float weight that quantization replaces."""
        raise NotImplementedError(
            f"{type(self).__name__} must define `weight_shape`."
        )


class ProjectionGeometry(QuantizationGeometry):
    """Geometry of a 2D kernel `(input_dim, units)` contracted by matmul."""

    family = "projection"

    @property
    def weight_shape(self):
        """Shape of the float weight that quantization replaces."""
        return self.layer._kernel.shape

    def prepare(self):
        """Computes any layout analysis the geometry needs (idempotent)."""

    def calibration_rows_columns(self, kernel_shape):
        """2D `(rows, columns)` view used by the calibration strategies.

        Kept apart from `rows_columns`: the calibration strategies may split
        a kernel by a different rule than the weight-only ones.
        """
        return kernel_shape[0], kernel_shape[1]

    def store_unpacked_columns(self, mode, columns):
        """Records the unpacked column count for the calibration call path."""
        del mode, columns  # The matmul case reads `layer.units` instead.

    def unpacked_columns(self, mode):
        """The unpacked column count recorded at calibration build time."""
        del mode
        return self.layer.units

    def contract(self, inputs, kernel):
        """Contracts `inputs` against a kernel in the contraction shape."""
        return ops.matmul(inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        """Gradient of `contract` with respect to its inputs."""
        return ops.matmul(upstream, ops.transpose(float_kernel))

    def reshape_kernel(self, kernel):
        """Restores a 2D dequantized kernel to the contraction shape."""
        return kernel

    def record_kernel_shape(self, kernel_shape):
        """Records the float kernel shape for a later reshape or write-back."""
        self.layer.kernel_shape = kernel_shape

    def rows_columns(self, kernel_shape):
        """2D `(rows, columns)` view: contracted axes times the rest."""
        return kernel_shape[0], kernel_shape[1]

    @property
    def kernel_reduced_axes(self):
        """Kernel axes a weight quantizer reduces over."""
        return 0

    @property
    def inputs_quantization_axis(self):
        """Input axes an activation quantizer reduces over."""
        return -1

    def align_inputs_scale(self, scale):
        """Aligns an activation scale with the contraction's outputs."""
        return scale

    def kernel_scale_shape(self, kernel_shape):
        """Shape of a per-channel scale stored alongside the kernel."""
        return (kernel_shape[1],)

    def kernel_scale_for_storage(self, scale):
        """Aligns a freshly computed kernel scale with its stored layout."""
        return ops.squeeze(scale, axis=0)

    def kernel_scale_for_dequant(self, scale):
        """Aligns the stored kernel scale with the kernel for dequantization."""
        return scale

    def add_lora_delta(self, inputs, x):
        """Adds the LoRA update to the contraction's output, when enabled."""
        layer = self.layer
        if layer.lora_enabled:
            lora_x = ops.matmul(inputs, layer.lora_kernel_a)
            lora_x = ops.matmul(lora_x, layer.lora_kernel_b)
            x = ops.add(x, (layer.lora_alpha / layer.lora_rank) * lora_x)
            x = ops.cast(x, layer.compute_dtype)
        return x

    def ternary_values(self):
        """Returns `(ternary_kernel, scale)` for ternary quantization.

        The default applies the BitNet b1.58 rule to the float kernel:
        `threshold = 0.5 * mean(|W|)` and `scale = mean(|W|)`. A layer that
        owns its own ternarization rule overrides this in its geometry.
        """
        kernel = self.layer._kernel
        kernel_np = ops.convert_to_numpy(kernel)
        abs_k = ops.convert_to_numpy(ops.abs(kernel))
        t = float(ops.convert_to_numpy(ops.mean(abs_k))) * 0.5
        kernel_ternary = np.sign(kernel_np) * (abs_k > t).astype(
            kernel_np.dtype
        )
        beta = float(np.mean(abs_k))
        return kernel_ternary, beta


def _lora_equations(equation):
    """The two einsums that apply a LoRA update to `equation` in low-rank
    form, contracting the rank axis by name.

    `lora_kernel_a` carries the rank on the kernel's last axis and
    `lora_kernel_b` maps it to that axis's size. Contracting the rank with
    a matmul would only work when the kernel's last axis is also the last
    axis of the output; naming it works for every equation (an ellipsis in
    the output, a permuted output, or a kernel whose last axis is
    contracted away).

    Returns:
        `(first, second, a_first)`: `first` contracts the inputs against
        the factor that shares their subscripts, `second` contracts the
        rank axis against the other factor; `a_first` is whether that
        order is `(lora_kernel_a, lora_kernel_b)`, which holds when the
        kernel's last axis survives in the output.
    """
    inputs_spec, rest = equation.split(",")
    kernel_spec, output_spec = rest.split("->")
    last = kernel_spec[-1]
    rank = next(c for c in string.ascii_letters if c not in equation)
    if last in output_spec:
        mid = output_spec.replace(last, rank)
        return (
            f"{inputs_spec},{kernel_spec[:-1]}{rank}->{mid}",
            f"{mid},{rank}{last}->{output_spec}",
            True,
        )
    mid = inputs_spec.replace(last, rank)
    return (
        f"{inputs_spec},{rank}{last}->{mid}",
        f"{mid},{kernel_spec[:-1]}{rank}->{output_spec}",
        False,
    )


class EinsumProjectionGeometry(ProjectionGeometry):
    """Geometry of an N-D einsum kernel (`EinsumDense`).

    The equation-derived axis analysis (reduced/transpose/expand/squeeze
    axes, the custom-gradient equation) is the layer's own geometry
    implementation; this class routes the strategies to it.
    """

    def prepare(self):
        self.layer._set_quantization_info()

    def calibration_rows_columns(self, kernel_shape):
        if len(kernel_shape) == 2:
            return kernel_shape[0], kernel_shape[1]
        # 3D kernels are split by locating the model dimension (the largest
        # one): [d_model, heads, head_dim] is a QKV projection, while
        # [heads, head_dim, d_model] is an attention output projection.
        shape = list(kernel_shape)
        d_model_dim_index = shape.index(max(shape))
        if d_model_dim_index == 0:  # QKV projection case
            in_features, heads, head_dim = shape
            return in_features, heads * head_dim
        elif d_model_dim_index in [1, 2]:  # Attention Output case
            heads, head_dim, out_features = shape
            return heads * head_dim, out_features
        raise ValueError("Could not determine row/column split.")

    def store_unpacked_columns(self, mode, columns):
        setattr(self.layer, f"{mode}_unpacked_column_size", columns)

    def unpacked_columns(self, mode):
        return getattr(self.layer, f"{mode}_unpacked_column_size")

    def contract(self, inputs, kernel):
        return ops.einsum(self.layer.equation, inputs, kernel)

    def contract_grad(self, upstream, float_kernel):
        # From https://stackoverflow.com/a/47609896
        return ops.einsum(
            self.layer._custom_gradient_equation, upstream, float_kernel
        )

    def reshape_kernel(self, kernel):
        return ops.reshape(kernel, self.layer.original_kernel_shape)

    def record_kernel_shape(self, kernel_shape):
        self.layer.original_kernel_shape = kernel_shape

    def rows_columns(self, kernel_shape):
        rows = 1
        columns = 1
        for i, dim in enumerate(kernel_shape):
            if i in self.layer._kernel_reduced_axes:
                rows *= dim
            else:
                columns *= dim
        return rows, columns

    @property
    def kernel_reduced_axes(self):
        return self.layer._kernel_reduced_axes

    @property
    def inputs_quantization_axis(self):
        return tuple(self.layer._input_reduced_axes)

    def align_inputs_scale(self, scale):
        return self.layer._adjust_scale_for_quant(scale, "input")

    def kernel_scale_shape(self, kernel_shape):
        return self.layer._get_kernel_scale_shape(kernel_shape)

    def kernel_scale_for_storage(self, scale):
        return self.layer._adjust_scale_for_quant(scale, "kernel")

    def kernel_scale_for_dequant(self, scale):
        return self.layer._adjust_scale_for_dequant(scale)

    def add_lora_delta(self, inputs, x):
        layer = self.layer
        if layer.lora_enabled:
            first, second, a_first = _lora_equations(layer.equation)
            factors = (layer.lora_kernel_a, layer.lora_kernel_b)
            if not a_first:
                factors = factors[::-1]
            lora_x = ops.einsum(first, inputs, factors[0])
            lora_x = ops.einsum(second, lora_x, factors[1])
            x = ops.add(x, (layer.lora_alpha / layer.lora_rank) * lora_x)
            x = ops.cast(x, dtype=layer.compute_dtype)
        return x


class LookupGeometry(QuantizationGeometry):
    """Geometry of an embeddings table indexed by integer inputs."""

    family = "lookup"

    @property
    def weight_shape(self):
        return (self.layer.input_dim, self.layer.output_dim)


class ReversibleLookupGeometry(LookupGeometry):
    """Lookup geometry with a reverse projection (`ReversibleEmbedding`)."""

    call_family = "reversible_lookup"
    reversible = True
