"""Quantization geometry: the layer-side protocol behind the mode registry.

A layer exposes its quantizable structure through
`Layer._quantization_geometry()`, which returns one of the geometry classes
below (the base `Layer` implementation returns `None`, meaning the layer has
no generic quantization support). The strategies in
`keras.src.quantizers.modes` consume the geometry to build variables, compute
quantized values, and run quantized forward passes, so layer classes hold no
per-mode methods.

Projections are the family so far: a float kernel contracted against the
inputs. A mode writes one projection implementation and the geometry
supplies what differs per layer: how to contract, which axes the
quantizers reduce over, how a scale lines up with the kernel and with the
outputs, and the 2D `(rows, columns)` view of the kernel.
`ProjectionGeometry` holds the plain-matmul answers (`Dense`); a layer
that contracts its kernel differently subclasses it and overrides those
hooks. Further families are added as their layers move onto the protocol.

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

The geometry is a thin adapter, so the mode implementations still read
state directly off the layer. Beyond what `Layer` already provides, a
quantizable layer must define:

- Projections: `_kernel` (the float kernel variable), `units`, `bias` and
  `activation` (either may be `None`), and, while LoRA is enabled,
  `lora_enabled`, `lora_kernel_a`, `lora_kernel_b`, `lora_alpha` and
  `lora_rank`.

The rest comes from `Layer` itself: modes read `compute_dtype`,
`dtype_policy` and `path`, create their quantized variables through
`add_weight`, and re-enter through `Layer.quantized_build`, which routes
straight back to the mode. A layer never needs to know which mode is
running, and implements none of these itself.

Customizing what a mode does to a layer
---------------------------------------

Override a geometry hook rather than a mode method: the hooks on the
classes below are the only points at which mode implementations vary per
layer. A layer that owns its own ternarization rule, for example, supplies
it through `ternary_values`, and the ternary mode needs no knowledge of
the layer.

Two things this protocol deliberately does not offer. A layer cannot
override one mode's math for itself alone, because that surface moved onto
the strategies; a layer that contracts its kernel differently overrides
the geometry hooks, and anything beyond that means replacing the mode (by
subclassing it, overriding the one handler, and registering it under a
new name). A new geometry family, on the other hand, needs no dispatcher
change at all: declare its `family` and implement the mode's
`_build_<family>`, `_call_<family>` and `_quantize_<family>` methods.
"""

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

    # Dispatch key for building, quantizing and the forward pass.
    family = None

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
        """2D `(rows, columns)` view used by the calibration modes.

        Kept apart from `rows_columns`: the calibration modes may split a
        kernel by a different rule than the weight-only modes.
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
