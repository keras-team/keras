"""The per-layer object of a calibration mode.

A `CalibrationRun` creates one `Calibrator` per float layer of a block, in
the layer's stage, passes every input the layer sees during the sweep of
that stage to `observe`, then calls `quantize`, which solves for the
layer's codes and swaps them in through the mode's strategy.
`GPTQCalibrator` (a Hessian) and `AWQCalibrator` (activation magnitudes) are
the calibrators of the built-in modes.
"""

import math

from keras.src import ops
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.geometry import ProjectionGeometry


class Calibrator:
    """Per-layer statistics and solve of one calibration mode.

    The constructor resolves the layer's `ContractionView`. `rows` is the
    number of contracted features of one input sample, and `batch` the
    number of independent problems that a kernel axis shared with the
    inputs splits the kernel into. A statistic has a leading problem axis
    only when `batch > 1`. `num_samples` counts the input samples observed
    so far, per problem. Subclasses implement `observe` and `_solve`, and
    set `mode`; a subclass that sets `warn_tokens_per_row` also words the
    warning in its classmethod `undersampling_warning(layers)`.

    Args:
        layer: A float layer with a projection geometry (`Dense`,
            `EinsumDense`) that supports the mode.
        config: The mode's config object. The solve reads its parameters,
            and the swap builds the layer's variables from it.
    """

    # The calibration mode the calibrator solves for; its strategy writes
    # the solved codes back.
    mode = None

    # Warn after the run when a layer saw fewer calibration tokens than
    # this per kernel row (input feature); `None` never warns.
    warn_tokens_per_row = None

    def __init__(self, layer, config):
        strategy = strategy_registry.get_strategy(self.mode)
        self.strategy = strategy
        self.layer = layer
        self.config = config
        self.num_samples = 0
        geometry = layer._quantization_geometry()
        if not isinstance(
            geometry, ProjectionGeometry
        ) or not layer._supports_quantization_mode(strategy):
            raise TypeError(
                f"Unsupported layer type for {strategy.name.upper()}: "
                f"{type(layer)}"
            )
        layer._check_quantized_build_attributes(geometry)
        self.view = geometry.contraction_view()
        self.batch = self.view.batch
        self.rows = self.view.rows

    def observe(self, inputs):
        """Accumulates statistics from one batch of the layer's inputs."""
        raise NotImplementedError

    def _inputs_view(self, inputs):
        """Validates `inputs` and lays them out through the view.

        Returns `float32` `(samples, rows)`, or `(batch, samples, rows)`
        when the view has several problems.
        """
        if inputs is None:
            raise ValueError("Input tensor cannot be None.")
        if len(inputs.shape) < 2:
            raise ValueError(
                "Input tensor must have rank >= 2 "
                f"(got rank {len(inputs.shape)})."
            )
        if ops.size(inputs) == 0:
            raise ValueError("Input tensor cannot be empty.")
        input_features = self.view.input_features(inputs)
        if input_features != self.rows:
            raise ValueError(
                f"Calibration statistics ({self.rows}) do not match input "
                f"features ({input_features})."
            )
        # A product over no axes is 1: inputs without batch axes hold one
        # problem.
        rank = len(inputs.shape)
        input_problems = math.prod(
            inputs.shape[axis % rank] for axis in self.view.input_batch_axes
        )
        if input_problems != self.batch:
            raise ValueError(
                f"The contraction view has {self.batch} problems, but the "
                f"inputs carry {input_problems} along their batch axes "
                f"{self.view.input_batch_axes}."
            )
        return ops.cast(self.view.inputs_to_view(inputs), "float32")

    def _per_problem(self, shape):
        """`shape` with a leading problem axis when there are several."""
        return shape if self.batch == 1 else (self.batch,) + shape

    def _problem(self, statistic, index):
        """Problem `index`'s slice of a per-problem statistic."""
        return statistic if self.batch == 1 else statistic[index]

    def quantize(self):
        """Solves each problem for its codes and swaps them in."""
        # The solve runs in float32 whatever the layer's variable dtype.
        kernel = ops.cast(self.layer.kernel, "float32")
        kernel = self.view.kernel_to_view(kernel)
        results = [
            self._solve(ops.transpose(kernel[index]), index)
            for index in range(self.batch)
        ]
        self.strategy.write_back(
            self.layer, self.config, *self._stack_problems(results)
        )

    def _solve(self, weights, index):
        """Quantizes `weights`, problem `index`'s `[out, in]` kernel.

        Returns:
            `(codes, scale, zero, g_idx, input_scales)`: the codes
            `[out, in]`, the scale and zero point `[out, n_groups]`, the
            group index `[in]`, and the per-input-row scales `[in]` that
            multiplied the weights before quantization, or `None`.
        """
        raise NotImplementedError

    @staticmethod
    def _stack_problems(results):
        """Stacks the problems' `_solve` results along the stored rows.

        The solvers work on `[out, in]`; the layer stores the view's
        `[in, out]` orientation with the group parameters as
        `[n_groups, out]`, so the forward pass never transposes. Each
        problem's groups are numbered after the previous problem's.
        """
        codes, scales, zeros, group_indices, input_scales = [], [], [], [], []
        for index, (code, scale, zero, g_idx, in_scales) in enumerate(results):
            n_groups = ops.shape(scale)[1]
            codes.append(ops.transpose(code))
            scales.append(ops.transpose(scale))
            zeros.append(ops.transpose(zero))
            group_indices.append(ops.add(g_idx, index * n_groups))
            if in_scales is not None:
                input_scales.append(in_scales)
        return (
            ops.concatenate(codes, axis=0),
            ops.concatenate(scales, axis=0),
            ops.concatenate(zeros, axis=0),
            ops.concatenate(group_indices, axis=0),
            ops.concatenate(input_scales, axis=0) if input_scales else None,
        )
