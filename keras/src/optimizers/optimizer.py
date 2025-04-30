from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from keras.src.backend.tensorflow.optimizer import (
        TFOptimizer as BackendOptimizer,
    )
elif backend.backend() == "torch":
    from keras.src.backend.torch.optimizers import (
        TorchOptimizer as BackendOptimizer,
    )
elif backend.backend() == "jax":
    from keras.src.backend.jax.optimizer import JaxOptimizer as BackendOptimizer
else:

    class BackendOptimizer(base_optimizer.BaseOptimizer):
        pass


@keras_export(["keras.Optimizer", "keras.optimizers.Optimizer"])
class Optimizer(BackendOptimizer, base_optimizer.BaseOptimizer):
    pass


@keras_export("keras.optimizers.VariableUpdater")
class VariableUpdater:
    """Allows special handling of variable updates."""

    def build(self, optimizer, variable):
        """Set up any state that might depend on the optimizer.

        This may add variables directly to the optimizer for updating state.

        Args:
          optimizer: The optimizer used to update the variables during training.
          variable: Variable to update.
        """
        pass

    def update_step(self, gradient, variable):
        """Update the variable state using the supplied gradient.

        Args:
          gradient: Gradient for the variable.
          variable: Variable to update.
        """
        pass

    def finalize_variable_value(self, variable):
        """Set the final value of the trainable variable.

        Sometimes there are some extra steps before ending the variable updates,
        such as overriding the model variables with its average value.

        Args:
          variable: Variable to finalize.
        """
        pass


class OverwriteScaleWithGradientUpdater(VariableUpdater):
    """Special variable update handler for float8 quantization scales.

    The "gradient" of the scale factor (scale, amax_history) is actually the
    updated scale to assign to the variable.  Supports gradient accumulation
    steps, in which the maximum scale factor between intermediate gradient
    steps is recorded.
    """

    def build(self, optimizer, variable):
        # Keep reference copy of iterations so we can update gradient
        # accumulators appropriately.
        self._iterations = optimizer._iterations
        # Support gradient accumulation by adding an accumulator directly
        # to the optimizer.
        self._gradient_accumulation_steps = (
            optimizer.gradient_accumulation_steps
        )
        if self._gradient_accumulation_steps:
            self.gradient_accumulator = optimizer.add_variable_from_reference(
                reference_variable=variable, name="gradient_accumulation"
            )

    def update_step(self, gradient, variable):
        if self._gradient_accumulation_steps:
            # Utilize a stateless manner for JAX compatibility
            steps = self._gradient_accumulation_steps
            is_update_step = (self._iterations + 1) % steps == 0
            # Keep track of the maximum scale factor encountered.
            new_g_acc = ops.cond(
                is_update_step,
                lambda: ops.zeros(gradient.shape, dtype=gradient.dtype),
                lambda: ops.maximum(gradient, self.gradient_accumulator),
            )
            new_g = ops.cond(
                is_update_step,
                lambda: ops.maximum(gradient, self.gradient_accumulator),
                lambda: gradient,
            )
            new_v = ops.cond(
                is_update_step, lambda: new_g, lambda: variable.value
            )
            variable.assign(new_v)
            self.gradient_accumulator.assign(new_g_acc)
        else:
            # Assign scale "gradient" directly to variable.
            variable.assign(gradient)


Optimizer.__doc__ = base_optimizer.BaseOptimizer.__doc__
base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args
