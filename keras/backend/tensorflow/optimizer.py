import tensorflow as tf

from keras import backend
from keras.optimizers import base_optimizer


class TFOptimizer(base_optimizer.BaseOptimizer):
    """A class for Tensorflow specific optimizer logic.

    The major behavior change for this class is for tf.distribute.

    It will override methods from base Keras core Optimizer,
    which provide distribute specific functionality, e.g. variable
    creation, loss reduction, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distribution_strategy = tf.distribute.get_strategy()

    def add_variable_from_reference(self, reference_variable, name=None):
        if isinstance(reference_variable, backend.Variable):
            colocate_var = reference_variable.value
        else:
            colocate_var = reference_variable

        with self._distribution_strategy.extended.colocate_vars_with(
            colocate_var
        ):
            return super().add_variable_from_reference(
                reference_variable, name=name
            )

    def _var_key(self, variable):
        if isinstance(variable, backend.Variable):
            variable = variable.value  # Convert to tf.Variable
        if hasattr(variable, "_distributed_container"):
            variable = variable._distributed_container()
        elif (
            isinstance(variable, tf.__internal__.CompositeTensor)
            and hasattr(variable, "handle")
            and hasattr(variable.handle, "_distributed_container")
        ):
            # For ResourceVariables, the _distributed_container attribute
            # is added to their handle tensors.
            variable = variable.handle._distributed_container()
        return variable._unique_id

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return

        def distributed_apply_weight_decay(distribution, variables, **kwargs):
            def weight_decay_fn(variable):
                if self._use_weight_decay(variable):
                    lr = tf.cast(self.learning_rate, variable.dtype)
                    wd = tf.cast(self.weight_decay, variable.dtype)
                    variable.assign(variable - variable * wd * lr)

            for variable in variables:
                distribution.extended.update(
                    variable, weight_decay_fn, group=False
                )

        tf.__internal__.distribute.interim.maybe_merge_call(
            distributed_apply_weight_decay,
            self._distribution_strategy,
            variables,
        )

    def _internal_apply_gradients(self, grads_and_vars):
        tf.__internal__.distribute.interim.maybe_merge_call(
            self._distributed_apply_gradients_fn,
            self._distribution_strategy,
            grads_and_vars,
        )

    def _distributed_apply_gradients_fn(
        self, distribution, grads_and_vars, **kwargs
    ):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad):
            learning_rate = self._get_current_learning_rate()
            grad = tf.convert_to_tensor(grad)
            return self.update_step(grad, var, learning_rate)

        for grad, var in grads_and_vars:
            distribution.extended.update(
                var, apply_grad_to_update_var, args=(grad,), group=False
            )

        if self.use_ema:
            _, var_list = zip(*grads_and_vars)
            self._update_model_variables_moving_average(var_list)
            if self.ema_overwrite_frequency:
                # Only when self.ema_overwrite_frequency is not None, we
                # overwrite the model variables.
                should_overwrite_model_vars = (
                    self.iterations + 1
                ) % self.ema_overwrite_frequency == 0
                tf.cond(
                    tf.cast(should_overwrite_model_vars, tf.bool),
                    true_fn=lambda: self._overwrite_model_variables_with_average_value(  # noqa: E501
                        var_list
                    ),
                    false_fn=lambda: None,
                )
        self.iterations.assign(self.iterations + 1)

    def _overwrite_model_variables_with_average_value(self, var_list):
        """Overwrite model variables with their moving average values.

        This function overwrites variables on each device.
        Args:
          var_list: list of model variables.
        """
        strategy = self._distribution_strategy
        # Override model variable by the stored average value on all devices.
        for var, average_var in zip(
            var_list, self._model_variables_moving_average
        ):
            strategy.extended.update(
                var, lambda a, b: a.assign(b), args=(average_var,)
            )
