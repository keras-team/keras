import tensorflow as tf

from keras import backend
from keras.backend.common import KerasVariable
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

    def add_variable_from_reference(
        self, reference_variable, name=None, initializer="zeros"
    ):
        if isinstance(reference_variable, backend.Variable):
            colocate_var = reference_variable.value
        else:
            colocate_var = reference_variable

        with self._distribution_strategy.extended.colocate_vars_with(
            colocate_var
        ):
            return super().add_variable_from_reference(
                reference_variable, name=name, initializer=initializer
            )

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        # This is mainly due to the interaction with tf.distribute.Strategy,
        # which requires tf.Variable as the inputs for most of its APIs.
        raise ValueError(
            "stateless_apply is not supported with the TensorFlow backend "
            "(as it is incompatible with tf.distribute)."
        )

    def assign_add(self, variable, value):
        if isinstance(variable, KerasVariable):
            variable = variable.value
        value = tf.cast(value, variable.dtype)
        if isinstance(value, tf.IndexedSlices):
            variable.scatter_add(value)
        else:
            variable.assign_add(value)

    def assign_sub(self, variable, value):
        if isinstance(variable, KerasVariable):
            variable = variable.value
        value = tf.cast(value, variable.dtype)
        if isinstance(value, tf.IndexedSlices):
            variable.scatter_sub(value)
        else:
            variable.assign_sub(value)

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
                    variable.assign_sub(variable * wd * lr)

            for variable in variables:
                if isinstance(variable, backend.Variable):
                    variable = variable.value  # Convert to tf.Variable
                distribution.extended.update(
                    variable, weight_decay_fn, group=False
                )

        tf.__internal__.distribute.interim.maybe_merge_call(
            distributed_apply_weight_decay,
            self._distribution_strategy,
            variables,
        )

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        trainable_variables = [
            v.value if isinstance(v, backend.Variable) else v
            for v in trainable_variables
        ]
        tf.__internal__.distribute.interim.maybe_merge_call(
            self._distributed_tf_update_step,
            self._distribution_strategy,
            list(zip(grads, trainable_variables)),
            learning_rate,
        )

    def _distributed_tf_update_step(
        self, distribution, grads_and_vars, learning_rate
    ):
        def apply_grad_to_update_var(var, grad):
            return self.update_step(grad, var, learning_rate)

        for grad, var in grads_and_vars:
            distribution.extended.update(
                var, apply_grad_to_update_var, args=(grad,), group=False
            )

    def _overwrite_model_variables_with_average_value(
        self, trainable_variables
    ):
        """Overwrite model variables with their moving average values.

        This function overwrites variables on each device.
        Args:
          var_list: list of model variables.
        """
        trainable_variables = [
            v.value if isinstance(v, backend.Variable) else v
            for v in trainable_variables
        ]
        # Override model variable by the stored average value on all devices.
        for var, average_var in zip(
            trainable_variables, self._model_variables_moving_average
        ):
            self._distribution_strategy.extended.update(
                var, lambda a, b: a.assign(b), args=(average_var,)
            )

    def _backend_increment_gradient_accumulators(self, grads):
        def update_accumulator(var, grad):
            var.assign(var + grad)

        accumulators = [v.value for v in self._accumulated_gradients]

        def _distributed_tf_increment_grad_acc(
            distribution, grads, accumulators
        ):
            for grad, var in zip(grads, accumulators):
                distribution.extended.update(
                    var, update_accumulator, args=(grad,), group=False
                )

        tf.__internal__.distribute.interim.maybe_merge_call(
            _distributed_tf_increment_grad_acc,
            self._distribution_strategy,
            grads,
            accumulators,
        )

    def _clip_by_norm(self, values, axes=None):
        # We need to use TF-specific OP to support the case,
        # when `values` are `tf.IndexedSlices`.
        return tf.clip_by_norm(values, self.clipnorm, axes)
