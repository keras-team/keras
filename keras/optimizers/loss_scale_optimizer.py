from keras import backend
from keras import initializers
from keras import ops
from keras.api_export import keras_export
from keras.optimizers import optimizer
from keras.saving import serialization_lib
from keras.utils import tracking


@keras_export(
    [
        "keras.optimizers.LossScaleOptimizer",
        "keras.mixed_precision.LossScaleOptimizer",
    ]
)
class LossScaleOptimizer(optimizer.Optimizer):
    """An optimizer that dynamically scales the loss to prevent underflow.

    Loss scaling is a technique to prevent numeric underflow in intermediate
    gradients when float16 is used. To prevent underflow, the loss is multiplied
    (or "scaled") by a certain factor called the "loss scale", which causes
    intermediate gradients to be scaled by the loss scale as well. The final
    gradients are divided (or "unscaled") by the loss scale to bring them back
    to their original value.

    `LossScaleOptimizer` wraps another optimizer and applies dynamic loss
    scaling to it. This loss scale is dynamically updated over time as follows:
    - On any train step, if a nonfinite gradient is encountered, the loss scale
      is halved, and the train step is skipped.
    - If `dynamic_growth_steps` have ocurred since the last time the loss scale
      was updated, and no nonfinite gradients have occurred, the loss scale
      is doubled.

    Args:
        inner_optimizer: The `keras.optimizers.Optimizer` instance to wrap.
        initial_scale: Float. The initial loss scale. This scale will be updated
            during training. It is recommended for this to be a very high
            number, because a loss scale that is too high gets lowered far more
            quickly than a loss scale that is too low gets raised.
        dynamic_growth_steps: Int. How often to update the scale upwards. After
            every `dynamic_growth_steps` steps with finite gradients, the
            loss scale is doubled.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        inner_optimizer,
        initial_scale=2.0**15,
        dynamic_growth_steps=2000,
        **kwargs,
    ):
        if not kwargs.pop("dynamic", True):
            raise ValueError(
                "LossScaleOptimizer no longer suports `dynamic=False`. "
                "Instead, simply set `loss_scale_factor` directly on the "
                "`inner_optimizer`."
            )
        super().__init__(learning_rate=0.0, **kwargs)
        self.inner_optimizer = inner_optimizer
        self.initial_scale = initial_scale
        self.dynamic_growth_steps = dynamic_growth_steps

    @tracking.no_automatic_dependency_tracking
    def build(self, var_list):
        self.step_counter = self.add_variable(
            shape=(),
            dtype="int",
            initializer=initializers.Zeros(),
            name="step_counter",
        )
        self.dynamic_scale = self.add_variable(
            shape=(),
            dtype="float32",
            initializer=initializers.Constant(self.initial_scale),
            name="dynamic_scale",
        )
        self.inner_optimizer.build(var_list)
        self.built = True

    @property
    def variables(self):
        return self._variables + self.inner_optimizer.variables

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        if not self.built:
            raise ValueError(
                f"To call `stateless_apply`, {self.__class__.__name__} "
                "must be built (i.e. its variables must have been created). "
                "You can build it via `optimizer.build(trainable_variables)`."
            )

        def handle_finite_grads():
            def upscale():
                mapping = list(zip(self.variables, optimizer_variables))
                with backend.StatelessScope(state_mapping=mapping) as scope:
                    self.step_counter.assign(0)
                    self.dynamic_scale.assign(self.dynamic_scale * 2.0)
                return [scope.get_current_value(v) for v in self._variables]

            def increment():
                mapping = list(zip(self.variables, optimizer_variables))
                with backend.StatelessScope(state_mapping=mapping) as scope:
                    self.step_counter.assign_add(1)
                return [scope.get_current_value(v) for v in self._variables]

            mapping = list(zip(self.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping):
                # Potentially upscale loss and reset counter.
                own_variables = ops.cond(
                    ops.equal(self.step_counter, self.dynamic_growth_steps - 1),
                    upscale,
                    increment,
                )

                # Unscale gradients.
                scale = self.dynamic_scale
                unscaled_grads = [
                    g if g is None else ops.divide(g, scale) for g in grads
                ]
                (
                    new_trainable_variables,
                    new_inner_variables,
                ) = self.inner_optimizer.stateless_apply(
                    self.inner_optimizer.variables,
                    unscaled_grads,
                    trainable_variables,
                )

            new_optimizer_variables = own_variables + new_inner_variables
            return new_trainable_variables, new_optimizer_variables

        def handle_non_finite_grads():
            mapping = list(zip(self.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping) as scope:
                self.step_counter.assign(0)
                self.dynamic_scale.assign(self.dynamic_scale / 2.0)
            new_optimizer_variables = []
            for v in self.variables:
                new_optimizer_variables.append(scope.get_current_value(v))
            return trainable_variables, new_optimizer_variables

        finite = self.check_finite(grads)
        return ops.cond(finite, handle_finite_grads, handle_non_finite_grads)

    def apply(self, grads, trainable_variables=None):
        # Optionally build optimizer.
        if not self.built:
            with backend.name_scope(self.name, caller=self):
                self.build(trainable_variables)
            self.built = True

        def handle_finite_grads():
            scale = self.dynamic_scale
            # Unscale gradients.
            unscaled_grads = [
                g if g is None else ops.divide(g, scale) for g in grads
            ]
            self.inner_optimizer.apply(
                unscaled_grads, trainable_variables=trainable_variables
            )

            def upscale():
                self.step_counter.assign(0)
                self.dynamic_scale.assign(self.dynamic_scale * 2.0)

            def increment():
                self.step_counter.assign_add(1)

            # Potentially upscale loss and reset counter.
            ops.cond(
                ops.equal(self.step_counter, self.dynamic_growth_steps - 1),
                upscale,
                increment,
            )

        def handle_non_finite_grads():
            # If any inf or nan in grads, downscale loss and reset counter.
            self.step_counter.assign(0)
            self.dynamic_scale.assign(self.dynamic_scale / 2.0)

        finite = self.check_finite(grads)
        ops.cond(finite, handle_finite_grads, handle_non_finite_grads)

    def check_finite(self, grads):
        tensor_grads = [g for g in grads if g is not None]
        finite_grads = [ops.all(ops.isfinite(g)) for g in tensor_grads]
        return ops.all(ops.convert_to_tensor(finite_grads))

    @property
    def learning_rate(self):
        return self.inner_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.inner_optimizer.learning_rate = learning_rate

    def scale_loss(self, loss):
        scale = self.dynamic_scale if self.built else self.initial_scale
        return loss * scale

    def finalize_variable_values(self, var_list):
        self.inner_optimizer.finalize_variable_values(var_list)

    def get_config(self):
        config = super().get_config()
        inner_optimizer_config = serialization_lib.serialize_keras_object(
            self.inner_optimizer
        )
        config.update(
            {
                "inner_optimizer": inner_optimizer_config,
                "initial_scale": self.initial_scale,
                "dynamic_growth_steps": self.dynamic_growth_steps,
            }
        )
        del config["learning_rate"]
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inner_optimizer = serialization_lib.deserialize_keras_object(
            config.pop("inner_optimizer"),
            custom_objects=custom_objects,
        )
        return cls(inner_optimizer, **config)


LossScaleOptimizer.__doc__ = LossScaleOptimizer.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
