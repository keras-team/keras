from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.Adagrad"])
class Adagrad(optimizer.Optimizer):
    """Optimizer that implements the Adagrad algorithm.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`. Note that `Adagrad`
            tends to benefit from higher initial learning rate values compared
            to other optimizers. To match the exact form in the original paper,
            use `1.0`.
        initial_accumulator_value: Floating point value. Starting value for the
            accumulators (per-parameter momentum values). Must be non-negative.
        epsilon: Small floating point value for maintaining numerical stability.
        {{base_optimizer_keyword_args}}

    Reference:

    - [Duchi et al., 2011](
        http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
    """

    def __init__(
        self,
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adagrad",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs,
        )
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        initializer = initializers.Constant(self.initial_accumulator_value)
        self._accumulators = self.add_optimizer_variables(
            var_list, "accumulator", initializer=initializer
        )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        accumulator = self._accumulators[self._get_variable_index(variable)]

        self.assign_add(accumulator, ops.square(gradient))
        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(lr, gradient),
                ops.sqrt(ops.add(accumulator, self.epsilon)),
            ),
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "initial_accumulator_value": self.initial_accumulator_value,
                "epsilon": self.epsilon,
            }
        )
        return config


Adagrad.__doc__ = Adagrad.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
