from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export("keras.optimizers.Lamb")
class Lamb(optimizer.Optimizer):
    """Optimizer that implements the Lamb algorithm.

    Lamb is a stochastic gradient descent method that
    uses layer-wise adaptive moments to adjusts the
    learning rate for each parameter based on the ratio of the
    norm of the weight to the norm of the gradient
    This helps to stabilize the training process and improves convergence
    especially for large batch sizes.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates. Defaults to
            `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability.
            Defaults to `1e-7`.
        {{base_optimizer_keyword_args}}

    References:
        - [Yang et al.](https://arxiv.org/pdf/1904.00962)
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
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
        name="lamb",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        Lamb optimizer has 2 types of variables: momentums and velocities

        Args:
            var_list: list of model variables to build Lamb variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._momentums, self._velocities = self.add_optimizer_variables(
            var_list, ["momentum", "velocity"]
        )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)

        beta_1_power = ops.power(
            ops.cast(self.beta_1, variable.dtype), local_step
        )
        beta_2_power = ops.power(
            ops.cast(self.beta_2, variable.dtype), local_step
        )

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), 1 - self.beta_1)
        )

        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - self.beta_2
            ),
        )

        m_t_hat = ops.divide(m, (1.0 - beta_1_power))
        v_sqrt = ops.add(
            ops.sqrt(ops.divide(v, (1.0 - beta_2_power))), self.epsilon
        )

        update = ops.divide(m_t_hat, v_sqrt)
        w_norm = ops.sqrt(ops.sum(ops.power(variable, 2)))
        g_norm = ops.sqrt(ops.sum(ops.power(update, 2)))

        # ratio = w_norm / g_norm if w_norm > 0 and g_norm > 0 else 1
        ratio = ops.where(
            ops.greater(w_norm, 0),
            ops.where(ops.greater(g_norm, 0), (w_norm / g_norm), 1.0),
            1.0,
        )

        self.assign_sub(variable, ratio * lr * update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
            }
        )
        return config


Lamb.__doc__ = Lamb.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
