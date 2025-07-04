from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.Nadam"])
class Nadam(optimizer.Optimizer):
    """Optimizer that implements the Nadam algorithm.

    Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
    Nesterov momentum.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to `1e-7`.
        {{base_optimizer_keyword_args}}

    Reference:

    - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).

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
        name="nadam",
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

        Nadam optimizer has 2 types of variables: momentums and velocities.

        Args:
            var_list: list of model variables to build Nadam variables on.
        """
        if self.built:
            return
        if var_list:
            dtype = var_list[0].dtype
        else:
            dtype = backend.floatx()
        super().build(var_list)
        self._momentums, self._velocities = self.add_optimizer_variables(
            var_list, ["momentum", "velocity"]
        )
        self._u_product = backend.Variable(1.0, dtype=dtype)

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        dtype = self._u_product.dtype
        self.assign(
            self._u_product,
            self._u_product
            * self.beta_1
            * (
                1.0
                - 0.5 * ops.power(0.96, ops.cast(self.iterations + 1, dtype))
            ),
        )
        super()._backend_update_step(grads, trainable_variables, learning_rate)

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        var_dtype = variable.dtype
        lr = ops.cast(learning_rate, var_dtype)
        gradient = ops.cast(gradient, var_dtype)

        local_step = ops.cast(self.iterations + 1, var_dtype)
        next_step = ops.cast(self.iterations + 2, var_dtype)
        decay = ops.cast(0.96, var_dtype)
        beta_1 = ops.cast(self.beta_1, var_dtype)
        beta_2 = ops.cast(self.beta_2, var_dtype)
        u_t = beta_1 * (1.0 - 0.5 * (ops.power(decay, local_step)))
        u_t_1 = beta_1 * (1.0 - 0.5 * (ops.power(decay, next_step)))
        u_product_t = ops.cast(self._u_product, var_dtype)

        u_product_t_1 = u_product_t * u_t_1
        beta_2_power = ops.power(beta_2, local_step)

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), (1 - beta_1))
        )
        self.assign_add(
            v, ops.multiply(ops.subtract(ops.square(gradient), v), (1 - beta_2))
        )
        m_hat = ops.add(
            ops.divide(ops.multiply(u_t_1, m), 1 - u_product_t_1),
            ops.divide(ops.multiply(1 - u_t, gradient), 1 - u_product_t),
        )
        v_hat = ops.divide(v, (1 - beta_2_power))

        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(m_hat, lr), ops.add(ops.sqrt(v_hat), self.epsilon)
            ),
        )

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


Nadam.__doc__ = Nadam.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
