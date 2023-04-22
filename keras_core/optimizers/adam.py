from keras_core import operations as ops
from keras_core.optimizers import optimizer


class Adam(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
      learning_rate: A `Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to `0.001`.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimates. Defaults to `0.9`.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 2nd moment estimates. Defaults to
        `0.999`.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        `1e-7`.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        name="Adam",
        **kwargs
    ):
        super().__init__(
            learning_rate=learning_rate
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            **kwargs
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
          var_list: list of model variables to build Adam variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = ops.cast(self.learning_rate, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(ops.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, variable.dtype), local_step)

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        m.assign(m + (gradient - m) * (1 - self.beta_1))
        v.assign(v + (ops.square(gradient) - v) * (1 - self.beta_2))
        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            v_hat.assign(ops.maximum(v_hat, v))
            v = v_hat
        variable.assign(m - (m * alpha) / (ops.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


Adam.__doc__ = Adam.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)