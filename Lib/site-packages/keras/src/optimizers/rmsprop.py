from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.RMSprop"])
class RMSprop(optimizer.Optimizer):
    """Optimizer that implements the RMSprop algorithm.

    The gist of RMSprop is to:

    - Maintain a moving (discounted) average of the square of gradients
    - Divide the gradient by the root of this average

    This implementation of RMSprop uses plain momentum, not Nesterov momentum.

    The centered version additionally maintains a moving average of the
    gradients, and uses that average to estimate the variance.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        rho: float, defaults to 0.9. Discounting factor for the old gradients.
        momentum: float, defaults to 0.0. If not 0.0., the optimizer tracks the
            momentum value, with a decay rate equals to `1 - momentum`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
            to 1e-7.
        centered: Boolean. If `True`, gradients are normalized by the estimated
            variance of the gradient; if False, by the uncentered second moment.
            Setting this to `True` may help with training, but is slightly more
            expensive in terms of computation and memory. Defaults to `False`.
        {{base_optimizer_keyword_args}}

    Example:

    >>> opt = keras.optimizers.RMSprop(learning_rate=0.1)
    >>> var1 = keras.backend.Variable(10.0)
    >>> loss = lambda: (var1 ** 2) / 2.0  # d(loss) / d(var1) = var1
    >>> opt.minimize(loss, [var1])
    >>> var1
    9.683772

    Reference:

    - [Hinton, 2012](
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        centered=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="rmsprop",
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
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

    def build(self, var_list):
        if self.built:
            return

        super().build(var_list)

        self._velocities = []
        for var in var_list:
            self._velocities.append(
                self.add_variable_from_reference(var, "velocity")
            )

        self._momentums = []
        if self.momentum > 0:
            for var in var_list:
                self._momentums.append(
                    self.add_variable_from_reference(var, "momentum")
                )

        self._average_gradients = []
        if self.centered:
            for var in var_list:
                self._average_gradients.append(
                    self.add_variable_from_reference(var, "average_gradient")
                )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        velocity = self._velocities[self._get_variable_index(variable)]
        momentum = None
        if self.momentum > 0:
            momentum = self._momentums[self._get_variable_index(variable)]
        average_grad = None
        if self.centered:
            average_grad = self._average_gradients[
                self._get_variable_index(variable)
            ]

        rho = self.rho

        self.assign(
            velocity,
            ops.add(
                ops.multiply(rho, velocity),
                ops.multiply(1 - rho, ops.square(gradient)),
            ),
        )
        if self.centered:
            self.assign(
                average_grad,
                ops.add(
                    ops.multiply(rho, average_grad),
                    ops.multiply(1 - rho, gradient),
                ),
            )
            denominator = velocity - ops.square(average_grad) + self.epsilon
        else:
            denominator = ops.add(velocity, self.epsilon)
        increment = ops.divide(
            ops.multiply(lr, gradient), ops.sqrt(denominator)
        )
        if self.momentum > 0:
            self.assign(
                momentum,
                ops.add(ops.multiply(self.momentum, momentum), increment),
            )
            self.assign_sub(variable, momentum)
        else:
            self.assign_sub(variable, increment)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
            }
        )
        return config


RMSprop.__doc__ = RMSprop.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
