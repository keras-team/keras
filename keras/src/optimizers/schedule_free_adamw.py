from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.ScheduleFreeAdamW"])
class ScheduleFreeAdamW(optimizer.Optimizer):
    """Optimizer that implements the Schedule-Free AdamW algorithm.

    Schedule-Free learning is a method that avoids the need for a learning rate
    schedule by maintaining a combination of interpolation and averaging.
    This approach eliminates the requirement to specify stopping time in advance
    and typically matches or outperforms cosine and linear decay schedules.

    The optimizer maintains two sets of variables internally:
    - `momentum`: The sequence where gradient updates are applied
    - `x`: The averaged sequence used for evaluation

    During training, the model parameters are set to an interpolation between
    `momentum` and `x`.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.0025`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates and controls
            the interpolation between `momentum` and `x`. Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates.
            Defaults to `0.999`.
        epsilon: A small constant for numerical stability.
            Defaults to `1e-8`.
        warmup_steps: Number of warmup steps for learning rate warmup.
            During warmup, the learning rate linearly increases from 0 to the
            specified learning rate. Defaults to `0`.
        {{base_optimizer_keyword_args}}

    References:

    - [Defazio et al., 2024](https://arxiv.org/abs/2405.15682)
    - [Schedule-Free repository](
        https://github.com/facebookresearch/schedule_free)

    Example:

    >>> optimizer = keras.optimizers.ScheduleFreeAdamW(learning_rate=0.0025)
    >>> model.compile(optimizer=optimizer, loss="mse")
    >>> model.fit(x_train, y_train)

    """

    def __init__(
        self,
        learning_rate=0.0025,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        warmup_steps=0,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="schedule_free_adamw",
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
        self.warmup_steps = warmup_steps

    def build(self, var_list):
        """Initialize optimizer variables.

        ScheduleFreeAdamW optimizer has the following variables:
        - `momentum`: Auxiliary variable where gradient updates are applied
        - `velocity`: Exponential moving average of squared gradients (Adam)

        Args:
            var_list: list of model variables to build optimizer variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._momentums, self._velocities = self.add_optimizer_variables(
            var_list, ["momentum", "velocity"]
        )

        # Initialize momentum to match the initial parameter values
        for momentum, var in zip(self._momentums, var_list):
            if momentum is not None:
                self.assign(momentum, ops.copy(var))

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)

        beta_1 = ops.cast(self.beta_1, variable.dtype)
        beta_2 = ops.cast(self.beta_2, variable.dtype)
        epsilon = ops.cast(self.epsilon, variable.dtype)

        # Apply warmup
        if self.warmup_steps > 0:
            warmup_steps = ops.cast(self.warmup_steps, variable.dtype)
            warmup_factor = ops.minimum(local_step / warmup_steps, 1.0)
            lr = lr * warmup_factor

        var_index = self._get_variable_index(variable)
        momentum = self._momentums[var_index]
        velocity = self._velocities[var_index]

        # Store momentum_old before any updates
        momentum_old = momentum.value

        # Bias correction for Adam's second moment
        bias_correction_2 = 1 - ops.power(beta_2, local_step)

        # Update velocity (second moment estimate)
        # velocity = beta_2 * velocity + (1 - beta_2) * gradient^2
        self.assign_add(
            velocity,
            ops.multiply(
                ops.subtract(ops.square(gradient), velocity), 1 - beta_2
            ),
        )

        # Compute the denominator (RMSprop-style with bias correction)
        denom = ops.add(ops.sqrt(velocity / bias_correction_2), epsilon)

        # Update momentum: momentum = momentum - lr * gradient / denom
        grad_scaled = ops.divide(ops.multiply(lr, gradient), denom)
        self.assign_sub(momentum, grad_scaled)

        # Compute weight for averaging: weight = 1 / step
        weight = 1.0 / local_step

        # Recover x_old from y_old and momentum_old
        # x_old = (y_old - (1 - beta_1) * momentum_old) / beta_1
        y_old = variable
        x_old = ops.divide(
            ops.subtract(y_old, ops.multiply(1 - beta_1, momentum_old)),
            beta_1,
        )

        # x_new = lerp(x_old, momentum, weight)
        # x_new = (1 - weight) * x_old + weight * momentum
        x_new = ops.add(
            ops.multiply(1 - weight, x_old), ops.multiply(weight, momentum)
        )

        # y_new = lerp(momentum, x_new, beta_1)
        # y_new = (1 - beta_1) * momentum + beta_1 * x_new
        y_new = ops.add(
            ops.multiply(1 - beta_1, momentum), ops.multiply(beta_1, x_new)
        )

        self.assign(variable, y_new)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "warmup_steps": self.warmup_steps,
            }
        )
        return config


ScheduleFreeAdamW.__doc__ = ScheduleFreeAdamW.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
