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
    - `z`: The sequence where gradient updates are applied
    - `x`: The averaged sequence used for evaluation

    During training, the model parameters are set to an interpolation between
    `z` and `x`. During evaluation, parameters should be set to `x` for optimal
    performance.

    **Important**: You should call `optimizer.swap_to_train()` before training
    and `optimizer.swap_to_eval()` before evaluation/inference to switch between
    parameter states. The methods `train()` and `eval()` are aliases.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.0025`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates and controls
            the interpolation between `z` and `x`. Defaults to `0.9`.
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
    >>> # Training loop
    >>> optimizer.swap_to_train()
    >>> model.fit(x_train, y_train)
    >>> # Evaluation
    >>> optimizer.swap_to_eval()
    >>> model.evaluate(x_test, y_test)

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
        self._is_training_mode = True

    def build(self, var_list):
        """Initialize optimizer variables.

        ScheduleFreeAdamW optimizer has the following variables:
        - `z`: Auxiliary variable where gradient updates are applied
        - `v`: Exponential moving average of squared gradients (Adam)

        Args:
            var_list: list of model variables to build optimizer variables on.
        """
        if self.built:
            return
        super().build(var_list)
        # z: auxiliary variable (gradient steps applied here)
        # v: second moment estimate (velocity)
        # The model variables act as 'y' during training and 'x' during eval
        self._z = []
        self._v = []

        for var in var_list:
            if not self._overwrite_variable_with_gradient(var):
                self._z.append(
                    self.add_variable_from_reference(
                        reference_variable=var,
                        name="z",
                        initializer="zeros",
                    )
                )
                self._v.append(
                    self.add_variable_from_reference(
                        reference_variable=var,
                        name="velocity",
                        initializer="zeros",
                    )
                )
            else:
                self._z.append(None)
                self._v.append(None)

        # Initialize z to match the initial parameter values
        # We use ops.copy to ensure no aliasing issues with JAX
        for i, var in enumerate(var_list):
            z = self._z[i]
            if z is not None:
                self.assign(z, ops.copy(var))

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
        z = self._z[var_index]
        v = self._v[var_index]

        # Store z_old before any updates (use copy to avoid aliasing in JAX)
        z_old = ops.copy(z)

        # Bias correction for Adam's second moment
        bias_correction_2 = 1 - ops.power(beta_2, local_step)

        # Update velocity (second moment estimate)
        # v = beta_2 * v + (1 - beta_2) * gradient^2
        self.assign_add(
            v,
            ops.multiply(ops.subtract(ops.square(gradient), v), 1 - beta_2),
        )

        # Compute the denominator (RMSprop-style with bias correction)
        denom = ops.add(ops.sqrt(v / bias_correction_2), epsilon)

        # Update z: z = z - lr * gradient / denom
        grad_scaled = ops.divide(ops.multiply(lr, gradient), denom)
        self.assign_sub(z, grad_scaled)

        # Compute weight for averaging: weight = 1 / step
        weight = 1.0 / local_step

        # Recover x_old from y_old and z_old
        # x_old = (y_old - (1 - beta_1) * z_old) / beta_1
        y_old = variable
        x_old = ops.divide(
            ops.subtract(y_old, ops.multiply(1 - beta_1, z_old)), beta_1
        )

        # x_new = lerp(x_old, z, weight) = (1 - weight) * x_old + weight * z
        x_new = ops.add(
            ops.multiply(1 - weight, x_old), ops.multiply(weight, z)
        )

        # y_new = lerp(z, x_new, beta_1) = (1 - beta_1) * z + beta_1 * x_new
        y_new = ops.add(
            ops.multiply(1 - beta_1, z), ops.multiply(beta_1, x_new)
        )

        self.assign(variable, y_new)

    def swap_to_train(self):
        """Switch parameters to training mode (y = interpolation of z and x).

        Call this before training. During training, model parameters are set
        to y = (1 - beta_1) * z + beta_1 * x, which is the point where
        gradients are computed.
        """
        if self._is_training_mode:
            return

        if not self.built:
            return

        for i, var in enumerate(self._trainable_variables):
            z = self._z[i]
            if z is None:
                continue

            beta_1 = ops.cast(self.beta_1, var.dtype)
            # Current variable holds x (eval mode)
            x = var
            # Compute y = (1 - beta_1) * z + beta_1 * x
            y = ops.add(ops.multiply(1 - beta_1, z), ops.multiply(beta_1, x))
            self.assign(var, y)

        self._is_training_mode = True

    def swap_to_eval(self):
        """Switch parameters to evaluation mode (x = averaged sequence).

        Call this before evaluation/inference. During evaluation, model
        parameters are set to x, which is the Polyak-averaged sequence that
        typically provides better generalization.
        """
        if not self._is_training_mode:
            return

        if not self.built:
            return

        for i, var in enumerate(self._trainable_variables):
            z = self._z[i]
            if z is None:
                continue

            beta_1 = ops.cast(self.beta_1, var.dtype)
            # Current variable holds y (training mode)
            y = var
            # Recover x from y: x = (y - (1 - beta_1) * z) / beta_1
            x = ops.divide(
                ops.subtract(y, ops.multiply(1 - beta_1, z)),
                beta_1,
            )
            self.assign(var, x)

        self._is_training_mode = False

    # Aliases for train/eval
    def train(self):
        """Alias for swap_to_train()."""
        self.swap_to_train()

    def eval(self):
        """Alias for swap_to_eval()."""
        self.swap_to_eval()

    @property
    def is_training_mode(self):
        """Returns True if optimizer is in training mode."""
        return self._is_training_mode

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
