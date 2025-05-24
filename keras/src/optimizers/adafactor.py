from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.Adafactor"])
class Adafactor(optimizer.Optimizer):
    """Optimizer that implements the Adafactor algorithm.

    Adafactor is commonly used in NLP tasks, and has the advantage
    of taking less memory because it only saves partial information of previous
    gradients.

    The default argument setup is based on the original paper (see reference).
    When gradients are of dimension > 2, Adafactor optimizer will delete the
    last 2 dimensions separately in its accumulator variables.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_2_decay: float, defaults to -0.8. The decay rate of `beta_2`.
        epsilon_1: float, defaults to 1e-30. A small offset to keep denominator
            away from 0.
        epsilon_2: float, defaults to 1e-3. A small offset to avoid learning
            rate becoming too small by time.
        clip_threshold: float, defaults to 1.0. Clipping threshold. This is a
            part of Adafactor algorithm, independent from `clipnorm`,
            `clipvalue`, and `global_clipnorm`.
        relative_step: bool, defaults to `True`. If `learning_rate` is a
            constant and `relative_step=True`, learning rate will be adjusted
            based on current iterations. This is a default learning rate decay
            in Adafactor.
        {{base_optimizer_keyword_args}}

    Reference:

    - [Shazeer, Noam et al., 2018](https://arxiv.org/abs/1804.04235).

    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_2_decay=-0.8,
        epsilon_1=1e-30,
        epsilon_2=1e-3,
        clip_threshold=1.0,
        relative_step=True,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adafactor",
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
        self.beta_2_decay = beta_2_decay
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.clip_threshold = clip_threshold
        self.relative_step = relative_step

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
        self._r = []
        self._c = []
        self._v = []
        for var in var_list:
            if len(var.shape) < 2:
                # Don't factor if variable is of dimension < 2, but we still
                # need to create dummy variables as placeholder.
                self._r.append(
                    backend.Variable(0, name=var.name, trainable=False)
                )
                self._c.append(
                    backend.Variable(0, name=var.name, trainable=False)
                )
            elif self._overwrite_variable_with_gradient(var):
                self._r.append(None)
                self._c.append(None)
            else:
                # Always factor the last 2 dimensions.
                r_shape = var.shape[:-1]
                c_shape = var.shape[:-2] + (var.shape[-1],)
                self._r.append(
                    self.add_variable(
                        shape=r_shape,
                        dtype=var.dtype,
                        name=var.name,
                    )
                )
                self._c.append(
                    self.add_variable(
                        shape=c_shape,
                        dtype=var.dtype,
                        name=var.name,
                    )
                )

            if self._overwrite_variable_with_gradient(var):
                self._v.append(None)
            else:
                self._v.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="velocity"
                    )
                )

    def _rms(self, x):
        return ops.sqrt(ops.mean(ops.square(x)))

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""

        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        epsilon_2 = ops.cast(self.epsilon_2, variable.dtype)
        one = ops.cast(1.0, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        if not callable(self._learning_rate) and self.relative_step:
            lr = ops.minimum(lr, 1 / ops.sqrt(local_step))

        r = self._r[self._get_variable_index(variable)]
        c = self._c[self._get_variable_index(variable)]
        v = self._v[self._get_variable_index(variable)]

        rho_t = ops.minimum(lr, 1 / ops.sqrt(local_step))
        alpha_t = ops.maximum(epsilon_2, self._rms(variable)) * rho_t
        regulated_grad_square = ops.add(ops.square(gradient), self.epsilon_1)
        beta_2_t = 1 - ops.power(local_step, self.beta_2_decay)

        if len(variable.shape) >= 2:
            # `r` deletes the last dimension of gradient, so it is of shape
            # `gradient.shape[:-1]`.
            self.assign(
                r,
                beta_2_t * r
                + (1 - beta_2_t) * ops.mean(regulated_grad_square, axis=-1),
            )
            # `c` deletes the second last dimension of gradient, so it is of
            # shape `gradient.shape[:-2] + gradient.shape[-1]`.
            self.assign(
                c,
                beta_2_t * c
                + (1 - beta_2_t) * ops.mean(regulated_grad_square, axis=-2),
            )
            self.assign(
                v,
                ops.expand_dims(
                    r / ops.mean(r, axis=-1, keepdims=True), axis=-1
                )
                * ops.expand_dims(c, -2),
            )
        else:
            self.assign(
                v, beta_2_t * v + (1 - beta_2_t) * regulated_grad_square
            )

        u_t = ops.divide(gradient, ops.sqrt(v))
        u_t_hat = ops.divide(
            u_t,
            ops.maximum(one, ops.divide(self._rms(u_t), self.clip_threshold)),
        )
        self.assign_sub(variable, ops.multiply(alpha_t, u_t_hat))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "beta_2_decay": self.beta_2_decay,
                "epsilon_1": self.epsilon_1,
                "epsilon_2": self.epsilon_2,
                "clip_threshold": self.clip_threshold,
                "relative_step": self.relative_step,
            }
        )
        return config


Adafactor.__doc__ = Adafactor.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
