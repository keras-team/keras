from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.Lion"])
class Lion(optimizer.Optimizer):
    """Optimizer that implements the Lion algorithm.

    The Lion optimizer is a stochastic-gradient-descent method that uses the
    sign operator to control the magnitude of the update, unlike other adaptive
    optimizers such as Adam that rely on second-order moments. This make
    Lion more memory-efficient as it only keeps track of the momentum. According
    to the authors (see reference), its performance gain over Adam grows with
    the batch size. Because the update of Lion is produced through the sign
    operation, resulting in a larger norm, a suitable learning rate for Lion is
    typically 3-10x smaller than that for AdamW. The weight decay for Lion
    should be in turn 3-10x larger than that for AdamW to maintain a
    similar strength (lr * wd).

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            rate to combine the current gradient and the 1st moment estimate.
            Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimate. Defaults to
            `0.99`.
        {{base_optimizer_keyword_args}}

    References:

    - [Chen et al., 2023](http://arxiv.org/abs/2302.06675)
    - [Authors' implementation](
        http://github.com/google/automl/tree/master/lion)

    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.99,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="lion",
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
        if beta_1 <= 0 or beta_1 > 1:
            raise ValueError(
                "Argument `beta_1` must be in the [0, 1] range. Otherwise, the "
                f"optimizer degenerates to SignSGD. Received: beta_1={beta_1}."
            )

    def build(self, var_list):
        """Initialize optimizer variables.

        Lion optimizer has one variable `momentums`.

        Args:
            var_list: list of model variables to build Lion variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        beta_1 = ops.cast(self.beta_1, variable.dtype)
        beta_2 = ops.cast(self.beta_2, variable.dtype)
        m = self._momentums[self._get_variable_index(variable)]

        self.assign_sub(
            variable,
            ops.multiply(
                lr,
                ops.sign(
                    ops.add(
                        ops.multiply(m, beta_1),
                        ops.multiply(gradient, (1.0 - beta_1)),
                    )
                ),
            ),
        )
        self.assign(
            m,
            ops.add(
                ops.multiply(m, beta_2), ops.multiply(gradient, (1.0 - beta_2))
            ),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
            }
        )
        return config


Lion.__doc__ = Lion.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
