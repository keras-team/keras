from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.Adadelta"])
class Adadelta(optimizer.Optimizer):
    """Optimizer that implements the Adadelta algorithm.

    Adadelta optimization is a stochastic gradient descent method that is based
    on adaptive learning rate per dimension to address two drawbacks:

    - The continual decay of learning rates throughout training.
    - The need for a manually selected global learning rate.

    Adadelta is a more robust extension of Adagrad that adapts learning rates
    based on a moving window of gradient updates, instead of accumulating all
    past gradients. This way, Adadelta continues learning even when many updates
    have been done. Compared to Adagrad, in the original version of Adadelta you
    don't have to set an initial learning rate. In this version, the initial
    learning rate can be set, as in most other Keras optimizers.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`. Note that `Adadelta`
            tends to benefit from higher initial learning rate values compared
            to other optimizers. To match the exact form in the original paper,
            use 1.0.
        rho: A floating point value. The decay rate. Defaults to `0.95`.
        epsilon: Small floating point value for maintaining numerical stability.
        {{base_optimizer_keyword_args}}

    Reference:

    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
    """

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.95,
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
        name="adadelta",
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
        self.epsilon = epsilon

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._accumulated_grads, self._accumulated_delta_vars = (
            self.add_optimizer_variables(
                var_list, ["accumulated_grad", "accumulated_delta_var"]
            )
        )

    def update_step(self, grad, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        grad = ops.cast(grad, variable.dtype)

        rho = self.rho
        accumulated_grad = self._accumulated_grads[
            self._get_variable_index(variable)
        ]
        accumulated_delta_var = self._accumulated_delta_vars[
            self._get_variable_index(variable)
        ]

        def rms(x):
            return ops.sqrt(ops.add(x, self.epsilon))

        self.assign(
            accumulated_grad,
            ops.add(
                rho * accumulated_grad, ops.multiply(1 - rho, ops.square(grad))
            ),
        )
        delta_var = ops.negative(
            ops.divide(
                ops.multiply(rms(accumulated_delta_var), grad),
                rms(accumulated_grad),
            )
        )
        self.assign(
            accumulated_delta_var,
            ops.add(
                ops.multiply(rho, accumulated_delta_var),
                ops.multiply(1 - rho, ops.square(delta_var)),
            ),
        )
        self.assign_add(variable, ops.multiply(lr, delta_var))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "rho": self.rho,
                "epsilon": self.epsilon,
            }
        )
        return config


Adadelta.__doc__ = Adadelta.__doc__.replace(
    "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
)
