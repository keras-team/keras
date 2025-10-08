import re

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.Muon"])
class Muon(optimizer.Optimizer):
    """Optimizer that implements the Muon algorithm.

    Note that this optimizer should not be used in the following layers:

    1. Embedding layer
    2. Final output fully connected layer
    3. Any {0,1}-D variables

    These should all be optimized using AdamW.

    The Muon optimizer can use both the Muon update step or the
    AdamW update step based on the following:

    - For any variable that isn't 2D, 3D or 4D, the AdamW step
        will be used. This is not configurable.
    - If the argument `exclude_embeddings` (defaults to `True`) is set
    to `True`, the AdamW step will be used.
    - For any variablewith a name that matches an expression
        listed in the argument `exclude_layers` (a list), the
        AdamW step will be used.
    - Any other variable uses the Muon step.

    Typically, you only need to pass the name of your densely-connected
    output layer to `exclude_layers`, e.g.
    `exclude_layers=["output_dense"]`.

    References:
        - [Original implementation](https://github.com/KellerJordan/Muon)
        - [Liu et al, 2025](https://arxiv.org/abs/2502.16982)

    Args:
        learning_rate: A float,
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        adam_beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use.
            The exponential decay rate for the 1st moment estimates. Defaults to
            `0.9`.
        adam_beta_2: A float value or a constant float tensor, ora callable
            that takes no arguments and returns the actual value to use.
            The exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability. This is
            "epsilon hat" in the Kingma and Ba paper
            (in the formula just before Section 2.1),
            not the epsilon in Algorithm 1 of the paper.
            It be used at Adamw.Defaults to `1e-7`.
        exclude_layers: List of strings, keywords of layer names to exclude.
            All layers with keywords in their path will use adamw.
        exclude_embeddings: Boolean value
            If True, embedding layers will use adamw.
        muon_a: Float, parameter a of the muon algorithm.
            It is recommended to use the default value
        muon_b: Float, parameter b of the muon algorithm.
            It is recommended to use the default value
        muon_c: Float, parameter c of the muon algorithm.
            It is recommended to use the default value
        adam_lr_ratio: Float, the ratio of the learning rate when
                using Adam to the main learning rate.
                it is recommended to set it to 0.1
        momentum: Float, momentum used by internal SGD.
        ns_steps: Integer, number of Newton-Schulz iterations to run.
        nesterov: Boolean, whether to use Nesterov-style momentum
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        learning_rate=0.001,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        epsilon=1e-7,
        weight_decay=0.1,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="muon",
        exclude_layers=None,
        exclude_embeddings=True,
        muon_a=3.4445,
        muon_b=-4.7750,
        muon_c=2.0315,
        adam_lr_ratio=0.1,
        momentum=0.95,
        ns_steps=6,
        nesterov=True,
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
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.muon_a = muon_a
        self.muon_b = muon_b
        self.muon_c = muon_c
        self.adam_lr_ratio = adam_lr_ratio
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.exclude_embeddings = exclude_embeddings
        self.exclude_layers = exclude_layers or []

    def _should_use_adamw(self, variable):
        # To use it with 4D convolutional filters,
        # it works well to just flatten their last 3 dimensions.
        # any {0,1}-D parameters should all be optimized by adam
        if not 1 < len(variable.shape) < 4:
            return True
        if self.exclude_embeddings and "embedding" in variable.path.lower():
            return True
        for keyword in self.exclude_layers:
            if re.search(keyword, variable.path):
                return True
        return False

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
        self.adam_momentums = {}
        self.adam_velocities = {}

        self.muon_momentums = {}
        self.muon_velocities = {}

        for var in var_list:
            if not self._overwrite_variable_with_gradient(var):
                self.adam_momentums[var.path] = (
                    self.add_variable_from_reference(
                        reference_variable=var, name="momentum"
                    )
                )
                if self._should_use_adamw(var):
                    self.adam_velocities[var.path] = (
                        self.add_variable_from_reference(
                            reference_variable=var, name="velocity"
                        )
                    )

    def update_step(self, gradient, variable, learning_rate):
        if self._should_use_adamw(variable):
            # It should be noted that lr is one-tenth when using adamw.
            self._adamw_update_step(
                gradient, variable, learning_rate * self.adam_lr_ratio
            )
        else:
            self._muon_update_step(gradient, variable, learning_rate)

    def _muon_update_step(self, gradient, variable, lr):
        m = self.adam_momentums[variable.path]
        self.assign_add(m, ops.add(gradient, m * (self.momentum - 1)))
        shape = variable.shape
        if self.nesterov:
            g = ops.add(gradient, self.momentum * m)
        else:
            g = m

        self.assign_sub(
            variable,
            lr
            * self.zeropower_via_newtonschulz5(g, self.ns_steps)
            * max(1, shape[0] / shape[1]) ** 0.5,
        )

    def _adamw_update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        adam_beta_1_power = ops.power(
            ops.cast(self.adam_beta_1, variable.dtype), local_step
        )
        adam_beta_2_power = ops.power(
            ops.cast(self.adam_beta_2, variable.dtype), local_step
        )

        m = self.adam_momentums[variable.path]
        v = self.adam_velocities[variable.path]

        alpha = lr * ops.sqrt(1 - adam_beta_2_power) / (1 - adam_beta_1_power)

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), 1 - self.adam_beta_1)
        )
        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - self.adam_beta_2
            ),
        )
        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)
            ),
        )

    def transpose_last_axis(self, X):
        shape = ops.shape(X)
        temp_order = list(range(len(shape)))
        temp_order[-2] = temp_order[-1]
        temp_order[-1] = len(shape) - 2
        X = ops.transpose(X, temp_order)
        return X

    def zeropower_via_newtonschulz5(self, x, steps: int):
        """We apply the Newton-Schulz iteration to compute matrix G.

        We select a quintic iteration that maximizes the slope at zero. This
        approach helps minimize steps, even if the iteration doesn't fully
        converge across the interval. The result isn't exactly UV^T (from the
        SVD of G), but rather an approximation like US'V^T. Despite this
        approximation, model performance remains unaffected compared to using
        the exact UV^T from the SVD.
        """
        shape = ops.shape(x)
        assert len(shape) >= 2

        a, b, c = self.muon_a, self.muon_b, self.muon_c
        if shape[-2] > shape[-1]:
            x = self.transpose_last_axis(x)

        # Ensure spectral norm is at most 1
        x = x / (ops.norm(x, axis=(-2, -1), keepdims=True) + 1e-7)
        # Perform the NS iterations
        for _ in range(steps):
            temp_a = x @ self.transpose_last_axis(x)
            temp_b = b * temp_a + c * temp_a @ temp_a
            x = a * x + temp_b @ x

        if shape[-2] > shape[-1]:
            x = self.transpose_last_axis(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "adam_beta_1": self.adam_beta_1,
                "adam_beta_2": self.adam_beta_2,
                "epsilon": self.epsilon,
                "exclude_layers": self.exclude_layers,
                "muon_a": self.muon_a,
                "muon_b": self.muon_b,
                "muon_c": self.muon_c,
                "adam_lr_ratio": self.adam_lr_ratio,
                "momentum": self.momentum,
                "ns_steps": self.ns_steps,
                "nesterov": self.nesterov,
                "exclude_embeddings": self.exclude_embeddings,
            }
        )
        return config
