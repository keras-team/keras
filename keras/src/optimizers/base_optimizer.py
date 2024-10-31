import re
import warnings

from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.saving.keras_saveable import KerasSaveable
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name


class BaseOptimizer(KerasSaveable):
    """Abstract optimizer base class.

    If you intend to create your own optimization algorithm, please inherit from
    this class and override the following methods:

    - `build`: Create your optimizer-related variables, such as momentum
        variables in the SGD optimizer.
    - `update_step`: Implement your optimizer's variable updating logic.
    - `get_config`: serialization of the optimizer.

    Example:

    ```python
    class SGD(Optimizer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.momentum = 0.9

        def build(self, variables):
            super().build(variables)
            self.momentums = []
            for variable in variables:
                self.momentums.append(
                    self.add_variable_from_reference(
                        reference_variable=variable, name="momentum"
                    )
                )

        def update_step(self, gradient, variable, learning_rate):
            learning_rate = ops.cast(learning_rate, variable.dtype)
            gradient = ops.cast(gradient, variable.dtype)
            m = self.momentums[self._get_variable_index(variable)]
            self.assign(
                m,
                ops.subtract(
                    ops.multiply(m, ops.cast(self.momentum, variable.dtype)),
                    ops.multiply(gradient, learning_rate),
                ),
            )
            self.assign_add(variable, m)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                }
            )
            return config
    ```
    """

    def __init__(
        self,
        learning_rate,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs,
    ):
        self._lock = False

        if kwargs.pop("decay", None) is not None:
            warnings.warn(
                "Argument `decay` is no longer supported and will be ignored."
            )
        if kwargs:
            raise ValueError(f"Argument(s) not recognized: {kwargs}")

        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm
        self.clipvalue = clipvalue
        self.use_ema = use_ema
        self.loss_scale_factor = loss_scale_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if gradient_accumulation_steps:
            if not gradient_accumulation_steps >= 2:
                raise ValueError(
                    "`gradient_accumulation_steps` must be an integer >= 2. "
                    "Received: gradient_accumulation_steps="
                    f"{gradient_accumulation_steps}"
                )

        if use_ema:
            # Verify the arguments related to EMA.
            if ema_momentum > 1 or ema_momentum < 0:
                raise ValueError(
                    "`ema_momentum` must be in the range [0, 1]. "
                    f"Received: ema_momentum={ema_momentum}"
                )
            if ema_overwrite_frequency and (
                not isinstance(ema_overwrite_frequency, int)
                or ema_overwrite_frequency < 1
            ):
                raise ValueError(
                    "`ema_overwrite_frequency` must be an integer >= 1 or "
                    "None. Received: ema_overwrite_frequency="
                    f"{ema_overwrite_frequency}"
                )
        self.ema_momentum = ema_momentum
        self.ema_overwrite_frequency = ema_overwrite_frequency

        clip_args_sum = sum(
            a is not None for a in [clipnorm, clipvalue, global_clipnorm]
        )
        if clip_args_sum > 1:
            raise ValueError(
                "Only one of `clipnorm`, `clipvalue` and `global_clipnorm` can "
                f"be set. Received: clipnorm={clipnorm}, "
                f"clipvalue={clipvalue}, global_clipnorm={global_clipnorm}"
            )
        self.built = False

        # Set up variable tracking.
        self._variables = []
        self._trainable_variables = []
        self._tracker = tracking.Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    self._variables,
                ),
            }
        )
        self._trainable_variables_indices = {}

        # Create iteration variable
        # Note: dtype="int" will resolve to int32 in JAX
        # (since int64 is disallowed in JAX) and to int64 in TF.
        with backend.name_scope(self.name, caller=self):
            iterations = backend.Variable(
                0,
                name="iteration",
                dtype="int",
                trainable=False,
                aggregation="only_first_replica",
            )
        self._track_variable(iterations)
        self._iterations = iterations

        # Create learning rate (schedule or variable)
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if not isinstance(learning_rate, float):
                raise ValueError(
                    "Argument `learning_rate` should be float, or an instance "
                    "of LearningRateSchedule, or a callable "
                    "(that takes in the current iteration value "
                    "and returns the corresponding learning rate value). "
                    f"Received instead: learning_rate={learning_rate}"
                )
            with backend.name_scope(self.name, caller=self):
                learning_rate = backend.Variable(
                    learning_rate,
                    name="learning_rate",
                    dtype=backend.floatx(),
                    trainable=False,
                    aggregation="only_first_replica",
                )
            self._track_variable(learning_rate)
            self._learning_rate = learning_rate

    @property
    def iterations(self):
        if self.gradient_accumulation_steps:
            return ops.floor_divide(
                self._iterations, self.gradient_accumulation_steps
            )

        return self._iterations

    def _track_variable(self, variable):
        self._tracker.add_to_store("variables", variable)

    @tracking.no_automatic_dependency_tracking
    def build(self, variables):
        if self.use_ema:
            self._model_variables_moving_average = []
        if self.gradient_accumulation_steps:
            self._accumulated_gradients = []
        for i, variable in enumerate(variables):
            self._trainable_variables_indices[self._var_key(variable)] = i
            if self.use_ema:
                self._model_variables_moving_average.append(
                    self.add_variable_from_reference(
                        variable,
                        name="average",
                    )
                )
            if self.gradient_accumulation_steps:
                self._accumulated_gradients.append(
                    self.add_variable_from_reference(
                        variable,
                        name="gradient_accumulator",
                    )
                )
        self._trainable_variables = variables[:]
        self.built = True

    def _var_key(self, variable):
        # Helper function to get a stable ID and the variable instance mapping.
        return id(variable)

    @property
    def variables(self):
        return self._variables[:]

    def _get_variable_index(self, variable):
        return self._trainable_variables_indices[self._var_key(variable)]

    def add_variable(
        self,
        shape,
        initializer="zeros",
        dtype=None,
        aggregation="mean",
        name=None,
    ):
        self._check_super_called()
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=False,
                aggregation=aggregation,
                name=name,
            )
        self._track_variable(variable)
        return variable

    def add_variable_from_reference(
        self, reference_variable, name=None, initializer="zeros"
    ):
        """Add an all-zeros variable with the shape and dtype of a reference
        variable.
        """
        name = name or "var"
        if hasattr(reference_variable, "path"):
            name = reference_variable.path.replace("/", "_") + "_" + name
        else:
            name = (
                str(reference_variable.name).replace("/", "_").replace(":", "_")
                + "_"
                + name
            )
        return self.add_variable(
            shape=reference_variable.shape,
            initializer=initializer,
            dtype=reference_variable.dtype,
            name=name,
        )

    def _check_variables_are_known(self, variables):
        for v in variables:
            if self._var_key(v) not in self._trainable_variables_indices:
                raise ValueError(
                    f"Unknown variable: {v}. This optimizer can only "
                    "be called for the variables it was originally built with. "
                    "When working with a new set of variables, you should "
                    "recreate a new optimizer instance."
                )

    def assign(self, variable, value):
        """Assign a value to a variable.

        This should be used in optimizers instead of `variable.assign(value)` to
        support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign(value)

    def assign_add(self, variable, value):
        """Add a value to a variable.

        This should be used in optimizers instead of
        `variable.assign_add(value)` to support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign_add(value)

    def assign_sub(self, variable, value):
        """Subtract a value from a variable.

        This should be used in optimizers instead of
        `variable.assign_sub(value)` to support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign_sub(value)

    def update_step(self, gradient, variable, learning_rate):
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars):
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations

    def apply(self, grads, trainable_variables=None):
        """Update traininable variables according to provided gradient values.

        `grads` should be a list of gradient tensors
        with 1:1 mapping to the list of variables the optimizer was built with.

        `trainable_variables` can be provided
        on the first call to build the optimizer.
        """
        if len(grads) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return

        if trainable_variables is None:
            if not self.built:
                raise ValueError(
                    "When passing `grads` without `variables`, the optimizer "
                    "must already be built on a list of variables. "
                    "Call `optimizer.build(trainable_variables)` first. "
                )
            if len(grads) != len(self._trainable_variables_indices):
                raise ValueError(
                    "When passing `grads` as a list of gradient tensors, the "
                    f"gradients must match `optimizer.variables` one-to-on. "
                    f"Received a list of {len(grads)} gradients, but the "
                    f"optimizer is tracking {len(self._trainable_variables)} "
                    "trainable variables."
                )
            trainable_variables = self._trainable_variables
        else:
            trainable_variables = list(trainable_variables)
            # Optionally build optimizer.
            if not self.built:
                with backend.name_scope(self.name, caller=self):
                    self.build(trainable_variables)
                self.built = True
            self._check_variables_are_known(trainable_variables)

        with backend.name_scope(self.name, caller=self):
            # Overwrite targeted variables directly with their gradients if
            # their `overwrite_with_gradient` is set.
            grads, trainable_variables = (
                self._overwrite_variables_directly_with_gradients(
                    grads, trainable_variables
                )
            )

            # Filter empty gradients.
            grads, trainable_variables = self._filter_empty_gradients(
                grads, trainable_variables
            )
            if len(list(grads)) == 0:
                return

            # Unscale gradients.
            scale = self.loss_scale_factor
            if scale is not None:
                grads = [g if g is None else g / scale for g in grads]

            # Apply gradient updates.
            self._backend_apply_gradients(grads, trainable_variables)
            # Apply variable constraints after applying gradients.
            for variable in trainable_variables:
                if variable.constraint is not None:
                    variable.assign(variable.constraint(variable))

    def _backend_apply_gradients(self, grads, trainable_variables):
        """Apply method that can be overridden by different backends.

        JAX overrides it in order to deal with statelessness in gradient
        accumulation and EMA handling.

        The below implementation is intended to be generally backend-agnostic,
        but may not work with all backends.

        This method does 4 things:
        - Call the optimizer's update_step() to update trainable variables
            and optimizer variables.
        - Update EMA variables, if EMA is configured.
        - Update gradient accumulators, if gradient accumulation is configured.
        - Update the iteration counter.
        """
        if self.gradient_accumulation_steps:
            is_update_step = (
                self._iterations + 1
            ) % self.gradient_accumulation_steps == 0
            # `trainable_variables` might have been filtered in previous
            # processing steps, so we need to ensure the correct mapping between
            # `self._accumulated_gradients` and `trainable_variables`
            acc_grads = [
                self._accumulated_gradients[self._get_variable_index(v)]
                for v in trainable_variables
            ]

            def _update_step_fn(grads, trainable_variables):
                # Run update step with accumulated grads + reset accumulators
                steps = self.gradient_accumulation_steps
                grads = [
                    (g + acc_g) / steps for g, acc_g in zip(grads, acc_grads)
                ]

                # Apply clipping and weight decay.
                grads = self._clip_gradients(grads)
                self._apply_weight_decay(trainable_variables)

                self._backend_update_step(
                    grads, trainable_variables, self.learning_rate
                )
                self._backend_reset_gradient_accumulators()

            ops.cond(
                is_update_step,
                lambda: _update_step_fn(grads, trainable_variables),
                lambda: self._backend_increment_gradient_accumulators(
                    grads, acc_grads
                ),
            )
        else:
            # Apply clipping and weight decay.
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)

            # Run udpate step.
            self._backend_update_step(
                grads, trainable_variables, self.learning_rate
            )

        if self.use_ema:
            self._update_model_variables_moving_average(
                self._trainable_variables
            )
            if self.ema_overwrite_frequency:
                # Only when self.ema_overwrite_frequency is not None, we
                # overwrite the model variables.
                should_overwrite_model_vars = (
                    self.iterations + 1
                ) % self.ema_overwrite_frequency == 0
                ops.cond(
                    should_overwrite_model_vars,
                    lambda: self._overwrite_model_variables_with_average_value(
                        self._trainable_variables
                    ),
                    lambda: None,
                )
        # Update iteration counter.
        self._iterations.assign_add(1)

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.

        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        for grad, var in zip(grads, trainable_variables):
            self.update_step(grad, var, learning_rate)

    def _backend_reset_gradient_accumulators(self):
        for g_acc in self._accumulated_gradients:
            g_acc.assign(ops.zeros(g_acc.shape, dtype=g_acc.dtype))

    def _backend_increment_gradient_accumulators(self, grads, acc_grads):
        new_g_accs = [(g + acc_g) for g, acc_g in zip(grads, acc_grads)]
        for n_g_acc, g_acc in zip(new_g_accs, acc_grads):
            g_acc.assign(n_g_acc)

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        self._check_super_called()

        if not self.built:
            raise ValueError(
                f"To call `stateless_apply`, {self.__class__.__name__} "
                "must be built (i.e. its variables must have been created). "
                "You can build it via `optimizer.build(trainable_variables)`."
            )
        if len(optimizer_variables) != len(self.variables):
            raise ValueError(
                "Argument `optimizer_variables` must be a list of tensors "
                f"corresponding 1:1 to {self.__class__.__name__}().variables. "
                f"Received list with length {len(optimizer_variables)}, but "
                f"expected {len(self.variables)} variables."
            )
        if len(trainable_variables) != len(self._trainable_variables):
            raise ValueError(
                "Argument `optimizer_variables` must be a list of tensors "
                "corresponding 1:1 to the trainable variables list that "
                "the optimizer was built with. Received "
                f"len(trainable_variables) == {len(trainable_variables)} "
                "whereas the optimizer was built with "
                f"{len(self._trainable_variables)} variables."
            )

        # Gather variable mapping
        mapping = list(
            zip(self._trainable_variables, trainable_variables)
        ) + list(zip(self.variables, optimizer_variables))

        # Call in stateless scope
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.apply(grads)

        # Gather updated variables
        trainable_variables = []
        for v in self._trainable_variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                trainable_variables.append(new_v)
            else:
                trainable_variables.append(v)
        optimizer_variables = []
        for v in self.variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                optimizer_variables.append(new_v)
            else:
                optimizer_variables.append(v)
        return trainable_variables, optimizer_variables

    def scale_loss(self, loss):
        """Scale the loss before computing gradients.

        Scales the loss before gradients are computed in a `train_step`. This
        is primarily useful during mixed precision training to prevent numeric
        underflow.
        """
        if self.loss_scale_factor is not None:
            return loss * self.loss_scale_factor
        return loss

    @property
    def learning_rate(self):
        return self._get_current_learning_rate()

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if isinstance(self._learning_rate, backend.Variable):
            prev_lr_var = self._learning_rate
        else:
            prev_lr_var = None
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if isinstance(
                self._learning_rate, learning_rate_schedule.LearningRateSchedule
            ):
                raise TypeError(
                    "This optimizer was created with a `LearningRateSchedule`"
                    " object as its `learning_rate` constructor argument, "
                    "hence its learning rate is not settable. If you need the"
                    " learning rate to be settable, you should instantiate "
                    "the optimizer with a float `learning_rate` argument."
                )
            self._learning_rate.assign(learning_rate)
        if prev_lr_var is not None and not isinstance(
            self._learning_rate, backend.Variable
        ):
            # Untrack learning rate variable
            self._untrack_variable(prev_lr_var)

    def set_weights(self, weights):
        """Set the weights of the optimizer."""
        if not self.built:
            raise ValueError(
                "You are calling `set_weights()` on an optimizer that has not "
                "yet been built. Please call "
                "`optimizer.build(trainable_variables)` to create the "
                "optimizer weights before calling `set_weights()`."
            )
        for variable, weight in zip(self._variables, weights):
            if variable.shape != weight.shape:
                raise ValueError(
                    f"Optimizer variable {self._var_key(variable)} has shape "
                    f"{str(variable.shape)} not compatible with provided "
                    f"weight shape {str(weight.shape)}."
                )
            variable.assign(weight)

    def save_own_variables(self, store):
        """Get the state of this optimizer object."""
        for i, variable in enumerate(self.variables):
            store[str(i)] = variable.numpy()

    def load_own_variables(self, store):
        """Set the state of this optimizer object."""
        if len(store.keys()) != len(self.variables):
            msg = (
                f"Skipping variable loading for optimizer '{self.name}', "
                f"because it has {len(self.variables)} variables whereas "
                f"the saved optimizer has {len(store.keys())} variables. "
            )
            if len(self.variables) == 0:
                msg += (
                    "This is likely because the optimizer has not been "
                    "called/built yet."
                )
            warnings.warn(msg, stacklevel=2)
            return
        for i, variable in enumerate(self.variables):
            variable.assign(store[str(i)])

    def _get_current_learning_rate(self):
        if isinstance(
            self._learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            return self._learning_rate(self._iterations)
        elif callable(self._learning_rate):
            return self._learning_rate()
        return self._learning_rate

    def _overwrite_variables_directly_with_gradients(self, grads, vars):
        """Overwrite the variables directly by their gradients.

        This method is designed for a special case where we want to overwrite
        the variable directly with its computed gradient. For example, in float8
        training, new `scale` and `amax_history` are computed as gradients, and
        we want to overwrite them directly instead of following the typical
        procedure such as gradient descent with a learning rate, gradient
        clipping and weight decaying.

        After the update, the processed pairs will be filtered out.
        """
        # Shortcut for `tf.Variable` because it doesn't have a
        # `overwrite_with_gradient` attr
        if not hasattr(vars[0], "overwrite_with_gradient"):
            return grads, vars

        # Shallow copies
        filtered_grads = list(grads)
        filtered_vars = list(vars)

        # Iterate from right to left for safe popping
        for i in range(len(filtered_grads) - 1, -1, -1):
            g, v = filtered_grads[i], filtered_vars[i]
            if v.overwrite_with_gradient:
                if self.gradient_accumulation_steps:
                    # Utilize a stateless manner for JAX compatibility
                    steps = self.gradient_accumulation_steps
                    is_update_step = (self._iterations + 1) % steps == 0
                    acc_g = self._accumulated_gradients[
                        self._get_variable_index(v)
                    ]
                    # `ops.maximum` is utilized for gradient accumulation for
                    # `overwrite_with_gradient=True` variables
                    new_g_acc = ops.cond(
                        is_update_step,
                        lambda: ops.zeros(g.shape, dtype=g.dtype),
                        lambda: ops.maximum(g, acc_g),
                    )
                    new_g = ops.cond(
                        is_update_step,
                        lambda: ops.maximum(g, acc_g),
                        lambda: g,
                    )
                    new_v = ops.cond(
                        is_update_step, lambda: new_g, lambda: v.value
                    )
                    v.assign(new_v)
                    acc_g.assign(new_g_acc)
                else:
                    v.assign(g)
                filtered_grads.pop(i)
                filtered_vars.pop(i)
        return filtered_grads, filtered_vars

    def _filter_empty_gradients(self, grads, vars):
        filtered_grads = list(grads)
        filtered_vars = list(vars)
        missing_grad_vars = []

        # Iterate from right to left for safe popping
        for i in range(len(filtered_grads) - 1, -1, -1):
            if filtered_grads[i] is None:
                filtered_grads.pop(i)
                v = filtered_vars.pop(i)
                missing_grad_vars.append(v.path)

        if not filtered_grads:
            raise ValueError("No gradients provided for any variable.")
        if missing_grad_vars:
            warnings.warn(
                "Gradients do not exist for variables "
                f"{list(reversed(missing_grad_vars))} when minimizing the loss."
                " If using `model.compile()`, did you forget to provide a "
                "`loss` argument?"
            )
        return filtered_grads, filtered_vars

    def _clip_gradients(self, grads):
        if self.clipnorm and self.clipnorm > 0:
            return [
                self._clip_by_norm(g) if g is not None else g for g in grads
            ]
        elif self.global_clipnorm and self.global_clipnorm > 0:
            return clip_by_global_norm(grads, self.global_clipnorm)
        elif self.clipvalue and self.clipvalue > 0:
            v = self.clipvalue
            return [ops.clip(g, -v, v) if g is not None else g for g in grads]
        else:
            return grads

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables from weight decay.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `Variable`s to exclude from weight decay.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from weight decay. For example, `var_names=['bias']`
                excludes all bias variables from weight decay.
        """
        if hasattr(self, "_built") and self._built:
            raise ValueError(
                "`exclude_from_weight_decay()` can only be configued before "
                "the optimizer is built."
            )

        # Use a `set` for the ids of `var_list` to speed up the searching
        if var_list:
            self._exclude_from_weight_decay = set(
                self._var_key(variable) for variable in var_list
            )
        else:
            self._exclude_from_weight_decay = set()

        # Precompile the pattern for `var_names` to speed up the searching
        if var_names and len(var_names) > 0:
            self._exclude_from_weight_decay_pattern = re.compile(
                "|".join(set(var_names))
            )
        else:
            self._exclude_from_weight_decay_pattern = None

        # Reset cache
        self._exclude_from_weight_decay_cache = dict()

    def _use_weight_decay(self, variable):
        variable_id = self._var_key(variable)

        # Immediately return the value if `variable_id` hits the cache
        if not hasattr(self, "_exclude_from_weight_decay_cache"):
            self._exclude_from_weight_decay_cache = dict()
        if variable_id in self._exclude_from_weight_decay_cache:
            return self._exclude_from_weight_decay_cache[variable_id]

        # Determine whether the variable should apply weight decay or not
        exclude_from_weight_decay = getattr(
            self, "_exclude_from_weight_decay", set()
        )
        exclude_from_weight_decay_pattern = getattr(
            self, "_exclude_from_weight_decay_pattern", None
        )
        if variable_id in exclude_from_weight_decay:
            self._exclude_from_weight_decay_cache[variable_id] = False
            return False
        if exclude_from_weight_decay_pattern is not None:
            if (
                re.search(exclude_from_weight_decay_pattern, variable.name)
                is not None
            ):
                self._exclude_from_weight_decay_cache[variable_id] = False
                return False
        self._exclude_from_weight_decay_cache[variable_id] = True
        return True

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = ops.cast(self.learning_rate, variable.dtype)
                wd = ops.cast(self.weight_decay, variable.dtype)
                variable.assign(variable - variable * wd * lr)

    def _check_super_called(self):
        if not hasattr(self, "_lock"):
            raise RuntimeError(
                f"In optimizer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. "
                "Go add it!"
            )

    def _update_model_variables_moving_average(self, trainable_variables):
        """Update the stored moving average using the latest value."""
        if self.use_ema:
            for var, average in zip(
                trainable_variables, self._model_variables_moving_average
            ):
                not_first_step = ops.not_equal(self.iterations, 0)
                momentum = (
                    ops.cast(not_first_step, var.dtype) * self.ema_momentum
                )
                average.assign(momentum * average + (1 - momentum) * var)

    def _overwrite_model_variables_with_average_value(
        self, trainable_variables
    ):
        """Overwrite model variables with its moving average."""
        if len(trainable_variables) != len(
            self._model_variables_moving_average
        ):
            raise ValueError(
                f"The length of model variables ({len(trainable_variables)}) "
                "to override does not match the length of model variables "
                "stored in the optimizer "
                f"({len(self._model_variables_moving_average)}). Please "
                "check if the optimizer was called on your model."
            )
        for var, average_var in zip(
            trainable_variables, self._model_variables_moving_average
        ):
            var.assign(average_var)

    def finalize_variable_values(self, var_list):
        """Set the final value of model's trainable variables.

        Sometimes there are some extra steps before ending the variable updates,
        such as overriding the model variables with its average value.

        Args:
          var_list: list of model variables.
        """
        if self.use_ema:
            # If the optimizer uses EMA, then when finalizing, we replace the
            # model variable value with its moving average stored inside
            # optimizer.
            self._overwrite_model_variables_with_average_value(var_list)

    def _obj_type(self):
        return "Optimizer"

    def get_config(self):
        """Returns the config of the optimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Subclass optimizer should override this method to include other
        hyperparameters.

        Returns:
            Python dictionary.
        """

        if isinstance(
            self._learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            learning_rate = learning_rate_schedule.serialize(
                self._learning_rate
            )
        elif isinstance(self._learning_rate, backend.Variable):
            learning_rate = float(self._learning_rate.numpy())
        elif ops.is_tensor(self._learning_rate):
            learning_rate = float(self._learning_rate)
        elif callable(self._learning_rate):
            learning_rate = serialization_lib.serialize_keras_object(
                self._learning_rate
            )
        else:
            learning_rate = 0.5

        config = {
            "name": self.name,
            "learning_rate": learning_rate,
            "weight_decay": self.weight_decay,
            "clipnorm": self.clipnorm,
            "global_clipnorm": self.global_clipnorm,
            "clipvalue": self.clipvalue,
            "use_ema": self.use_ema,
            "ema_momentum": self.ema_momentum,
            "ema_overwrite_frequency": self.ema_overwrite_frequency,
            "loss_scale_factor": self.loss_scale_factor,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates an optimizer from its config.

        This method is the reverse of `get_config`, capable of instantiating the
        same optimizer from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of get_config.
            custom_objects: A Python dictionary mapping names to additional
              user-defined Python objects needed to recreate this optimizer.

        Returns:
            An optimizer instance.
        """
        if "learning_rate" in config:
            if isinstance(config["learning_rate"], dict):
                config["learning_rate"] = (
                    serialization_lib.deserialize_keras_object(
                        config["learning_rate"], custom_objects=custom_objects
                    )
                )
        return cls(**config)

    def __setattr__(self, name, value):
        # Prevent users from attaching state to the
        # layer before `super()` is called -- since that
        # state would silently not be tracked.
        if name != "_lock":
            self._check_super_called()
        # Track Variables.
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def _clip_by_norm(self, values, axes=None):
        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = ops.sum(ops.square(values), axes, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = ops.where(pred, l2sum, ops.ones_like(l2sum))
        l2norm = ops.where(pred, ops.sqrt(l2sum_safe), l2sum)
        intermediate = ops.multiply(values, self.clipnorm)
        values_clip = ops.convert_to_tensor(intermediate) / ops.maximum(
            l2norm, self.clipnorm
        )
        return values_clip

    def _untrack_variable(self, variable):
        previous_lock_state = self._tracker.locked
        self._tracker.unlock()
        self._tracker.untrack(variable)
        if previous_lock_state is True:
            self._tracker.lock()


base_optimizer_keyword_args = """name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).
"""


def global_norm(value_list):
    """Computes the global norm of multiple tensors."""
    squared_norms = [
        ops.sum(ops.square(v)) for v in value_list if v is not None
    ]
    squared_norm = ops.sum(ops.stack(squared_norms))
    return ops.sqrt(squared_norm)


def clip_by_global_norm(value_list, clip_norm):
    use_norm = global_norm(value_list)
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale_for_finite = clip_norm * ops.minimum(1.0 / use_norm, 1.0 / clip_norm)
    # If use_norm is any finite number, this is a no-op. For inf/-inf/NaN,
    # this will make scale NaN.
    scale = scale_for_finite + (use_norm - use_norm)
    return [v * scale if v is not None else v for v in value_list]
