# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class of optimizer."""

import abc
import re

import tensorflow.compat.v2 as tf
from absl import logging

from keras import backend
from keras import initializers
from keras.optimizers.optimizer_v2 import utils as optimizer_utils
from keras.optimizers.schedules import learning_rate_schedule
from keras.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


class _BaseOptimizer(tf.__internal__.tracking.AutoTrackable):
    """Optimizer base class, which only supports non-distribute use case."""

    def __init__(
        self,
        name,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        **kwargs,
    ):
        self.name = name
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm
        self.clipvalue = clipvalue
        self.use_ema = use_ema
        self.jit_compile = jit_compile
        if not tf.config.list_physical_devices("GPU"):
            # Optimizer only benefits from XLA when training on GPU. So if no
            # GPU is found, we turn off XLA.
            self.jit_compile = False
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
                    "`ema_overwrite_frequency` must be an integer > 1 or None. "
                    "Received: ema_overwrite_frequency="
                    f"{ema_overwrite_frequency}"
                )
        self.ema_momentum = ema_momentum
        self.ema_overwrite_frequency = ema_overwrite_frequency

        if self.clipnorm is not None and self.global_clipnorm is not None:
            raise ValueError(
                "At most one of `clipnorm` and `global_clipnorm` can "
                f"be set. Received: clipnorm={self.clipnorm}, "
                f"global_clipnorm={self.global_clipnorm}."
            )

        self._variables = []
        self._create_iteration_variable()
        self._process_kwargs(kwargs)

    def _create_iteration_variable(self):
        """Create the iterations counter variable."""
        with tf.init_scope():
            # Lift the variable creation to init scope to avoid environment
            # issue.
            self._iterations = tf.Variable(
                0, name="iteration", dtype=tf.int64, trainable=False
            )
        self._variables.append(self._iterations)

    def _process_kwargs(self, kwargs):
        # Remove the `is_legacy_optimizer` arg, which is for serialization only.
        kwargs.pop("is_legacy_optimizer", None)
        lr = kwargs.pop("lr", None)
        if lr:
            logging.warning(
                "`lr` is deprecated, please use "
                "`learning_rate` instead, or use the legacy optimizer, e.g.,"
                f"tf.keras.optimizers.legacy.{self.__class__.__name__}."
            )
        legacy_kwargs = {
            "decay",
            "gradient_aggregator",
            "gradient_transformers",
        }
        for k in kwargs:
            if k in legacy_kwargs:
                raise ValueError(
                    f"{k} is deprecated in the new Keras optimizer, please"
                    "check the docstring for valid arguments, or use the "
                    "legacy optimizer, e.g., "
                    f"tf.keras.optimizers.legacy.{self.__class__.__name__}."
                )
            else:
                raise TypeError(
                    f"{k} is not a valid argument, kwargs should be empty "
                    " for `optimizer_experimental.Optimizer`."
                )

    def _create_or_restore_slot_variable(self, **kwargs):
        raise ValueError(
            "You are trying to restore a checkpoint from a legacy Keras "
            "optimizer into a v2.11+ Optimizer, which can cause "
            "errors. Please update the optimizer referenced in your code "
            "to be an instance of "
            "`tf.keras.optimizers.legacy.Optimizer`, e.g.: "
            f"`tf.keras.optimizers.legacy.{self.__class__.__name__}`."
        )

    def _var_key(self, variable):
        """Get a unique identifier of the given variable."""
        # Get the distributed variable if it exists.
        # TODO(b/199214315): replace _unique_id with ref() after fixing ref()
        # issues on AggregatingVariable.
        return variable._unique_id

    def _deduplicate_sparse_grad(self, grads):
        """Deduplicate sparse gradient.

        For sparse gradients, i.e., gradient is of type `tf.IndexedSlices`,
        it is possible that `gradient.indices` has duplicated indices.
        This function adds up values for the duplicated indices, and returns
        a `tf.IndexedSlices` with indices of unique values.
        """
        processed_grads = []
        for grad in grads:
            if isinstance(grad, tf.IndexedSlices):
                values = grad.values
                indices = grad.indices
                unique_indices, new_index_positions = tf.unique(indices)
                summed_values = tf.math.unsorted_segment_sum(
                    values, new_index_positions, tf.shape(unique_indices)[0]
                )
                processed_grads.append(
                    tf.IndexedSlices(
                        summed_values, unique_indices, grad.dense_shape
                    )
                )
            else:
                processed_grads.append(grad)

        return processed_grads

    @abc.abstractmethod
    def update_step(self, gradient, variable):
        """Function to update variable value based on given gradients.

        This method must be implemented in customized optimizers.

        Args:
          gradient: backpropagated gradient of the given variable.
          variable: variable whose value needs to be updated.

        Returns:
          An `Operation` that applies the specified gradients.

        """
        raise NotImplementedError

    @tf.function(jit_compile=True)
    def _update_step_xla(self, gradient, variable, key):
        """A wrapper of `update_step` to enable XLA acceleration.

        Due to `tf.function` tracing mechanism, for (gradient, variable) pairs
        of the same shape and dtype, the execution graph always invoke the first
        pair it has seen. Thus, we need a `key` argument to make each (gradient,
        variable) pair unique. In additions, XLA cannot understand string input,
        so the key is an integer.

        Args:
          gradient: backpropagated gradient of the given variable.
          variable: variable whose value needs to be updated.
          key (int): a unique key that identifies the variable.

        Returns:
          An `Operation` that applies the specified gradients.
        """
        return self._update_step(gradient, variable)

    def _update_step(self, gradient, variable):
        if getattr(variable, "_unique_id", None) is None:
            # Variable has no `_unique_id` if called during `model.save()`, in
            # which case we do not want to update the variable.
            return
        if self._var_key(variable) not in self._index_dict:
            raise KeyError(
                f"The optimizer cannot recognize variable {variable.name}. "
                "This usually means you are trying to call the optimizer to "
                "update different parts of the model separately. Please call "
                "`optimizer.build(variables)` with the full list of trainable "
                "variables before the training loop or use legacy optimizer "
                "`tf.keras.optimizers.legacy.{self.__class__.__name__}."
            )
        self.update_step(gradient, variable)

    def compute_gradients(self, loss, var_list, tape=None):
        """Compute gradients of loss on trainable variables.

        Args:
          loss: `Tensor` or callable. If a callable, `loss` should take no
            arguments and return the value to minimize.
          var_list: list or tuple of `Variable` objects to update to minimize
            `loss`, or a callable returning the list or tuple of `Variable`
            objects. Use callable when the variable list would otherwise be
            incomplete before `minimize` since the variables are created at the
            first time `loss` is called.
          tape: (Optional) `tf.GradientTape`. If `loss` is provided as a
            `Tensor`, the tape that computed the `loss` must be provided.

        Returns:
          A list of (gradient, variable) pairs. Variable is always present, but
          gradient can be `None`.
        """
        if not callable(loss) and tape is None:
            raise ValueError(
                "`tape` is required when a `Tensor` loss is passed. "
                f"Received: loss={loss}, tape={tape}."
            )
        if tape is None:
            tape = tf.GradientTape()
        if callable(loss):
            with tape:
                if not callable(var_list):
                    tape.watch(var_list)
                loss = loss()
                if callable(var_list):
                    var_list = var_list()

        grads = tape.gradient(loss, var_list)
        return list(zip(grads, var_list))

    def _clip_gradients(self, grads):
        clipped_grads = []
        if self.clipnorm and self.clipnorm > 0:
            for g in grads:
                if g is None:
                    clipped_grads.append(g)
                else:
                    clipped_grads.append(tf.clip_by_norm(g, self.clipnorm))
            return clipped_grads

        if self.global_clipnorm and self.global_clipnorm > 0:
            return tf.clip_by_global_norm(grads, self.global_clipnorm)[0]

        if self.clipvalue and self.clipvalue > 0:
            for g in grads:
                if g is None:
                    clipped_grads.append(g)
                else:
                    clipped_grads.append(
                        tf.clip_by_value(
                            g,
                            clip_value_min=-self.clipvalue,
                            clip_value_max=self.clipvalue,
                        )
                    )
            return clipped_grads

        return grads

    @property
    def iterations(self):
        """The number of training steps this `optimizer` has run.

        By default, iterations would be incremented by one every time
        `apply_gradients()` is called.
        """
        return self._iterations

    @iterations.setter
    def iterations(self, variable):
        if getattr(self, "_built", False):
            raise RuntimeError(
                "Cannot set `iterations` to a new Variable after "
                "the Optimizer weights have been created. Here it is "
                f"attempting to set `iterations` to {variable}."
                "Usually this means you are trying to set `iterations`"
                " after calling `apply_gradients()`. Please set "
                "`iterations` before calling `apply_gradients()`."
            )
        self._iterations = variable

    @property
    def learning_rate(self):
        if not hasattr(self, "_learning_rate") or self._learning_rate is None:
            raise ValueError(
                "Missing learning rate, please set self.learning_rate at"
                " optimizer creation time."
            )
        lr = self._learning_rate
        if isinstance(lr, learning_rate_schedule.LearningRateSchedule):
            # If the optimizer takes in LearningRateSchedule, then each call to
            # learning_rate would return `self._current_learning_rate`, which is
            # updated at each call to `apply_gradients`.
            return self._current_learning_rate
        return lr

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
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

    @property
    @doc_controls.do_not_generate_docs
    def lr(self):
        """Alias of `learning_rate()`.

        `lr()` is heavily called in workflows using `optimizer_v2.OptimizerV2`,
        so we keep it for backward compabitliy.
        """
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    def _build_learning_rate(self, learning_rate):
        with tf.init_scope():
            if isinstance(
                learning_rate, learning_rate_schedule.LearningRateSchedule
            ):
                # Create a variable to hold the current learning rate.
                current_learning_rate = tf.convert_to_tensor(
                    learning_rate(self.iterations)
                )
                self._current_learning_rate = tf.Variable(
                    current_learning_rate,
                    name="current_learning_rate",
                    dtype=current_learning_rate.dtype,
                    trainable=False,
                )
                return learning_rate

            return tf.Variable(
                learning_rate,
                name="learning_rate",
                dtype=backend.floatx(),
                trainable=False,
            )

    @abc.abstractmethod
    def build(self, var_list):
        """Initialize the optimizer's variables, such as momemtum variables.

        This function has to be implemented by subclass optimizers, and subclass
        optimizers need to call `super().build(var_list)`.

        Args:
          var_list: List of model variables to build optimizers on. For example,
            SGD optimizer with momentum will store one momentum variable
            corresponding to each model variable.
        """
        if getattr(self, "_built", False):
            return
        self._build_index_dict(var_list)
        if self.use_ema:
            self._model_variables_moving_average = []
            for var in var_list:
                # Make a copy of the model variables, we will use the copy to
                # store the moving average of model variables.
                self._model_variables_moving_average.append(
                    self.add_variable_from_reference(
                        var, "average", initial_value=var
                    )
                )

    def _build_index_dict(self, var_list):
        """Build variable to index dictionary.

        Build a dictionary that maps variable to the index of it in the given
        var_list.

        Args:
          var_list: List of variables to build index dict on.

        Returns:
          None
        """
        self._index_dict = {}
        for i, var in enumerate(var_list):
            var_key = self._var_key(var)
            self._index_dict[var_key] = i

    def add_variable(self, shape, dtype=None, initializer="zeros", name=None):
        """Create an optimizer variable.

        Args:
          shape: A list of integers, a tuple of integers, or a 1-D Tensor of
            type int32. Defaults to scalar if unspecified.
          dtype: The DType of the optimizer variable to be created. Defaults to
            `tf.keras.backend.floatx` if unspecified.
          initializer: string or callable. Initializer instance.
          name: The name of the optimizer variable to be created.

        Returns:
          An optimizer variable, in the format of tf.Variable.

        """
        if isinstance(initializer, str):
            initializer = initializers.get(initializer)
        if dtype is None:
            dtype = backend.floatx()
        if shape is None:
            shape = []
        variable = tf.Variable(
            initial_value=initializer(shape, dtype), name=name, trainable=False
        )
        self._variables.append(variable)
        return variable

    def add_variable_from_reference(
        self, model_variable, variable_name, shape=None, initial_value=None
    ):
        """Create an optimizer variable from model variable.

        Create an optimizer variable based on the information of model variable.
        For example, in SGD optimizer momemtum, for each model variable, a
        corresponding momemtum variable is created of the same shape and dtype.

        Args:
          model_variable: tf.Variable. The corresponding model variable to the
            optimizer variable to be created.
          variable_name: String. The name prefix of the optimizer variable to be
            created. The create variables name will follow the pattern
            `{variable_name}/{model_variable.name}`, e.g., `momemtum/dense_1`.
          shape: List or Tuple, defaults to None. The shape of the optimizer
            variable to be created. If None, the created variable will have the
            same shape as `model_variable`.
          initial_value: A Tensor, or Python object convertible to a Tensor,
            defaults to None. The initial value of the optimizer variable, if
            None, the initial value will be default to 0.

        Returns:
          An optimizer variable.
        """
        if initial_value is None:
            if shape is None:
                if model_variable.shape.rank is None:
                    # When the rank is None, we cannot get a concrete
                    # `model_variable.shape`, we use dynamic shape.
                    initial_value = tf.zeros_like(
                        model_variable, dtype=model_variable.dtype
                    )
                else:
                    # We cannot always use `zeros_like`, because some cases
                    # the shape exists while values don't.
                    initial_value = tf.zeros(
                        model_variable.shape, dtype=model_variable.dtype
                    )
            else:
                initial_value = tf.zeros(shape, dtype=model_variable.dtype)
        variable = tf.Variable(
            initial_value=initial_value,
            name=f"{variable_name}/{model_variable._shared_name}",
            dtype=model_variable.dtype,
            trainable=False,
        )
        self._variables.append(variable)
        return variable

    def minimize(self, loss, var_list, tape=None):
        """Minimize `loss` by updating `var_list`.

        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`. If you want to process the gradient before applying
        then call `tf.GradientTape` and `apply_gradients()` explicitly instead
        of using this function.

        Args:
          loss: `Tensor` or callable. If a callable, `loss` should take no
            arguments and return the value to minimize.
          var_list: list or tuple of `Variable` objects to update to minimize
            `loss`, or a callable returning the list or tuple of `Variable`
            objects.  Use callable when the variable list would otherwise be
            incomplete before `minimize` since the variables are created at the
            first time `loss` is called.
          tape: (Optional) `tf.GradientTape`.

        Returns:
          None
        """
        grads_and_vars = self.compute_gradients(loss, var_list, tape)
        self.apply_gradients(grads_and_vars)

    def _compute_current_learning_rate(self):
        if isinstance(
            self._learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            # Compute the current learning rate at the beginning of variable
            # update.
            if hasattr(self, "_current_learning_rate"):
                self._current_learning_rate.assign(
                    self._learning_rate(self.iterations)
                )
            else:
                current_learning_rate = tf.convert_to_tensor(
                    self._learning_rate(self.iterations)
                )
                self._current_learning_rate = tf.Variable(
                    current_learning_rate,
                    name="current_learning_rate",
                    dtype=current_learning_rate.dtype,
                    trainable=False,
                )

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables from weight decay.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `tf.Variable`s to exclude from weight decay.
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

        if var_list:
            self._exclude_from_weight_decay = [
                self._var_key(variable) for variable in var_list
            ]
        else:
            self._exclude_from_weight_decay = []
        self._exclude_from_weight_decay_names = var_names or []

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(
            self, "_exclude_from_weight_decay", []
        )
        exclude_from_weight_decay_names = getattr(
            self, "_exclude_from_weight_decay_names", []
        )
        variable_id = self._var_key(variable)
        for exclude_id in exclude_from_weight_decay:
            if variable_id == exclude_id:
                return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def apply_gradients(self, grads_and_vars, name=None):
        """Apply gradients to variables.

        Args:
          grads_and_vars: List of `(gradient, variable)` pairs.
          name: string, defaults to None. The name of the namescope to
            use when creating variables. If None, `self.name` will be used.

        Returns:
          A `tf.Variable`, representing the current iteration.

        Raises:
          TypeError: If `grads_and_vars` is malformed.
        """
        self._compute_current_learning_rate()
        grads_and_vars = list(grads_and_vars)
        if len(grads_and_vars) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return self._iterations
        grads, trainable_variables = zip(*grads_and_vars)
        scope_name = name or self.name or "optimizer"
        with tf.name_scope(scope_name):
            with tf.init_scope():
                # Lift variable creation to init scope to avoid environment
                # issues.
                self.build(trainable_variables)
        grads_and_vars = list(zip(grads, trainable_variables))
        grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        if len(list(grads_and_vars)) == 0:
            # Check again after filtering gradients.
            return self._iterations

        grads, trainable_variables = zip(*grads_and_vars)

        grads = self._clip_gradients(grads)
        grads = self._deduplicate_sparse_grad(grads)
        self._apply_weight_decay(trainable_variables)
        grads_and_vars = list(zip(grads, trainable_variables))
        iteration = self._internal_apply_gradients(grads_and_vars)

        # Apply variable constraints after applying gradients.
        for variable in trainable_variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))
        return iteration

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = tf.cast(self.learning_rate, variable.dtype)
                wd = tf.cast(self.weight_decay, variable.dtype)
                variable.assign_sub(variable * wd * lr)

    def _internal_apply_gradients(self, grads_and_vars):
        """Helper function of apply gradients.

        This is required for separating out distributed training logic.

        Args:
          grads_and_vars: List of (gradient, variable) pairs.
        """
        if self.jit_compile:
            for grad, var in grads_and_vars:
                self._update_step_xla(grad, var, id(self._var_key(var)))
        else:
            for grad, var in grads_and_vars:
                self._update_step(grad, var)
        return self.iterations.assign_add(1)

    def _update_model_variables_moving_average(self, var_list):
        """Update the stored moving average using the latest value."""
        if self.use_ema:
            for var, average in zip(
                var_list, self._model_variables_moving_average
            ):
                average.assign(
                    self.ema_momentum * average + (1 - self.ema_momentum) * var
                )

    def _overwrite_model_variables_with_average_value(self, var_list):
        """Overwrite model variables with its moving average."""
        if len(var_list) != len(self._model_variables_moving_average):
            raise ValueError(
                f"The length of model variables ({len(var_list)}) to "
                "override does not match the length of model variables "
                "stored in the optimizer "
                f"({len(self._model_variables_moving_average)}). Please "
                "check if the optimizer was called on your model."
            )
        self._overwrite_model_variables_with_average_value_helper(var_list)

    def _overwrite_model_variables_with_average_value_helper(self, var_list):
        """Helper function that overwrites model variables."""
        for var, average_var in zip(
            var_list, self._model_variables_moving_average
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

    def _serialize_hyperparameter(self, hyperparameter):
        """Serialize a hyperparameter that can be a numeric or callable."""
        if isinstance(
            hyperparameter, learning_rate_schedule.LearningRateSchedule
        ):
            return learning_rate_schedule.serialize(hyperparameter)
        if isinstance(hyperparameter, tf.Variable):
            return hyperparameter.numpy()
        if callable(hyperparameter):
            return hyperparameter()
        return hyperparameter

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
        config = {
            "name": self.name,
            "weight_decay": self.weight_decay,
            "clipnorm": self.clipnorm,
            "global_clipnorm": self.global_clipnorm,
            "clipvalue": self.clipvalue,
            "use_ema": self.use_ema,
            "ema_momentum": self.ema_momentum,
            "ema_overwrite_frequency": self.ema_overwrite_frequency,
            "jit_compile": self.jit_compile,
            "is_legacy_optimizer": False,
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
                config["learning_rate"] = learning_rate_schedule.deserialize(
                    config["learning_rate"], custom_objects=custom_objects
                )
        return cls(**config)

    def variables(self):
        """Returns variables of this optimizer."""
        return self._variables

    def set_weights(self, weights):
        """Set the weights of the optimizer.

        Args:
            weights: a list of `tf.Variable`s or numpy arrays, the target values
                of optimizer variables. It should have the same order as
                `self._variables`.
        """
        if not getattr(self, "_built", False):
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

    def _save_own_variables(self, store):
        """Get the state of this optimizer object."""
        for i, variable in enumerate(self.variables()):
            store[str(i)] = variable.numpy()

    def _load_own_variables(self, store):
        """Set the state of this optimizer object."""
        for i, variable in enumerate(self.variables()):
            variable.assign(store[str(i)])


base_optimizer_keyword_args = """name: String. The name to use
        for momentum accumulator weights created by
        the optimizer.
      weight_decay: Float, defaults to None. If set, weight decay is applied.
      clipnorm: Float. If set, the gradient of each weight is individually
        clipped so that its norm is no higher than this value.
      clipvalue: Float. If set, the gradient of each weight is clipped to be no
        higher than this value.
      global_clipnorm: Float. If set, the gradient of all weights is clipped so
        that their global norm is no higher than this value.
      use_ema: Boolean, defaults to False. If True, exponential moving average
        (EMA) is applied. EMA consists of computing an exponential moving
        average of the weights of the model (as the weight values change after
        each training batch), and periodically overwriting the weights with
        their moving average.
      ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`. This is  # noqa: E501
        the momentum to use when computing the EMA of the model's weights:
        `new_average = ema_momentum * old_average + (1 - ema_momentum) *
        current_variable_value`.
      ema_overwrite_frequency: Int or None, defaults to None. Only used if
        `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations, we
        overwrite the model variable by its moving average. If None, the optimizer  # noqa: E501
         does not overwrite model variables in the middle of training, and you
        need to explicitly overwrite the variables at the end of training
        by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
        variables in-place). When using the built-in `fit()` training loop, this
        happens automatically after the last epoch, and you don't need to do
        anything.
      jit_compile: Boolean, defaults to True. If True, the optimizer will use XLA  # noqa: E501
        compilation. If no GPU device is found, this flag will be ignored.
      **kwargs: keyword arguments only used for backward compatibility."""


@keras_export(
    "keras.optimizers.Optimizer",
    "keras.optimizers.experimental.Optimizer",
    v1=[],
)
class Optimizer(_BaseOptimizer):
    """Abstract optimizer base class.

    This class supports distributed training. If you want to implement your own
    optimizer, please subclass this class instead of _BaseOptimizer.

    Args:
      {{base_optimizer_keyword_args}}

    ### Usage

    ```python
    # Create an optimizer with the desired parameters.
    opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.1)
    var1, var2 = tf.Variable(1.0), tf.Variable(2.0)
    # `loss` is a callable that takes no argument and returns the value
    # to minimize.
    loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
    # Call minimize to update the list of variables.
    opt.minimize(loss, var_list=[var1, var2])
    ```

    ### Processing gradients before applying them

    Calling `minimize()` takes care of both computing the gradients and
    applying them to the variables. If you want to process the gradients
    before applying them you can instead use the optimizer in three steps:

    1.  Compute the gradients with `tf.GradientTape`.
    2.  Process the gradients as you wish.
    3.  Apply the processed gradients with `apply_gradients()`.

    Example:

    ```python
    # Create an optimizer.
    opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.1)
    var1, var2 = tf.Variable(1.0), tf.Variable(2.0)

    # Compute the gradients for a list of variables.
    with tf.GradientTape() as tape:
      loss = 3 * var1 * var1 + 2 * var2 * var2
    grads = tape.gradient(loss, [var1, var2])

    # Process the gradients.
    grads[0] = grads[0] + 1

    # Ask the optimizer to apply the gradients on variables.
    opt.apply_gradients(zip(grads, [var1, var2]))
    ```

    ### Dynamic learning rate

    Dynamic learning rate can be achieved by setting learning rate as a built-in
    or customized `tf.keras.optimizers.schedules.LearningRateSchedule`.

    Example:

    >>> var = tf.Variable(np.random.random(size=(1,)))
    >>> learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    ...   initial_learning_rate=.01, decay_steps=20, decay_rate=.1)
    >>> opt = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate)
    >>> loss = lambda: 3 * var
    >>> opt.minimize(loss, var_list=[var])

    ### Gradients clipping

    Users can clip the gradients before applying to variables by setting
    `clipnorm`, `clipvalue` and `global_clipnorm`. Notice that `clipnorm` and
    `global_clipnorm` can only have one being set.

    Example:

    >>> opt = tf.keras.optimizers.experimental.SGD(learning_rate=1, clipvalue=1)
    >>> var1, var2 = tf.Variable(2.0), tf.Variable(2.0)
    >>> with tf.GradientTape() as tape:
    ...   loss = 2 * var1 + 2 * var2
    >>> grads = tape.gradient(loss, [var1, var2])
    >>> print([grads[0].numpy(), grads[1].numpy()])
    [2.0, 2.0]
    >>> opt.apply_gradients(zip(grads, [var1, var2]))
    >>> # Without clipping, we should get [0, 0], but as gradients are clipped
    >>> # to have max value 1, we get [1.0, 1.0].
    >>> print([var1.numpy(), var2.numpy()])
    [1.0, 1.0]

    ### Using weight decay.

    Weight decay in certain scenarios can boost the model's performance. Keras
    has built-in support for weight decay in all optimizers. Users can apply
    weight decay by setting `weight_decay` argument.

    >>> opt = tf.keras.optimizers.experimental.SGD(1, weight_decay=0.004)
    >>> grads, var1, var2 = tf.zeros(()), tf.Variable(2.0), tf.Variable(2.0)
    >>> # You can exclude variables from weight decay, in this case we
    >>> # exclude `var2`.
    >>> opt.exclude_from_weight_decay(var_list=[var2])
    >>> opt.apply_gradients(zip([grads, grads], [var1, var2]))
    >>> print([var1.numpy(), var2.numpy()])
    [1.992, 2.0]


    ### Using exponential moving average.

    Empirically it has been found that using the exponential moving average
    (EMA) of the trained parameters of a deep network achieves a better
    performance than using its trained parameters directly. Keras optimizers
    allows users to compute this moving average and overwrite the model
    variables at desired time.

    Example:

    ```python
    # Create an SGD optimizer with EMA on. `ema_momentum` controls the decay
    # rate of the moving average. `ema_momentum=1` means no decay and the stored
    # moving average is always model variable's initial value before training.
    # Reversely, `ema_momentum=0` is equivalent to not using EMA.
    # `ema_overwrite_frequency=3` means every 3 iterations, we overwrite the
    # trainable variables with their moving average values.
    opt = tf.keras.optimizers.experimental.SGD(
        learning_rate=1,
        use_ema=True,
        ema_momentum=0.5,
        ema_overwrite_frequency=3)
    var1, var2 = tf.Variable(2.0), tf.Variable(2.0)
    with tf.GradientTape() as tape:
      loss = var1 + var2
    grads = tape.gradient(loss, [var1, var2])
    # First iteration: [var1, var2] = [1.0, 1.0]
    opt.apply_gradients(zip(grads, [var1, var2]))
    print([var1, var2])

    # Second iteration: [var1, var2] = [0.0, 0.0]
    opt.apply_gradients(zip(grads, [var1, var2]))
    print([var1, var2])

    # Third iteration, without EMA, we should see [var1, var2] = [-1.0, -1.0],
    # but overwriting results in [var1, var2] = [-0.125, -0.125]. The full
    # calculation for the moving average of var1 is:
    # var1=2*0.5**3+1*(1-0.5)*0.5**2+0*(1-0.5)*0.5**1+(-1)*(1-0.5)=-0.125.
    opt.apply_gradients(zip(grads, [var1, var2]))
    print([var1, var2])

    ```
    When optimizer is constructed with `use_ema=True`, in custom training loop,
    users can explicitly call `finalize_variable_values()` to overwrite
    trainable variables with their EMA values. `finalize_variable_values()` is
    by default called at the end of `model.fit()`.

    ### Use with `tf.distribute.Strategy`

    This optimizer class is `tf.distribute.Strategy` aware, which means it
    automatically sums gradients across all replicas. To aggregate gradients
    yourself, call `apply_gradients` with `skip_aggregate_gradients` set to
    True.  This is useful if you need to process aggregated gradients.

    ```python
    # This example is not runnable, it consists of dummy code for simple
    # tutorial.
    strategy = tf.distribute.experimental.TPUStrategy()

    with strategy.scope():
      opt = tf.keras.optimizers.experimental.SGD()
      model = magic_function_that_returns_model()
      gradients = magic_function_that_returns_gradients()
      # Custom logic to aggregate gradients.
      gradients = strategy.reduce("SUM", gradients, axis=None)
      opt.apply_gradients(zip(gradients, model.trainable_variables),
          skip_aggregate_gradients=True)
    ```

    ### Creating a custom optimizer

    If you intend to create your own optimization algorithm, please inherit from
    this class and override the following methods:

      - `build`: Create your optimizer-related variables, such as `momentums` in
        SGD optimizer.
      - `update_step`: Implement your optimizer's updating logic.
      - `get_config`: serialization of the optimizer, include all hyper
        parameters.

    Your optimizer would automatically be compatible with tensorflow distributed
    training if you subclass `optimizer_experimental.Optimizer`.

    """

    def __init__(
        self,
        name,
        weight_decay=0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        **kwargs,
    ):
        """Create a new Optimizer."""

        super().__init__(
            name,
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            ema_overwrite_frequency,
            jit_compile,
            **kwargs,
        )
        self._distribution_strategy = tf.distribute.get_strategy()

    def add_variable_from_reference(
        self, model_variable, variable_name, shape=None, initial_value=None
    ):
        strategy = tf.distribute.get_strategy()
        with strategy.extended.colocate_vars_with(model_variable):
            return super().add_variable_from_reference(
                model_variable, variable_name, shape, initial_value
            )

    def _var_key(self, variable):
        """Get a unique identifier of the given variable."""

        # Get the distributed variable if it exists.
        # TODO(b/197554203): replace _distributed_container() with a public api.
        if hasattr(variable, "_distributed_container"):
            variable = variable._distributed_container()
        elif (
            tf_utils.is_extension_type(variable)
            and hasattr(variable, "handle")
            and hasattr(variable.handle, "_distributed_container")
        ):
            # For ResourceVariables, the _distributed_container attribute
            # is added to their handle tensors.
            variable = variable.handle._distributed_container()
        return super()._var_key(variable)

    def aggregate_gradients(self, grads_and_vars):
        """Aggregate gradients on all devices.

        By default we will perform reduce_sum of gradients across devices. Users
        can implement their own aggregation logic by overriding this method.

        Args:
          grads_and_vars: List of (gradient, variable) pairs.

        Returns:
          List of (gradient, variable) pairs.
        """
        return optimizer_utils.all_reduce_sum_gradients(grads_and_vars)

    def apply_gradients(
        self,
        grads_and_vars,
        name=None,
        skip_gradients_aggregation=False,
        **kwargs,
    ):
        """Apply gradients to variables.

        Args:
          grads_and_vars: List of `(gradient, variable)` pairs.
          name: string, defaults to None. The name of the namescope to
            use when creating variables. If None, `self.name` will be used.
          skip_gradients_aggregation: If true, gradients aggregation will not be
            performed inside optimizer. Usually this arg is set to True when you
            write custom code aggregating gradients outside the optimizer.
          **kwargs: keyword arguments only used for backward compatibility.

        Returns:
          A `tf.Variable`, representing the current iteration.

        Raises:
          TypeError: If `grads_and_vars` is malformed.
          RuntimeError: If called in a cross-replica context.
        """
        # `experimental_aggregate_gradients` is an arg in `apply_gradients` of
        # v2 optimizer -- the reverse of `skip_gradients_aggregation`.
        # We read it from kwargs for backward compatibility.
        experimental_aggregate_gradients = kwargs.pop(
            "experimental_aggregate_gradients", True
        )
        if not skip_gradients_aggregation and experimental_aggregate_gradients:
            grads_and_vars = self.aggregate_gradients(grads_and_vars)
        return super().apply_gradients(grads_and_vars, name=name)

    def _apply_weight_decay(self, variables):
        # Apply weight decay in distributed setup.
        if self.weight_decay is None:
            return

        def distributed_apply_weight_decay(distribution, variables, **kwargs):
            def weight_decay_fn(variable):
                if self._use_weight_decay(variable):
                    lr = tf.cast(self.learning_rate, variable.dtype)
                    wd = tf.cast(self.weight_decay, variable.dtype)
                    variable.assign_sub(variable * wd * lr)

            for variable in variables:
                distribution.extended.update(
                    variable, weight_decay_fn, group=False
                )

        tf.__internal__.distribute.interim.maybe_merge_call(
            distributed_apply_weight_decay,
            self._distribution_strategy,
            variables,
        )

    def _internal_apply_gradients(self, grads_and_vars):
        return tf.__internal__.distribute.interim.maybe_merge_call(
            self._distributed_apply_gradients_fn,
            self._distribution_strategy,
            grads_and_vars,
        )

    def _overwrite_model_variables_with_average_value_helper(self, var_list):
        """Helper function to _overwrite_model_variables_with_average_value.

        This function overwrites variables on each device.
        Args:
          var_list: list of model variables.
        """
        strategy = self._distribution_strategy
        # Override model variable by the stored average value on all devices.
        for var, average_var in zip(
            var_list, self._model_variables_moving_average
        ):
            strategy.extended.update(
                var, lambda a, b: a.assign(b), args=(average_var,)
            )

    def _update_model_variables_moving_average(self, var_list):
        """Update the stored moving average using the latest value."""
        if self.use_ema:

            def update_average(average, var):
                average.assign(
                    self.ema_momentum * average + (1 - self.ema_momentum) * var
                )

            for var, average in zip(
                var_list, self._model_variables_moving_average
            ):
                self._distribution_strategy.extended.update(
                    average, update_average, args=(var,), group=False
                )

    def _distributed_apply_gradients_fn(
        self, distribution, grads_and_vars, **kwargs
    ):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad):
            if self.jit_compile:
                return self._update_step_xla(grad, var, id(self._var_key(var)))
            else:
                return self._update_step(grad, var)

        for grad, var in grads_and_vars:
            distribution.extended.update(
                var, apply_grad_to_update_var, args=(grad,), group=False
            )

        if self.use_ema:
            _, var_list = zip(*grads_and_vars)
            self._update_model_variables_moving_average(var_list)
            if self.ema_overwrite_frequency:
                # Only when self.ema_overwrite_frequency is not None, we
                # overwrite the model variables.
                should_overwrite_model_vars = (
                    self.iterations + 1
                ) % self.ema_overwrite_frequency == 0
                tf.cond(
                    tf.cast(should_overwrite_model_vars, tf.bool),
                    true_fn=lambda: self._overwrite_model_variables_with_average_value(  # noqa: E501
                        var_list
                    ),
                    false_fn=lambda: None,
                )
        return self.iterations.assign_add(1)


class RestoredOptimizer(Optimizer):
    def __init__(self):
        super().__init__("RestoredOptimizer")

    def get_config(self):
        raise NotImplementedError(
            "Restoring functional Optimizers from SavedModels is not currently "
            "supported. Please file a feature request if this limitation "
            "bothers you."
        )


# Register the optimizer for loading from saved_model purpose.
tf.__internal__.saved_model.load.register_revived_type(
    "experimentalOptimizer",
    lambda obj: isinstance(obj, Optimizer),
    versions=[
        tf.__internal__.saved_model.load.VersionedTypeRegistration(
            object_factory=lambda proto: RestoredOptimizer(),
            version=2,
            min_producer_version=1,
            min_consumer_version=1,
        )
    ],
)

Optimizer.__doc__ = Optimizer.__doc__.replace(
    "{{base_optimizer_keyword_args}}", base_optimizer_keyword_args
)
