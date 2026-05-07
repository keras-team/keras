"""Various learning rate schedule functions."""

import math

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.saving import serialization_lib


@keras_export("keras.optimizers.schedules.LearningRateSchedule")
class LearningRateSchedule:
    """The learning rate schedule base class.

    You can use a learning rate schedule to modulate how the learning rate
    of your optimizer changes over time.

    Several built-in learning rate schedules are available, such as
    `keras.optimizers.schedules.ExponentialDecay` or
    `keras.optimizers.schedules.PiecewiseConstantDecay`:

    ```python
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    ```

    A `LearningRateSchedule` instance can be passed in as the `learning_rate`
    argument of any optimizer.

    To implement your own schedule object, you should implement the `__call__`
    method, which takes a `step` argument (scalar integer tensor, the
    current training step count).
    Like for any other Keras object, you can also optionally
    make your object serializable by implementing the `get_config`
    and `from_config` methods.

    Example:

    ```python
    class MyLRSchedule(keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, initial_learning_rate):
            self.initial_learning_rate = initial_learning_rate

        def __call__(self, step):
            return self.initial_learning_rate / (step + 1)

    optimizer = keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))
    ```
    """

    def __call__(self, step):
        raise NotImplementedError(
            f"Learning rate schedule '{self.__class__.__name__}' "
            "must override `__call__(self, step)`."
        )

    def get_config(self):
        raise NotImplementedError(
            f"Learning rate schedule '{self.__class__.__name__}' "
            "must override `get_config()` in order to be serializable."
        )

    @classmethod
    def from_config(cls, config):
        """Instantiates a `LearningRateSchedule` from its config.

        Args:
            config: Output of `get_config()`.

        Returns:
            A `LearningRateSchedule` instance.
        """
        return cls(**config)


@keras_export("keras.optimizers.schedules.ExponentialDecay")
class ExponentialDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses an exponential decay schedule.

    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies an exponential decay function
    to an optimizer step, given a provided initial learning rate.

    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
        return initial_learning_rate * decay_rate ^ (step / decay_steps)
    ```

    If the argument `staircase` is `True`, then `step / decay_steps` is
    an integer division and the decayed learning rate follows a
    staircase function.

    You can pass this schedule directly into a `keras.optimizers.Optimizer`
    as the learning rate.
    Example: When fitting a Keras model, decay every 100000 steps with a base
    of 0.96:

    ```python
    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data, labels, epochs=5)
    ```

    The learning rate schedule is also serializable and deserializable using
    `keras.optimizers.schedules.serialize` and
    `keras.optimizers.schedules.deserialize`.

    Args:
        initial_learning_rate: A Python float. The initial learning rate.
        decay_steps: A Python integer. Must be positive. See the decay
            computation above.
        decay_rate: A Python float. The decay rate.
        staircase: Boolean.  If `True` decay the learning rate at discrete
            intervals.
        name: String.  Optional name of the operation.  Defaults to
            `"ExponentialDecay`".

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=False,
        name="ExponentialDecay",
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

        if self.decay_steps <= 0:
            raise ValueError(
                "Argument `decay_steps` must be > 0. "
                f"Received: decay_steps={self.decay_steps}"
            )

    def __call__(self, step):
        with ops.name_scope(self.name):
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate
            )
            dtype = initial_learning_rate.dtype
            decay_steps = ops.cast(self.decay_steps, dtype)
            decay_rate = ops.cast(self.decay_rate, dtype)

            global_step_recomp = ops.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = ops.floor(p)
            return ops.multiply(initial_learning_rate, ops.power(decay_rate, p))

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        }


@keras_export("keras.optimizers.schedules.PiecewiseConstantDecay")
class PiecewiseConstantDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a piecewise constant decay schedule.

    The function returns a 1-arg callable to compute the piecewise constant
    when passed the current optimizer step. This can be useful for changing the
    learning rate value across different invocations of optimizer functions.

    Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
        for the next 10000 steps, and 0.1 for any additional steps.

    ```python
    step = ops.array(0)
    boundaries = [100000, 110000]
    values = [1.0, 0.5, 0.1]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)
    ```

    You can pass this schedule directly into a `keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `keras.optimizers.schedules.serialize` and
    `keras.optimizers.schedules.deserialize`.

    Args:
        boundaries: A list of Python numbers with strictly increasing
            entries, and with all elements having the same type as the
            optimizer step.
        values: A list of Python numbers that specifies the values for the
            intervals defined by `boundaries`. It should have one more
            element than `boundaries`, and all elements should have the same
            type.
        name: A string. Optional name of the operation. Defaults to
            `"PiecewiseConstant"`.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as the boundary tensors.

        The output of the 1-arg function that takes the `step`
        is `values[0]` when `step <= boundaries[0]`,
        `values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`,
        ..., and `values[-1]` when `step > boundaries[-1]`.


    Raises:
        ValueError: if the number of elements in the `boundaries` and `values`
        lists do not match.
    """

    def __init__(self, boundaries, values, name="PiecewiseConstant"):
        super().__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError(
                "The length of boundaries should be 1 less than the length of "
                f"values. Received: boundaries={boundaries} of length "
                f"{len(boundaries)}, and values={values} "
                f"of length {len(values)}."
            )

        self.boundaries = boundaries
        self.values = values
        self.name = name

    def __call__(self, step):
        with ops.name_scope(self.name):
            boundaries = [ops.convert_to_tensor(x) for x in self.boundaries]
            values = [ops.convert_to_tensor(x) for x in self.values]
            step = ops.convert_to_tensor(step)

            for i, b in enumerate(boundaries):
                if b.dtype != step.dtype:
                    # We cast the boundaries to have the same type as the step
                    b = ops.cast(b, step.dtype)
                    boundaries[i] = b

            result_dtype = values[0].dtype
            result_value = ops.array(0, dtype=result_dtype)

            # For each range between boundaries, we check whether the step is
            # within that range, cast the resulting boolean to a number,
            # and multiply the result by the corresponding value for the range.
            # Taking the sum of these yields a piecewise constant function.
            step_less_than_first_boundary = ops.cast(
                step <= boundaries[0], result_dtype
            )
            result_value += step_less_than_first_boundary * values[0]

            step_greater_than_last_boundary = ops.cast(
                step > boundaries[-1], result_dtype
            )
            result_value += step_greater_than_last_boundary * values[-1]

            for low, high, value in zip(
                boundaries[:-1], boundaries[1:], values[1:-1]
            ):
                # Need to bind v here; can do this with lambda v=v: ...
                step_in_range = ops.cast(
                    (step > low) & (step <= high), result_dtype
                )
                result_value += step_in_range * value

            return result_value

    def get_config(self):
        return {
            "boundaries": self.boundaries,
            "values": self.values,
            "name": self.name,
        }


@keras_export("keras.optimizers.schedules.PolynomialDecay")
class PolynomialDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a polynomial decay schedule.

    It is commonly observed that a monotonically decreasing learning rate, whose
    degree of change is carefully chosen, results in a better performing model.
    This schedule applies a polynomial decay function to an optimizer step,
    given a provided `initial_learning_rate`, to reach an `end_learning_rate`
    in the given `decay_steps`.

    It requires a `step` value to compute the decayed learning rate. You
    can just pass a backend variable that you increment at each training
    step.

    The schedule is a 1-arg callable that produces a decayed learning rate
    when passed the current optimizer step. This can be useful for changing the
    learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        return ((initial_learning_rate - end_learning_rate) *
                (1 - step / decay_steps) ^ (power)
               ) + end_learning_rate
    ```

    If `cycle` is True then a multiple of `decay_steps` is used, the first one
    that is bigger than `step`.

    ```python
    def decayed_learning_rate(step):
        decay_steps = decay_steps * ceil(step / decay_steps)
        return ((initial_learning_rate - end_learning_rate) *
                (1 - step / decay_steps) ^ (power)
               ) + end_learning_rate
    ```

    You can pass this schedule directly into a `keras.optimizers.Optimizer`
    as the learning rate.
    Example: Fit a model while decaying from 0.1 to 0.01 in 10000 steps using
    sqrt (i.e. power=0.5):

    ```python
    ...
    starter_learning_rate = 0.1
    end_learning_rate = 0.01
    decay_steps = 10000
    learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)

    model.compile(optimizer=keras.optimizers.SGD(
                      learning_rate=learning_rate_fn),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data, labels, epochs=5)
    ```

    The learning rate schedule is also serializable and deserializable using
    `keras.optimizers.schedules.serialize` and
    `keras.optimizers.schedules.deserialize`.

    Args:
        initial_learning_rate: A Python float. The initial learning rate.
        decay_steps: A Python integer. Must be positive. See the decay
            computation above.
        end_learning_rate: A Python float. The minimal end learning rate.
        power: A Python float. The power of the polynomial. Defaults to
            `1.0`.
        cycle: A boolean, whether it should cycle beyond decay_steps.
        name: String.  Optional name of the operation. Defaults to
            `"PolynomialDecay"`.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        end_learning_rate=0.0001,
        power=1.0,
        cycle=False,
        name="PolynomialDecay",
    ):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name

        if self.decay_steps <= 0:
            raise ValueError(
                "Argument `decay_steps` must be > 0. "
                f"Received: decay_steps={self.decay_steps}"
            )

    def __call__(self, step):
        with ops.name_scope(self.name):
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate
            )
            dtype = initial_learning_rate.dtype
            end_learning_rate = ops.cast(self.end_learning_rate, dtype)
            power = ops.cast(self.power, dtype)

            global_step_recomp = ops.cast(step, dtype)
            decay_steps_recomp = ops.cast(self.decay_steps, dtype)
            if self.cycle:
                # Find the first multiple of decay_steps that is bigger than
                # global_step. If global_step is zero set the multiplier to 1
                multiplier = ops.where(
                    ops.equal(global_step_recomp, 0),
                    1.0,
                    ops.ceil(global_step_recomp / self.decay_steps),
                )
                decay_steps_recomp = ops.multiply(
                    decay_steps_recomp, multiplier
                )
            else:
                # Make sure that the global_step used is not bigger than
                # decay_steps.
                global_step_recomp = ops.minimum(
                    global_step_recomp, decay_steps_recomp
                )

            p = ops.divide(global_step_recomp, decay_steps_recomp)
            return ops.add(
                ops.multiply(
                    initial_learning_rate - end_learning_rate,
                    ops.power(1 - p, power),
                ),
                end_learning_rate,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "cycle": self.cycle,
            "name": self.name,
        }


@keras_export("keras.optimizers.schedules.InverseTimeDecay")
class InverseTimeDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses an inverse time decay schedule.

    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies the inverse decay function
    to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a backend variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
        return initial_learning_rate / (1 + decay_rate * step / decay_step)
    ```

    or, if `staircase` is `True`, as:

    ```python
    def decayed_learning_rate(step):
        return initial_learning_rate /
               (1 + decay_rate * floor(step / decay_step))
    ```

    You can pass this schedule directly into a `keras.optimizers.Optimizer`
    as the learning rate.
    Example: Fit a Keras model when decaying 1/t with a rate of 0.5:

    ```python
    ...
    initial_learning_rate = 0.1
    decay_steps = 1.0
    decay_rate = 0.5
    learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate, decay_steps, decay_rate)

    model.compile(optimizer=keras.optimizers.SGD(
                      learning_rate=learning_rate_fn),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data, labels, epochs=5)
    ```

    Args:
        initial_learning_rate: A Python float. The initial learning rate.
        decay_steps: How often to apply decay.
        decay_rate: A Python number.  The decay rate.
        staircase: Whether to apply decay in a discrete staircase, as o
        pposed to continuous, fashion.
        name: String.  Optional name of the operation.  Defaults to
            `"InverseTimeDecay"`.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=False,
        name="InverseTimeDecay",
    ):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

        if self.decay_steps <= 0:
            raise ValueError(
                "Argument `decay_steps` must be > 0. "
                f"Received: decay_steps={self.decay_steps}"
            )

    def __call__(self, step):
        with ops.name_scope(self.name):
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate
            )
            dtype = initial_learning_rate.dtype
            decay_steps = ops.cast(self.decay_steps, dtype)
            decay_rate = ops.cast(self.decay_rate, dtype)

            global_step_recomp = ops.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = ops.floor(p)
            const = ops.cast(ops.array(1), dtype)
            denom = ops.add(const, ops.multiply(decay_rate, p))
            return ops.divide(initial_learning_rate, denom)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        }


@keras_export("keras.optimizers.schedules.CosineDecay")
class CosineDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a cosine decay with optional warmup.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    For the idea of a linear warmup of our learning rate,
    see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).

    When we begin training a model, we often want an initial increase in our
    learning rate followed by a decay. If `warmup_target` is an int, this
    schedule applies a linear increase per optimizer step to our learning rate
    from `initial_learning_rate` to `warmup_target` for a duration of
    `warmup_steps`. Afterwards, it applies a cosine decay function taking our
    learning rate from `warmup_target` to `alpha` for a duration of
    `decay_steps`. If `warmup_target` is None we skip warmup and our decay
    will take our learning rate from `initial_learning_rate` to `alpha`.
    It requires a `step` value to  compute the learning rate. You can
    just pass a backend variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a warmup followed by a
    decayed learning rate when passed the current optimizer step. This can be
    useful for changing the learning rate value across different invocations of
    optimizer functions.

    Our warmup is computed as:

    ```python
    def warmup_learning_rate(step):
        completed_fraction = step / warmup_steps
        total_delta = target_warmup - initial_learning_rate
        return completed_fraction * total_delta
    ```

    And our decay is computed as:

    ```python
    if warmup_target is None:
        initial_decay_lr = initial_learning_rate
    else:
        initial_decay_lr = warmup_target

    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_decay_lr * decayed
    ```

    Example usage without warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0.1
    lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    ```

    Example usage with warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0
    warmup_steps = 1000
    target_learning_rate = 0.1
    lr_warmup_decayed_fn = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )
    ```

    You can pass this schedule directly into a `keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `keras.optimizers.schedules.serialize` and
    `keras.optimizers.schedules.deserialize`.

    Args:
        initial_learning_rate: A Python float. The initial learning rate.
        decay_steps: A Python int. Number of steps to decay over.
        alpha: A Python float. Minimum learning rate value for decay as a
            fraction of `initial_learning_rate`.
        name: String. Optional name of the operation.  Defaults to
            `"CosineDecay"`.
        warmup_target: A Python float. The target learning rate for our
            warmup phase. Will cast to the `initial_learning_rate` datatype.
            Setting to `None` will skip warmup and begins decay phase from
            `initial_learning_rate`. Otherwise scheduler will warmup from
            `initial_learning_rate` to `warmup_target`.
        warmup_steps: A Python int. Number of steps to warmup over.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        alpha=0.0,
        name="CosineDecay",
        warmup_target=None,
        warmup_steps=0,
    ):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target

        if self.decay_steps <= 0:
            raise ValueError(
                "Argument `decay_steps` must be > 0. "
                f"Received: decay_steps={self.decay_steps}"
            )

    def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
        with ops.name_scope(self.name):
            completed_fraction = ops.divide(step, decay_steps)
            pi = ops.array(math.pi, dtype=dtype)
            cosine_decayed = 0.5 * (
                1.0 + ops.cos(ops.multiply(pi, completed_fraction))
            )
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return ops.multiply(decay_from_lr, decayed)

    def _warmup_function(
        self, step, warmup_steps, warmup_target, initial_learning_rate
    ):
        with ops.name_scope(self.name):
            completed_fraction = step / warmup_steps
            total_step_delta = warmup_target - initial_learning_rate
            return total_step_delta * completed_fraction + initial_learning_rate

    def __call__(self, step):
        with ops.name_scope(self.name):
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate
            )
            dtype = initial_learning_rate.dtype
            decay_steps = ops.cast(self.decay_steps, dtype)
            global_step_recomp = ops.cast(step, dtype)

            if self.warmup_target is None:
                global_step_recomp = ops.minimum(
                    global_step_recomp, decay_steps
                )
                return self._decay_function(
                    global_step_recomp,
                    decay_steps,
                    initial_learning_rate,
                    dtype,
                )

            warmup_target = ops.cast(self.warmup_target, dtype)
            warmup_steps = ops.cast(self.warmup_steps, dtype)

            global_step_recomp = ops.minimum(
                global_step_recomp, decay_steps + warmup_steps
            )

            return ops.cond(
                global_step_recomp < warmup_steps,
                lambda: self._warmup_function(
                    global_step_recomp,
                    warmup_steps,
                    warmup_target,
                    initial_learning_rate,
                ),
                lambda: self._decay_function(
                    global_step_recomp - warmup_steps,
                    decay_steps,
                    warmup_target,
                    dtype,
                ),
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
        }


@keras_export("keras.optimizers.schedules.CosineDecayRestarts")
class CosineDecayRestarts(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a cosine decay schedule with restarts.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function with
    restarts to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a backend variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.

    The learning rate multiplier first decays
    from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
    restart is performed. Each new warm restart runs for `t_mul` times more
    steps and with `m_mul` times initial learning rate as the new learning rate.

    Example:
    ```python
    first_decay_steps = 1000
    lr_decayed_fn = (
        keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps))
    ```

    You can pass this schedule directly into a `keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `keras.optimizers.schedules.serialize` and
    `keras.optimizers.schedules.deserialize`.

    Args:
        initial_learning_rate: A Python float. The initial learning rate.
        first_decay_steps: A Python integer. Number of steps to decay over.
        t_mul: A Python float. Used to derive the number of iterations in
            the i-th period.
        m_mul: A Python float. Used to derive the initial learning rate of
            the i-th period.
        alpha: A Python float. Minimum learning rate value as a fraction of
            the `initial_learning_rate`.
        name: String. Optional name of the operation. Defaults to
            `"SGDRDecay"`.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        name="SGDRDecay",
    ):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name

        if self.first_decay_steps <= 0:
            raise ValueError(
                "Argument `first_decay_steps` must be > 0. "
                f"Received: first_decay_steps={self.first_decay_steps}"
            )

    def __call__(self, step):
        with ops.name_scope(self.name):
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate
            )
            dtype = initial_learning_rate.dtype
            first_decay_steps = ops.cast(self.first_decay_steps, dtype)
            alpha = ops.cast(self.alpha, dtype)
            t_mul = ops.cast(self._t_mul, dtype)
            m_mul = ops.cast(self._m_mul, dtype)

            global_step_recomp = ops.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    # ops.log is sensitive to the precision of dtype, so we need
                    # the additional casting
                    i_restart = ops.floor(
                        ops.log(
                            ops.cast(
                                1.0 - completed_fraction * (1.0 - t_mul), dtype
                            )
                        )
                        / ops.log(t_mul)
                    )

                    sum_r = ops.divide(
                        1.0 - ops.power(t_mul, i_restart), (1.0 - t_mul)
                    )
                    completed_fraction = ops.divide(
                        ops.subtract(completed_fraction, sum_r),
                        ops.power(t_mul, i_restart),
                    )

                else:
                    i_restart = ops.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction

            i_restart, completed_fraction = ops.cond(
                ops.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True),
            )

            m_fac = ops.power(m_mul, i_restart)
            cosine_decayed = (
                0.5
                * m_fac
                * (
                    1.0
                    + ops.cos(
                        ops.multiply(
                            ops.array(math.pi, dtype=dtype), completed_fraction
                        )
                    )
                )
            )
            decayed = ops.add(ops.multiply((1 - alpha), cosine_decayed), alpha)

            return ops.multiply(initial_learning_rate, decayed)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": self.name,
        }


@keras_export("keras.optimizers.schedules.serialize")
def serialize(learning_rate_schedule):
    """Serializes a `LearningRateSchedule` into a JSON-compatible dict.

    Args:
        learning_rate_schedule: The `LearningRateSchedule` object to serialize.

    Returns:
        A JSON-serializable dict representing the object's config.

    Example:

    >>> lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    ...     0.1, decay_steps=100000, decay_rate=0.96, staircase=True)
    >>> keras.optimizers.schedules.serialize(lr_schedule)
    {'module': 'keras.optimizers.schedules',
    'class_name': 'ExponentialDecay', 'config': {...},
    'registered_name': None}
    """
    return serialization_lib.serialize_keras_object(learning_rate_schedule)


@keras_export("keras.optimizers.schedules.deserialize")
def deserialize(config, custom_objects=None):
    """Instantiates a `LearningRateSchedule` object from a serialized form.

    Args:
        config: The serialized form of the `LearningRateSchedule`. Dictionary of
            the form {'class_name': str, 'config': dict}.
        custom_objects: A dictionary mapping class names (or function names) of
            custom (non-Keras) objects to class/functions.

    Returns:
        A `LearningRateSchedule` object.

    Example:

    ```python
    # Configuration for PolynomialDecay
    config = {
        'class_name': 'PolynomialDecay',
        'config': {'cycle': False,
            'decay_steps': 10000,
            'end_learning_rate': 0.01,
            'initial_learning_rate': 0.1,
            'name': None,
            'power': 0.5
        }
    }
    lr_schedule = keras.optimizers.schedules.deserialize(config)
    ```
    """
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="decay",
    )
