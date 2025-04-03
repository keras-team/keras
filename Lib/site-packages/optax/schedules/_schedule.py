# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Optax Schedules.

Schedules may be used to anneal the value of a hyper-parameter over time; for
instance, they may be used to anneal the learning rate used to update an agent's
parameters or the exploration factor used to select actions.
"""

from typing import Iterable, Optional, Union

from absl import logging
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base
from optax.schedules import _join


def constant_schedule(value: Union[float, int]) -> base.Schedule:
  """Constructs a constant schedule.

  Args:
    value: value to be held constant throughout.

  Returns:
    schedule
      A function that maps step counts to values.

  Examples:
    >>> schedule_fn = optax.constant_schedule(5)
    >>> schedule_fn(0)
    5
    >>> schedule_fn(100)
    5
  """
  return lambda count: value


def polynomial_schedule(
    init_value: chex.Scalar,
    end_value: chex.Scalar,
    power: chex.Scalar,
    transition_steps: int,
    transition_begin: int = 0,
) -> base.Schedule:
  r"""Constructs a schedule with polynomial transition from init to end value.

  This function transitions the learning rate from an initial value
  (``init_value``) to a final value (``end_value``) over a specified number of
  steps (``transition_steps``) with a polynomial function of power ``power``.
  The transition can optionally begin after a specified number of initial steps
  (``transition_begin``).

  More precisely, the learning rate at iteration :math:`t` is given by:

  .. math::
    \begin{cases}
      I, & \text{if } t < B \\
      (I - E) \left( 1 - \frac{t - B}{T} \right)^{P} + E, &
        \text{if } B \leq t < B + T \\
      E, & \text{if } t \geq B + T
    \end{cases}

  where :math:`I` is the initial value, :math:`E` is the end value,
  :math:`B` is the transition begin, :math:`T` is the transition steps,
  and :math:`P` is the power used for the polynomial transition.

  Args:
    init_value: initial value for the scalar to be annealed.
    end_value: end value of the scalar to be annealed.
    power: the power of the polynomial used to transition from init to end.
    transition_steps: number of steps over which annealing takes place.
      The scalar starts changing at ``transition_begin`` steps and completes
      the transition by ``transition_begin + transition_steps`` steps.
      If ``transition_steps <= 0``, then the entire annealing process is
      disabled and the value is held fixed at ``init_value``.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at ``init_value``).

  Returns:
    schedule
      A function that maps step counts to values.

  Examples:
    >>> schedule_fn = optax.polynomial_schedule(
    ...    init_value=1.0, end_value=0.01, transition_steps=100, power=2)
    >>> schedule_fn(0)  # learning rate on the first iteration
    Array(1., dtype=float32, weak_type=True)
    >>> schedule_fn(100)  # learning rate on the last iteration
    Array(0.01, dtype=float32, weak_type=True)

    The following example uses a non-zero ``transition_begin``. In this case
    the learning rate is kept constant for the first ``transition_begin``
    iterations:

    >>> schedule_fn = optax.polynomial_schedule(
    ...    init_value=1.0,
    ...    end_value=0.01,
    ...    transition_steps=100,
    ...    transition_begin=5,
    ...    power=2,
    ... )
    >>> counts = [0, 5, 6, 104, 105, 110]
    >>> print(
    ...    *[f'count:{i} value:{schedule_fn(i):.4f}' for i in counts],
    ...    sep='\n')
    count:0 value:1.0000
    count:5 value:1.0000
    count:6 value:0.9803
    count:104 value:0.0101
    count:105 value:0.0100
    count:110 value:0.0100
  """
  if transition_steps <= 0:
    logging.info(
        'A polynomial schedule was set with a non-positive `transition_steps` '
        'value; this results in a constant schedule with value `init_value`.'
    )
    return lambda count: init_value

  if transition_begin < 0:
    logging.info(
        'A polynomial schedule was set with a negative `transition_begin` '
        'value; this will result in `transition_begin` falling back to `0`.'
    )
    transition_begin = 0

  def schedule(count):
    count = jnp.clip(count - transition_begin, 0, transition_steps)
    frac = 1 - count / transition_steps
    return (init_value - end_value) * (frac**power) + end_value

  return schedule


def linear_schedule(
    init_value: chex.Scalar,
    end_value: chex.Scalar,
    transition_steps: int,
    transition_begin: int = 0,
) -> base.Schedule:
  r"""Schedule with linear transition from ``init_value`` to ``end_value``.

  More precisely, the learning rate at iteration :math:`t` is given by:

  .. math::
    \begin{cases}
      I, & \text{if } t < B \\
      I + \frac{t - B}{T} (E - I), & \text{if } B \leq t < B + T \\
      E, & \text{if } t \geq B + T
    \end{cases}

  where :math:`I` is the initial value, :math:`E` is the end value,
  :math:`B` is the transition begin, and :math:`T` is the transition steps.

  This schedule is equivalent to :func:`optax.polynomial_schedule` with
  ``power=1``.

  Args:
    init_value: initial value for the scalar to be annealed.
    end_value: end value of the scalar to be annealed.
    transition_steps: number of steps over which annealing takes place. The
      scalar starts changing at ``transition_begin`` steps and completes the
      transition by ``transition_begin + transition_steps`` steps. If
      ``transition_steps <= 0``, then the entire annealing process is disabled
      and the value is held fixed at ``init_value``.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at ``init_value``).

  Returns:
    schedule
      A function that maps step counts to values.

  Examples:
    >>> schedule_fn = optax.linear_schedule(
    ...    init_value=1.0, end_value=0.01, transition_steps=100)
    >>> schedule_fn(0)  # learning rate on the first iteration
    Array(1., dtype=float32, weak_type=True)
    >>> schedule_fn(100)  # learning rate on the last iteration
    Array(0.01, dtype=float32, weak_type=True)
  """
  return polynomial_schedule(
      init_value=init_value,
      end_value=end_value,
      power=1,
      transition_steps=transition_steps,
      transition_begin=transition_begin,
  )


def piecewise_constant_schedule(
    init_value: float, boundaries_and_scales: Optional[dict[int, float]] = None
) -> base.Schedule:
  """Returns a function which implements a piecewise constant schedule.

  Args:
    init_value: An initial value ``init_v``.
    boundaries_and_scales: A map from boundaries ``b_i`` to non-negative scaling
      factors ``f_i``. For any step count `s`, the schedule returns ``init_v``
      scaled by the product of all factors ``f_i`` such that ``b_i < s``.

  Returns:
    schedule
      A function that maps step counts to values.
  """
  if boundaries_and_scales is not None:
    all_positive = all(scale >= 0.0 for scale in boundaries_and_scales.values())
    if not all_positive:
      raise ValueError(
          '`piecewise_constant_schedule` expects non-negative scale factors'
      )

  def schedule(count):
    v = init_value
    if boundaries_and_scales is not None:
      for threshold, scale in sorted(boundaries_and_scales.items()):
        indicator = jnp.maximum(0.0, jnp.sign(threshold - count))
        v = v * indicator + (1 - indicator) * scale * v
    return v

  return schedule


def exponential_decay(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> base.Schedule:
  """Constructs a schedule with either continuous or discrete exponential decay.

  This function applies an exponential decay function to a provided initial
  value. When ``count >= transition_begin`` the function returns the decayed
  value as:

  .. code-block::

    rate_factor = ((count - transition_begin) / transition_steps)
    decayed_value = init_value * (decay_rate ** rate_factor)

  If the argument ``staircase`` is ``True`` then ``count / transition_steps`` is
  an integer division and the decayed value follows a staircase function.

  Args:
    init_value: the initial learning rate.
    transition_steps: must be positive. See the decay computation above.
    decay_rate: must not be zero. The decay rate.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at `init_value`).
    staircase: if ``True``, decay the values at discrete intervals.
    end_value: the value at which the exponential decay stops. When ``decay_rate
      < 1``, ``end_value`` is treated as a lower bound, otherwise as an upper
      bound. Has no effect when ``decay_rate = 0``.

  Returns:
    schedule
      A function that maps step counts to values.
  """

  if transition_steps <= 0:
    logging.info(
        'An exponential schedule was set with a non-positive `transition_steps`'
        ' value; this will result in a constant schedule with value '
        '`init_value`.'
    )
    return lambda count: init_value

  if decay_rate == 0:
    logging.info(
        'An exponential schedule was set with a zero `decay_rate` value; '
        'this will result in a constant schedule with value `init_value`.'
    )
    return lambda count: init_value

  if transition_begin < 0:
    logging.info(
        'An exponential schedule was set with a negative `transition_begin` '
        'value; this will result in `transition_begin` falling back to `0`.'
    )
    transition_begin = 0

  if end_value is not None:
    clip_fn = jnp.maximum if decay_rate < 1.0 else jnp.minimum

  def schedule(count):
    decreased_count = count - transition_begin
    p = decreased_count / transition_steps
    if staircase:
      p = jnp.floor(p)
    decayed_value = jnp.where(
        decreased_count <= 0, init_value, init_value * jnp.power(decay_rate, p)
    )
    if end_value is not None:
      decayed_value = clip_fn(decayed_value, end_value)  # pylint: disable=undefined-variable
    return decayed_value

  return schedule


def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0,
    exponent: float = 1.0,
) -> base.Schedule:
  r"""Returns a function which implements cosine learning rate decay.

  This schedule smoothly decreases the learning rate over a specified number of
  steps (``decay_steps``). The decay follows a cosine function, with an optional
  exponent to modify the decay curve. A minimum value (``alpha``) ensures the
  learning rate does not drop entirely to zero.

  More precisely, the learning rate at iteration :math:`t` is given by:

  .. math::
    \begin{cases}
      \frac{I (1 - \alpha)}{2}(1+\cos(\pi\,\frac{t}{T})^p) + I \alpha\, 
      & \text{if } t \leq T \\
      I \alpha, & \text{if } t > T 
    \end{cases}

  where :math:`T` is the number of decay steps (``decay_steps``), :math:`p` is
  the ``exponent`` and :math:`I` is the initial value (``init_value``).

  Args:
    init_value: An initial value for the learning rate.
    decay_steps: Positive integer - the number of steps for which to apply
      the decay for.
    alpha: The minimum value of the multiplier used to adjust the
      learning rate. Defaults to 0.0.
    exponent:  The default decay is ``0.5 * (1 + cos(pi * t/T))``, where 
      ``t`` is the current timestep and ``T`` is the ``decay_steps``. The
      exponent modifies this to be ``(0.5 * (1 + cos(pi * t/T))) ** exponent``.
      Defaults to 1.0.

  Returns:
    schedule
      A function that maps step counts to values.

  References:
    Loshchilov et al., `SGDR: Stochastic Gradient Descent with Warm Restarts
    <https://arxiv.org/abs/1608.03983>`_, 2017
  """
  if not decay_steps > 0:
    raise ValueError(
        'The cosine_decay_schedule requires positive decay_steps, got'
        f' {decay_steps=}.'
    )

  def schedule(count):
    # Avoid int -> int32 overflow in jitted code.
    nonlocal decay_steps
    decay_steps, count = jax.tree.map(
        lambda x: float(x) if isinstance(x, int) else x, (decay_steps, count)
    )

    count = jnp.minimum(count, decay_steps)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / decay_steps))
    decayed = (1 - alpha) * cosine_decay**exponent + alpha
    return init_value * decayed

  return schedule


def _linear_interpolate(start: float, end: float, pct: float):
  return (end - start) * pct + start


def _cosine_interpolate(start: float, end: float, pct: float):
  return end + (start - end) / 2.0 * (jnp.cos(jnp.pi * pct) + 1)


def piecewise_interpolate_schedule(
    interpolate_type: str,
    init_value: float,
    boundaries_and_scales: Optional[dict[int, float]] = None,
) -> base.Schedule:
  """Returns a function which implements a piecewise interpolated schedule.

  Args:
    interpolate_type: 'linear' or 'cosine', specifying the interpolation
      strategy.
    init_value: An initial value ``init_v``.
    boundaries_and_scales: A map from boundaries ``b_i`` to non-negative scaling
      factors ``f_i``. At boundary step ``b_i``, the schedule returns ``init_v``
      scaled by the product of all factors ``f_j`` such that ``b_j <= b_i``. The
      values in between each boundary will be interpolated as per ``type``.

  Returns:
    schedule
      A function that maps step counts to values.
  """
  if interpolate_type == 'linear':
    interpolate_fn = _linear_interpolate
  elif interpolate_type == 'cosine':
    interpolate_fn = _cosine_interpolate
  else:
    raise ValueError("`interpolate_type` must be either 'cos' or 'linear'")

  if boundaries_and_scales:
    boundaries, scales = zip(*sorted(boundaries_and_scales.items()))
    if not all(scale >= 0.0 for scale in scales):
      raise ValueError(
          '`piecewise_interpolate_schedule` expects non-negative scale factors'
      )
  else:
    boundaries, scales = (), ()

  bounds = np.stack((0,) + boundaries)
  values = np.cumprod(np.stack((init_value,) + scales))
  interval_sizes = bounds[1:] - bounds[:-1]

  def schedule(count):
    indicator = (bounds[:-1] <= count) & (count < bounds[1:])
    pct = (count - bounds[:-1]) / interval_sizes
    interp_vals = interpolate_fn(values[:-1], values[1:], pct)
    return indicator.dot(interp_vals) + (bounds[-1] <= count) * values[-1]

  return schedule


def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> base.Schedule:
  r"""Returns a learning rate with three linear phases.

  * *Phase 1*, from iteration 0 to ``pct_start * transition_steps``. The
    learning rate increases linearly from ``peak_value / div_factor`` to
    ``peak_value``.
  * *Phase 2*, from iteration ``pct_start * transition_steps`` to
    ``pct_final * transition_steps``. The learning rate decreases linearly from
    ``peak_value`` back to the initial ``peak_value/div_factor``.
  * *Phase 3*: For the remaining steps, the learning rate interpolates between
    ``peak_value/div_factor`` and ``peak_value / final_div_factor``. If
    ``final_div_factor`` is larger than ``div_factor``, this is a decreasing
    phase.

  Args:
    transition_steps: Number of steps over which annealing takes place.
    peak_value: Maximum value attained by schedule at pct_start percent of the
      cycle (in number of steps).
    pct_start: The percentage of the cycle (in number of steps) spent increasing
      the learning rate.
    pct_final: The percentage of the cycle (in number of steps) spent increasing
      to ``peak_value`` then decreasing back to ``init_value``.
    div_factor: Determines the initial value via ``init_value = peak_value /
      div_factor``.
    final_div_factor: Determines the final value via ``final_value = init_value
      / final_div_factor``.

  Returns:
    schedule
      A function that maps step counts to values

  References:
    Smith et al, `Super-Convergence: Very Fast Training of Neural Networks Using
    Large Learning Rates <https://arxiv.org/abs/1708.07120>`_, 2017
  """
  if transition_steps <= 0:
    raise ValueError(
        'A linear onecycle schedule was set with a non-positive '
        '`transition_steps`'
    )

  return piecewise_interpolate_schedule(
      'linear',
      peak_value / div_factor,
      {
          int(pct_start * transition_steps): div_factor,
          int(pct_final * transition_steps): 1.0 / div_factor,
          transition_steps: 1.0 / final_div_factor,
      },
  )


def cosine_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> base.Schedule:
  """Returns a function which implements the onecycle learning rate schedule.

  This learning rate increases the learning rate and then decreases it in a
  cosine-like manner. The number of steps over which the learning rate increases
  is determined by the ``pct_start`` argument. The maximum value of the learning
  rate is determined by the ``peak_value`` argument, the initial value of the
  learning rate is determined through the formula ``init_value = peak_value /
  div_factor``, and the final value is determined by the ``final_div_factor``
  argument.

  Args:
    transition_steps: Number of steps over which annealing takes place.
    peak_value: Maximum value attained by schedule at pct_start percent of the
      cycle (in number of steps).
    pct_start: The percentage of the cycle (in number of steps) spent increasing
      the learning rate.
    div_factor: Determines the initial value via ``init_value = peak_value /
      div_factor``.
    final_div_factor: Determines the final value via ``final_value = init_value
      / final_div_factor``.

  Returns:
    schedule
      A function that maps step counts to values

  References:
    Smith et al, `Super-Convergence: Very Fast Training of Neural Networks Using
    Large Learning Rates <https://arxiv.org/abs/1708.07120>`_, 2017
  """
  if transition_steps <= 0:
    raise ValueError(
        'A linear onecycle schedule was set with a non-positive '
        '`transition_steps`'
    )

  return piecewise_interpolate_schedule(
      'cosine',
      peak_value / div_factor,
      {
          int(pct_start * transition_steps): div_factor,
          int(transition_steps): 1.0 / (div_factor * final_div_factor),
      },
  )


def warmup_constant_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
) -> base.Schedule:
  r"""Linear warmup followed by constant schedule i.e no decay.

  Args:
    init_value: Initial value for the scalar to be annealed.
    peak_value: Peak value for scalar to be annealed at end of warmup.
    warmup_steps: Positive integer, the length of the linear warmup.

  Returns:
    schedule
      A function that maps step counts to values
  """
  return linear_schedule(
      init_value=init_value,
      end_value=peak_value,
      transition_steps=warmup_steps,
  )


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
    exponent: float = 1.0,
) -> base.Schedule:
  r"""Linear warmup followed by cosine decay.

  Args:
    init_value: Initial value for the scalar to be annealed.
    peak_value: Peak value for scalar to be annealed at end of warmup.
    warmup_steps: Positive integer, the length of the linear warmup.
    decay_steps: Positive integer, the total length of the schedule. Note that
      this includes the warmup time, so the number of steps during which cosine
      annealing is applied is ``decay_steps - warmup_steps``.
    end_value: End value of the scalar to be annealed.
    exponent: The default decay is ``0.5 * (1 + cos(pi t/T))``, where ``t`` is
      the current timestep and ``T`` is ``decay_steps``. The exponent modifies
      this to be ``(0.5 * (1 + cos(pi * t/T))) ** exponent``. Defaults to 1.0.

  Returns:
    schedule
      A function that maps step counts to values
  """
  alpha = 0.0 if peak_value == 0.0 else end_value / peak_value
  schedules = [
      linear_schedule(
          init_value=init_value,
          end_value=peak_value,
          transition_steps=warmup_steps,
      ),
      cosine_decay_schedule(
          init_value=peak_value,
          decay_steps=decay_steps - warmup_steps,
          alpha=alpha,
          exponent=exponent,
      ),
  ]
  return _join.join_schedules(schedules, [warmup_steps])


def warmup_exponential_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> base.Schedule:
  """Linear warmup followed by exponential decay.

  Args:
    init_value: Initial value for the scalar to be annealed.
    peak_value: Peak value for scalar to be annealed at end of warmup.
    warmup_steps: Positive integer, the length of the linear warmup.
    transition_steps: must be positive. See :func:`optax.exponential_decay` for
      more details.
    decay_rate: must not be zero. The decay rate.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at ``peak_value``).
    staircase: if ``True``, decay the values at discrete intervals.
    end_value: the value at which the exponential decay stops. When ``decay_rate
      < 1``, ``end_value`` is treated as a lower bound, otherwise as an upper
      bound. Has no effect when ``decay_rate = 0``.

  Returns:
    schedule
      A function that maps step counts to values
  """
  schedules = [
      linear_schedule(
          init_value=init_value,
          end_value=peak_value,
          transition_steps=warmup_steps,
      ),
      exponential_decay(
          init_value=peak_value,
          transition_steps=transition_steps,
          decay_rate=decay_rate,
          transition_begin=transition_begin,
          staircase=staircase,
          end_value=end_value,
      ),
  ]
  return _join.join_schedules(schedules, [warmup_steps])


def sgdr_schedule(
    cosine_kwargs: Iterable[dict[str, chex.Numeric]],
) -> base.Schedule:
  """SGD with warm restarts.

  This learning rate schedule applies multiple joined cosine decay cycles.

  Args:
    cosine_kwargs: An Iterable of dicts, where each element specifies the
      arguments to pass to each cosine decay cycle. The ``decay_steps`` kwarg
      will specify how long each cycle lasts for, and therefore when to
      transition to the next cycle.

  Returns:
    schedule
      A function that maps step counts to values

  References:
    Loshchilov et al., `SGDR: Stochastic Gradient Descent with Warm Restarts
    <https://arxiv.org/abs/1608.03983>`_, 2017
  """
  boundaries = []
  schedules = []
  step = 0
  for kwargs in cosine_kwargs:
    schedules += [warmup_cosine_decay_schedule(**kwargs)]
    boundaries += [step + kwargs['decay_steps']]
    step += kwargs['decay_steps']
  return _join.join_schedules(schedules, boundaries[:-1])
