# Copyright 2024 The Flax Authors.
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

"""Learning rate schedules used in FLAX image classification examples.

Note that with `FLIP #1009`_ learning rate schedules in ``flax.training`` are
**effectively deprecated** in favor of Optax_ schedules. Please refer to
`Optimizer Schedules`_ for more information.

.. _FLIP #1009: https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md
.. _Optax: https://github.com/deepmind/optax
.. _Optimizer Schedules: https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules
"""

import jax.numpy as jnp
import numpy as np
from absl import logging


def _piecewise_constant(boundaries, values, t):
  index = jnp.sum(boundaries < t)
  return jnp.take(values, index)


def create_constant_learning_rate_schedule(
  base_learning_rate, steps_per_epoch, warmup_length=0.0
):
  """Create a constant learning rate schedule with optional warmup.

  Note that with `FLIP #1009`_ learning rate schedules in ``flax.training`` are
  **effectively deprecated** in favor of Optax_ schedules. Please refer to
  `Optimizer Schedules`_ for more information.

  .. _FLIP #1009: https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md
  .. _Optax: https://github.com/deepmind/optax
  .. _Optimizer Schedules: https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules

  Holds the learning rate constant. This function also offers a learing rate
  warmup as per https://arxiv.org/abs/1706.02677, for the purpose of training
  with large mini-batches.

  Args:
    base_learning_rate: the base learning rate
    steps_per_epoch: the number of iterations per epoch
    warmup_length: if > 0, the learning rate will be modulated by a warmup
      factor that will linearly ramp-up from 0 to 1 over the first
      ``warmup_length`` epochs

  Returns:
    Function ``f(step) -> lr`` that computes the learning rate for a given step.
  """
  logging.warning(
    'Learning rate schedules in ``flax.training`` are effectively deprecated '
    'in favor of Optax schedules. Please refer to '
    'https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules'
    ' for alternatives.'
  )

  def learning_rate_fn(step):
    lr = base_learning_rate
    if warmup_length > 0.0:
      lr = lr * jnp.minimum(1.0, step / float(warmup_length) / steps_per_epoch)
    return lr

  return learning_rate_fn


def create_stepped_learning_rate_schedule(
  base_learning_rate, steps_per_epoch, lr_sched_steps, warmup_length=0.0
):
  """Create a stepped learning rate schedule with optional warmup.

  Note that with `FLIP #1009`_ learning rate schedules in ``flax.training`` are
  **effectively deprecated** in favor of Optax_ schedules. Please refer to
  `Optimizer Schedules`_ for more information.

  .. _FLIP #1009: https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md
  .. _Optax: https://github.com/deepmind/optax
  .. _Optimizer Schedules: https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules

  A stepped learning rate schedule decreases the learning rate
  by specified amounts at specified epochs. The steps are given as
  the ``lr_sched_steps`` parameter. A common ImageNet schedule decays the
  learning rate by a factor of 0.1 at epochs 30, 60 and 80. This would be
  specified as::

    [
      [30, 0.1],
      [60, 0.01],
      [80, 0.001]
    ]

  This function also offers a learing rate warmup as per
  https://arxiv.org/abs/1706.02677, for the purpose of training with large
  mini-batches.

  Args:
    base_learning_rate: the base learning rate
    steps_per_epoch: the number of iterations per epoch
    lr_sched_steps: the schedule as a list of steps, each of which is
      a ``[epoch, lr_factor]`` pair; the step occurs at epoch ``epoch`` and
      sets the learning rate to ``base_learning_rage * lr_factor``
    warmup_length: if > 0, the learning rate will be modulated by a warmup
      factor that will linearly ramp-up from 0 to 1 over the first
      ``warmup_length`` epochs

  Returns:
    Function ``f(step) -> lr`` that computes the learning rate for a given step.
  """
  logging.warning(
    'Learning rate schedules in ``flax.training`` are effectively deprecated '
    'in favor of Optax schedules. Please refer to '
    'https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules'
    ' for alternatives.'
  )
  boundaries = [step[0] for step in lr_sched_steps]
  decays = [step[1] for step in lr_sched_steps]
  boundaries = np.array(boundaries) * steps_per_epoch
  boundaries = np.round(boundaries).astype(int)
  values = np.array([1.0] + decays) * base_learning_rate

  def learning_rate_fn(step):
    lr = _piecewise_constant(boundaries, values, step)
    if warmup_length > 0.0:
      lr = lr * jnp.minimum(1.0, step / float(warmup_length) / steps_per_epoch)
    return lr

  return learning_rate_fn


def create_cosine_learning_rate_schedule(
  base_learning_rate, steps_per_epoch, halfcos_epochs, warmup_length=0.0
):
  """Create a cosine learning rate schedule with optional warmup.

  Note that with `FLIP #1009`_ learning rate schedules in ``flax.training`` are
  **effectively deprecated** in favor of Optax_ schedules. Please refer to
  `Optimizer Schedules`_ for more information.

  .. _FLIP #1009: https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md
  .. _Optax: https://github.com/deepmind/optax
  .. _Optimizer Schedules: https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules

  A cosine learning rate schedule modules the learning rate with
  half a cosine wave, gradually scaling it to 0 at the end of training.

  This function also offers a learing rate warmup as per
  https://arxiv.org/abs/1706.02677, for the purpose of training with large
  mini-batches.

  Args:
    base_learning_rate: the base learning rate
    steps_per_epoch: the number of iterations per epoch
    halfcos_epochs: the number of epochs to complete half a cosine wave;
      normally the number of epochs used for training
    warmup_length: if > 0, the learning rate will be modulated by a warmup
      factor that will linearly ramp-up from 0 to 1 over the first
      ``warmup_length`` epochs

  Returns:
    Function ``f(step) -> lr`` that computes the learning rate for a given step.
  """
  logging.warning(
    'Learning rate schedules in ``flax.training`` are effectively deprecated '
    'in favor of Optax schedules. Please refer to '
    'https://optax.readthedocs.io/en/latest/api.html#optimizer-schedules'
    ' for alternatives.'
  )
  halfwavelength_steps = halfcos_epochs * steps_per_epoch

  def learning_rate_fn(step):
    scale_factor = jnp.cos(step * jnp.pi / halfwavelength_steps) * 0.5 + 0.5
    lr = base_learning_rate * scale_factor
    if warmup_length > 0.0:
      lr = lr * jnp.minimum(1.0, step / float(warmup_length) / steps_per_epoch)
    return lr

  return learning_rate_fn
