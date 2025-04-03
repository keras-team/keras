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

"""Early stopping."""

import math

from flax import struct


class EarlyStopping(struct.PyTreeNode):
  """Early stopping to avoid overfitting during training.

  The following example stops training early if the difference between losses
  recorded in the current epoch and previous epoch is less than 1e-3
  consecutively for 2 times::

    >>> from flax.training.early_stopping import EarlyStopping

    >>> def train_epoch(optimizer, train_ds, batch_size, epoch, input_rng):
    ...   ...
    ...   loss = [4, 3, 3, 3, 2, 2, 2, 2, 1, 1][epoch]
    ...   return None, {'loss': loss}

    >>> early_stop = EarlyStopping(min_delta=1e-3, patience=2)
    >>> optimizer = None
    >>> for epoch in range(10):
    ...   optimizer, train_metrics = train_epoch(
    ...       optimizer=optimizer, train_ds=None, batch_size=None, epoch=epoch, input_rng=None)
    ...   early_stop = early_stop.update(train_metrics['loss'])
    ...   if early_stop.should_stop:
    ...     print(f'Met early stopping criteria, breaking at epoch {epoch}')
    ...     break
    Met early stopping criteria, breaking at epoch 7

  Attributes:
    min_delta: Minimum delta between updates to be considered an
        improvement.
    patience: Number of steps of no improvement before stopping.
    best_metric: Current best metric value.
    patience_count: Number of steps since last improving update.
    should_stop: Whether the training loop should stop to avoid
        overfitting.
    has_improved: Whether the metric has improved greater or
      equal to the min_delta in the last ``.update`` call.
  """

  min_delta: float = 0
  patience: int = 0
  best_metric: float = float('inf')
  patience_count: int = 0
  should_stop: bool = False
  has_improved: bool = False

  def reset(self):
    return self.replace(
      best_metric=float('inf'),
      patience_count=0,
      should_stop=False,
      has_improved=False,
    )

  def update(self, metric):
    """Update the state based on metric.

    Returns:
      The updated EarlyStopping class. The ``.has_improved`` attribute is True
      when there was an improvement greater than ``min_delta`` from the previous
      ``best_metric``.
    """

    if (
      math.isinf(self.best_metric) or self.best_metric - metric > self.min_delta
    ):
      return self.replace(
        best_metric=metric, patience_count=0, has_improved=True
      )
    else:
      should_stop = self.patience_count >= self.patience or self.should_stop
      return self.replace(
        patience_count=self.patience_count + 1,
        should_stop=should_stop,
        has_improved=False,
      )
