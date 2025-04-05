# Copyright 2024 The Orbax Authors.
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

"""Defines policies for when a checkpoint is saved."""

import dataclasses
import datetime
import typing
from typing import Container, Protocol, Sequence
from orbax.checkpoint._src.metadata import checkpoint_info


@dataclasses.dataclass(kw_only=True)
class DecisionContext:
  """Additional properties for making a save decision."""

  is_saving_in_progress: bool
  reached_preemption: bool


@typing.runtime_checkable
class SaveDecisionPolicy(Protocol):
  """A policy that defines when to save a checkpoint.

  Implementations should return True from `should_save` when saving a checkpoint
  is desired at the given step.
  """

  def should_save(
      self,
      step: checkpoint_info.CheckpointInfo,
      previous_steps: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: DecisionContext
  ) -> bool:
    ...


@dataclasses.dataclass
class FixedIntervalPolicy(SaveDecisionPolicy):
  """Checkpoint at a fixed interval."""

  interval: int

  def should_save(
      self,
      step: checkpoint_info.CheckpointInfo,
      previous_steps: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: DecisionContext
  ) -> bool:
    del previous_steps
    del context
    return step.step % self.interval == 0


@dataclasses.dataclass
class SpecificStepsPolicy(SaveDecisionPolicy):
  """Checkpoint at specific steps."""

  steps: Container[int]

  def should_save(
      self,
      step: checkpoint_info.CheckpointInfo,
      previous_steps: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: DecisionContext
  ) -> bool:
    del previous_steps
    del context
    return step.step in self.steps


@dataclasses.dataclass(kw_only=True)
class ContinuousCheckpointingPolicy(SaveDecisionPolicy):
  """Checkpoint as often as possible, as long as a save is not ongoing."""

  minimum_interval_secs: int | None = None

  def should_save(
      self,
      step: checkpoint_info.CheckpointInfo,
      previous_steps: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: DecisionContext
  ) -> bool:
    if context.is_saving_in_progress:
      return False
    if not previous_steps or self.minimum_interval_secs is None:
      return True
    return step.time - previous_steps[-1].time >= datetime.timedelta(
        seconds=self.minimum_interval_secs
    )


class PreemptionCheckpointingPolicy(SaveDecisionPolicy):
  """Save a checkpoint when a preemption is detected."""

  def should_save(
      self,
      step: checkpoint_info.CheckpointInfo,
      previous_steps: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: DecisionContext
  ) -> bool:
    del step
    del previous_steps
    return context.reached_preemption


class InitialSavePolicy(SaveDecisionPolicy):
  """Checkpoint as soon as possible if no checkpoints already exist."""

  def should_save(
      self,
      step: checkpoint_info.CheckpointInfo,
      previous_steps: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: DecisionContext
  ) -> bool:
    del step
    del context
    return not previous_steps


@dataclasses.dataclass
class AnySavePolicy(SaveDecisionPolicy):
  """Evaluates all policies and saves if any of them returns True.

  Each policy is evaluated in order, and if all are False, the final result is
  False. If at least one is True, the final result is True.
  """

  policies: Sequence[SaveDecisionPolicy]

  def should_save(
      self,
      step: checkpoint_info.CheckpointInfo,
      previous_steps: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: DecisionContext
  ) -> bool:
    return any(
        policy.should_save(step, previous_steps=previous_steps, context=context)
        for policy in self.policies
    )
