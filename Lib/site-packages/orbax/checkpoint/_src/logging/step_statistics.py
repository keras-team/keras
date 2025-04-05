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

"""Step statistics."""

import dataclasses
from typing import Optional

# Note:`*_end_time` fields are not required here as it can be calculated by
# *_start_time + *_duration_secs


@dataclasses.dataclass
class SaveStepStatistics:
  """Attributes for save step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    directory: The directory of the checkpoint.
    reached_preemption: Whether the event reached preemption.
    preemption_received_at: The time when preemption was received.
    synchronous: Whether the event is synchronous.
    wait_for_prev_start_time: The start time of waiting for previous checkpoint.
    wait_for_prev_duration_secs: The duration of waiting for previous
      checkpoint.
    checkpointer_blocking_start_time: The start time of blocking time introduced
      by checkpointer.
    checkpointer_blocking_duration_secs: The duration of blocking time
      introduced by checkpointer.
    get_old_steps_start_time: The start time of getting old steps.
    get_old_steps_duration_secs: The duration of getting old steps.
    checkpoint_manager_blocking_start_time: The start time of checkpoint manager
      blocking section.
    checkpoint_manager_blocking_duration_secs: The duration of checkpoint
      manager blocking section.
  """

  step: Optional[int] = None
  event_type: Optional[str] = "save"
  directory: Optional[str] = None
  reached_preemption: Optional[bool] = False
  preemption_received_at: Optional[float] = None
  synchronous: Optional[bool] = False
  wait_for_prev_start_time: Optional[float] = None
  wait_for_prev_duration_secs: Optional[float] = None
  checkpointer_blocking_start_time: Optional[float] = None
  checkpointer_blocking_duration_secs: Optional[float] = None
  get_old_steps_start_time: Optional[float] = None
  get_old_steps_duration_secs: Optional[float] = None
  checkpoint_manager_blocking_start_time: Optional[float] = None
  checkpoint_manager_blocking_duration_secs: Optional[float] = None


@dataclasses.dataclass
class RestoreStepStatistics:
  """Attributes for restore step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    directory: The directory of the checkpoint.
    checkpointer_start_time: The start time of restoring the checkpoint, while
      using the checkpointer.
    checkpointer_duration_secs: The total duration for restoring the checkpoint,
      while using the checkpointer.
    checkpoint_manager_start_time: The start time for restoring the checkpoint,
      while using the checkpoint manager.
    checkpoint_manager_duration_secs: The total duration for restoring the
      checkpoint, while using the checkpoint manager.
  """

  step: Optional[int] = None
  event_type: Optional[str] = "restore"
  directory: Optional[str] = None
  checkpointer_start_time: Optional[float] = None
  checkpointer_duration_secs: Optional[float] = None
  checkpoint_manager_start_time: Optional[float] = None
  checkpoint_manager_duration_secs: Optional[float] = None


@dataclasses.dataclass
class EmergencyRestoreStepStatistics:
  """Attributes for emergency restore step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    checkpoint_manager_start_time: The start time of checkpoint manager
      restore event.
    directory: The directory of the checkpoint.
    is_restoring_slice: Whether the event takes place on the slice responsible
      for reading from the storage location. (Note that in_primary_slice=True
      necessarily implies is_restoring_slice=True.)
    in_primary_slice: Whether the event takes place on the slice designated as
      primary (responsible for restoring from persistent storage).
    checkpointer_start_time: The start time of restoring the checkpoint, while
      using the checkpointer.
    checkpointer_duration_secs: The total duration for restoring the checkpoint,
      while using the checkpointer.
    broadcast_start_time: The start time of broadcasting(Restore).The broadcast
      operation performed by SingleReplicaArrayHandler won't be captured in this
      context.
    broadcast_duration_secs: The duration of broadcasting(Restore).
    checkpoint_manager_duration_secs: The total duration of checkpoint
      manager restore event.
  """

  step: Optional[int] = None
  event_type: Optional[str] = "emergency_restore"
  checkpoint_manager_start_time: Optional[float] = None
  directory: Optional[str] = None
  is_restoring_slice: Optional[bool] = False
  in_primary_slice: Optional[bool] = False
  checkpointer_start_time: Optional[float] = None
  checkpointer_duration_secs: Optional[float] = None
  broadcast_start_time: Optional[float] = None
  broadcast_duration_secs: Optional[float] = None
  checkpoint_manager_duration_secs: Optional[float] = None
