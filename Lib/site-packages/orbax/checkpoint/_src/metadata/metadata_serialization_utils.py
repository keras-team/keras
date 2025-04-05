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

"""Utilities for serializing and deserializing metadata."""

import dataclasses
from typing import Any, Mapping, Optional, Sequence, TypeAlias

from absl import logging
from orbax.checkpoint._src import composite
from orbax.checkpoint._src.logging import step_statistics


CompositeCheckpointHandlerTypeStrs: TypeAlias = Mapping
CheckpointHandlerTypeStr = str
CompositeItemMetadata = composite.Composite
SingleItemMetadata = Any
StepStatistics = step_statistics.SaveStepStatistics


def _validate_type(obj: Any, field_type: type[Any] | Sequence[type[Any]]):
  if isinstance(field_type, Sequence):
    if not any(isinstance(obj, f_type) for f_type in field_type):
      raise ValueError(
          f'Object must be any one of types {list(field_type)}, got '
          f'{type(obj)}.'
      )
  elif not isinstance(obj, field_type):
    raise ValueError(f'Object must be of type {field_type}, got {type(obj)}.')


def validate_and_process_item_handlers(
    item_handlers: Any,
) -> (
    CompositeCheckpointHandlerTypeStrs[str, Any]
    | CheckpointHandlerTypeStr
    | None
):
  """Validates and processes item_handlers field."""
  if item_handlers is None:
    return None

  _validate_type(item_handlers, [dict, str])
  if isinstance(item_handlers, CompositeCheckpointHandlerTypeStrs):
    for k in item_handlers or {}:
      _validate_type(k, str)
    return item_handlers
  elif isinstance(item_handlers, CheckpointHandlerTypeStr):
    return item_handlers


def validate_and_process_item_metadata(
    item_metadata: Any,
) -> CompositeItemMetadata | SingleItemMetadata | None:
  """Validates and processes item_metadata field."""
  if item_metadata is None:
    return None

  if isinstance(item_metadata, CompositeItemMetadata):
    for k in item_metadata:
      _validate_type(k, str)
    return item_metadata
  else:
    return item_metadata


def validate_and_process_metrics(
    metrics: Any, additional_metrics: Optional[Any] = None
) -> dict[str, Any]:
  """Validates and processes metrics field."""
  metrics = metrics or {}

  _validate_type(metrics, dict)
  for k in metrics:
    _validate_type(k, str)
  validated_metrics = metrics.copy()

  if additional_metrics is not None:
    _validate_type(additional_metrics, dict)
    for k, v in additional_metrics.items():
      _validate_type(k, str)
      validated_metrics[k] = v

  return validated_metrics


def validate_and_process_performance_metrics(
    performance_metrics: Any,
) -> dict[str, float]:
  """Validates and processes performance_metrics field."""
  if performance_metrics is None:
    return {}

  _validate_type(performance_metrics, [dict, StepStatistics])
  if isinstance(performance_metrics, StepStatistics):
    performance_metrics = dataclasses.asdict(performance_metrics)

  for k in performance_metrics:
    _validate_type(k, str)

  return {
      metric: val
      for metric, val in performance_metrics.items()
      if isinstance(val, float)
  }


def validate_and_process_init_timestamp_nsecs(
    init_timestamp_nsecs: Any,
) -> int | None:
  """Validates and processes init_timestamp_nsecs field."""
  if init_timestamp_nsecs is None:
    return None

  _validate_type(init_timestamp_nsecs, int)
  return init_timestamp_nsecs


def validate_and_process_commit_timestamp_nsecs(
    commit_timestamp_nsecs: Any,
) -> int | None:
  """Validates and processes commit_timestamp_nsecs field."""
  if commit_timestamp_nsecs is None:
    return None

  _validate_type(commit_timestamp_nsecs, int)
  return commit_timestamp_nsecs


def validate_and_process_custom_metadata(
    custom_metadata: Any,
) -> dict[str, Any]:
  """Validates and processes custom field."""
  if custom_metadata is None:
    return {}

  _validate_type(custom_metadata, dict)
  for k in custom_metadata:
    _validate_type(k, str)
  return custom_metadata


def process_unknown_key(key: str, metadata_dict: dict[str, Any]) -> Any:
  if 'custom_metadata' in metadata_dict and metadata_dict['custom_metadata']:
    raise ValueError(
        'Provided metadata contains unknown key %s, and the custom_metadata'
        ' field is already defined.' % key
    )
  logging.warning(
      'Provided metadata contains unknown key %s. Adding it to'
      ' custom_metadata.',
      key,
  )
  return metadata_dict[key]
