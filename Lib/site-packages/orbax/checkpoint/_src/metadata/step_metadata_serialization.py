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

"""Internal IO utilities for metadata of a checkpoint at step level."""

import dataclasses
from typing import Any

from etils import epath
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
StepMetadata = checkpoint.StepMetadata
CompositeItemMetadata = checkpoint.CompositeItemMetadata
SingleItemMetadata = checkpoint.SingleItemMetadata
StepStatistics = step_statistics.SaveStepStatistics


def get_step_metadata(path: epath.PathLike) -> StepMetadata:
  """Returns StepMetadata for a given checkpoint directory."""
  metadata_file_path = checkpoint.step_metadata_file_path(path)
  serialized_metadata = checkpoint.metadata_store(enable_write=False).read(
      metadata_file_path
  )
  if serialized_metadata is None:
    raise ValueError(f'Step metadata not found at {metadata_file_path}')
  return deserialize(serialized_metadata)


def serialize(metadata: StepMetadata) -> SerializedMetadata:
  """Serializes `metadata` to a dictionary."""
  serialized_fields = {
      field.name: field.metadata['processor'](getattr(metadata, field.name))
      for field in dataclasses.fields(metadata)
  }

  # Per item metadata is already saved in the item subdirectories.
  del serialized_fields['item_metadata']

  return serialized_fields


def serialize_for_update(**kwargs) -> SerializedMetadata:
  """Validates and serializes `kwargs` to a dictionary.

  To be used with MetadataStore.update().

  Args:
    **kwargs: The kwargs to be serialized.

  Returns:
    A dictionary of the serialized kwargs.
  """
  fields = dataclasses.fields(StepMetadata)
  field_names = {field.name for field in fields}

  for k in kwargs:
    if k not in field_names:
      raise ValueError('Provided metadata contains unknown key %s.' % k)

  validated_kwargs = {
      field.name: field.metadata['processor'](kwargs[field.name])
      for field in fields
      if field.name in kwargs
  }

  return validated_kwargs


def deserialize(
    metadata_dict: SerializedMetadata,
    item_metadata: CompositeItemMetadata | SingleItemMetadata | None = None,
    metrics: dict[str, Any] | None = None,
) -> StepMetadata:
  """Deserializes `metadata_dict` and other kwargs to `StepMetadata`."""
  fields = dataclasses.fields(StepMetadata)
  field_names = {field.name for field in fields}

  field_processor_args = {
      field_name: (metadata_dict.get(field_name, None),)
      for field_name in field_names
  }
  field_processor_args['item_metadata'] = (item_metadata,)
  field_processor_args['metrics'] = (
      metadata_dict.get('metrics', None),
      metrics,
  )

  validated_metadata_dict = {
      field.name: field.metadata['processor'](*field_processor_args[field.name])
      for field in fields
  }

  validated_metadata_dict['performance_metrics'] = StepStatistics(
      **validated_metadata_dict['performance_metrics']
  )

  assert isinstance(validated_metadata_dict['custom_metadata'], dict)
  for k in metadata_dict:
    if k not in validated_metadata_dict:
      validated_metadata_dict['custom_metadata'][k] = utils.process_unknown_key(
          k, metadata_dict
      )

  return StepMetadata(**validated_metadata_dict)
