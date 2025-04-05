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

"""Internal IO utilities for metadata of a checkpoint at root level."""
import dataclasses

from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
RootMetadata = checkpoint.RootMetadata


def serialize(metadata: RootMetadata) -> SerializedMetadata:
  """Serializes `metadata` to a dictionary."""
  serialized_fields = {
      field.name: field.metadata['processor'](getattr(metadata, field.name))
      for field in dataclasses.fields(metadata)
  }

  return serialized_fields


def deserialize(metadata_dict: SerializedMetadata) -> RootMetadata:
  """Deserializes `metadata_dict` to `RootMetadata`."""
  fields = dataclasses.fields(RootMetadata)
  field_names = {field.name for field in fields}

  field_processor_args = {
      field_name: (metadata_dict.get(field_name, None),)
      for field_name in field_names
  }

  validated_metadata_dict = {
      field.name: field.metadata['processor'](*field_processor_args[field.name])
      for field in fields
  }

  for k in metadata_dict:
    if k not in validated_metadata_dict:
      validated_metadata_dict['custom_metadata'][k] = utils.process_unknown_key(
          k, metadata_dict
      )

  return RootMetadata(**validated_metadata_dict)
