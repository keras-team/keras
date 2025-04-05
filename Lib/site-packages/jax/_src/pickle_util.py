# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
import io
from typing import Any

try:
  import cloudpickle  # type: ignore[import-not-found]
except ImportError:
  cloudpickle = None

from jax._src import profiler


@functools.partial(profiler.annotate_function, name='pickle_util.dumps')
def dumps(obj: Any) -> bytes:
  """See `pickle.dumps`. Used for serializing host callbacks in jaxlib."""
  if cloudpickle is None:
    raise ModuleNotFoundError('No module named "cloudpickle"')

  class Pickler(cloudpickle.CloudPickler):
    """Customizes the behavior of cloudpickle."""

    # Make a copy to avoid modifying cloudpickle for other users.
    dispatch_table = cloudpickle.CloudPickler.dispatch_table.copy()

    # Fixes for dataclass internal singleton object serialization.
    # Bug: https://github.com/cloudpipe/cloudpickle/issues/386
    # pylint: disable=protected-access
    # pytype: disable=module-attr
    dispatch_table[dataclasses._FIELD_BASE] = lambda x: f'{x.name}'
    dispatch_table[dataclasses._MISSING_TYPE] = lambda _: 'MISSING'
    dispatch_table[dataclasses._HAS_DEFAULT_FACTORY_CLASS] = (
        lambda _: '_HAS_DEFAULT_FACTORY'
    )
    if hasattr(dataclasses, '_KW_ONLY_TYPE'):
      dispatch_table[dataclasses._KW_ONLY_TYPE] = (
          lambda _: '_KW_ONLY_TYPE'
      )  # Added in Python 3.10.
    # pytype: enable=module-attr
    # pylint: enable=protected-access

  with io.BytesIO() as file:
    Pickler(file).dump(obj)
    return file.getvalue()


@functools.partial(profiler.annotate_function, name='pickle_util.loads')
def loads(data: bytes) -> Any:
  """See `pickle.loads`."""
  if cloudpickle is None:
    raise ModuleNotFoundError('No module named "cloudpickle"')

  return cloudpickle.loads(data)
