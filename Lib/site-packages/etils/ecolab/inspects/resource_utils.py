# Copyright 2024 The etils Authors.
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

"""Resource utils."""

import functools
from typing import Optional

from etils import epath


def _static_path() -> epath.Path:
  """Path to the static resources (`.js`, `.css`)."""
  return epath.resource_path('etils.ecolab') / 'inspects' / 'static'


# TODO(epot): Use gstatic to serve those files.
# TODO(epot): Expose in public API ?
@functools.lru_cache()
def resource_import(
    filename: str,
    *,
    module: Optional[epath.PathLike] = None,
) -> str:
  """Returns the `HTML` associated with the resource.

  Args:
    filename: Path to the `.css`, `.js` resource
    module: Python module name from which the filename is relative too.
  """
  path = epath.resource_path(module) if module else _static_path()
  path = path.joinpath(filename)
  content = path.read_text()
  if path.suffix == '.css':
    return f'<style>{content}</style>'
  elif path.suffix == '.js':
    return f'<script>{content}</script>'
  else:
    raise ValueError('')
