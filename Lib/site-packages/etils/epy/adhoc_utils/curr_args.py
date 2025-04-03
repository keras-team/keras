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

"""Utils to record the current adhoc arguments.

Required for mixing `epy.lazy_imports()` with adhoc imports, like:

```
with ecolab.adhoc():
  with epy.lazy_imports():
    import xxx


xxx.__version__  # < Resolving the lazy-import will re-use the adhoc scope.
```
"""

import contextlib
import enum
from typing import Any, Iterator

# Store the current adhoc kwargs (for future re-use)
_CURR_ADHOC_KWARGS: dict[str, Any] | None = None


class Scope(enum.Enum):
  """Scope of the current adhoc kwargs."""

  COLAB = enum.auto()  # ecolab.adhoc() scope
  BINARY = enum.auto()  # epy.binary_adhoc() scope


@contextlib.contextmanager
def set_curr_adhoc_kwargs(
    adhoc_kwargs: dict[str, Any],
    *,
    scope: Scope,
) -> Iterator[None]:
  """Set the current adhoc kwargs (accessed by `epy.lazy_imports()`)."""
  global _CURR_ADHOC_KWARGS
  try:
    _CURR_ADHOC_KWARGS = dict(adhoc_kwargs) | {'__scope__': scope}
    yield
  finally:
    _CURR_ADHOC_KWARGS = None


def get_curr_adhoc_kwargs() -> dict[str, Any] | None:
  if _CURR_ADHOC_KWARGS is None:
    return None
  else:
    return dict(_CURR_ADHOC_KWARGS)


def replay_adhoc_ctx(**adhoc_kwargs: Any):
  """Replay the adhoc context."""

  scope = adhoc_kwargs.pop('__scope__')

  match scope:
    case Scope.COLAB:
      from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

      return ecolab.adhoc(**adhoc_kwargs)
    case Scope.BINARY:
      # Added by LazyModule but not supported by binary_adhoc
      adhoc_kwargs.pop('collapse_prefix')

      from etils.epy.adhoc_utils import binary_import  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

      return binary_import.binary_adhoc(**adhoc_kwargs)
    case _:
      raise ValueError(f'Unknown scope: {scope}')
