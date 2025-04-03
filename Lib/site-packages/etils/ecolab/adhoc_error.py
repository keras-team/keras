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

"""Better error for adhoc reload."""

import functools

import IPython


@functools.cache
def register_better_reload_error() -> None:
  ip = IPython.get_ipython()

  if ip is None:  # In tests
    return

  # What if this conflict with other `ip.set_custom_exc` ?
  # Ideally, should support multiple handlers
  ip.set_custom_exc((NameError,), _maybe_better_error)


def _maybe_better_error(self, type_, value, traceback, tb_offset=None):
  """Update the error message."""

  if (
      type(value) is NameError  # pylint: disable=unidiomatic-typecheck
      and len(value.args) == 1
      and _is_from_invalidate_module(value)
  ):
    (msg,) = value.args
    value.args = tuple([
        msg
        + "\nYou're trying to use an object created with an old version of a"
        ' module you reloaded. Please re-create the object with the reloaded'
        ' module.'
    ])
  self.showtraceback(
      (type_, value, traceback),
      tb_offset=tb_offset,
  )


def _is_from_invalidate_module(exc: Exception) -> bool:
  """Check whether the exception is from an invalidated module."""
  tb = exc.__traceback__
  while tb is not None:
    frame = tb.tb_frame
    if '__etils_invalidated__' in frame.f_globals:
      return True
    tb = tb.tb_next

  return False
