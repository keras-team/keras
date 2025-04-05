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

"""Dummy implementation of the `exm` API to simplify open-sourcing."""

import functools
from typing import TypeVar

_T = TypeVar('_T')


def _noop(*args, return_value: _T, **kwargs) -> _T:
  del args, kwargs
  return return_value


def _error(*args, **kwargs):
  del args, kwargs
  raise NotImplementedError('This `exm` function does not work in open-source.')


current_experiment = _error
current_work_unit = _error
is_running_under_xmanager = functools.partial(_noop, return_value=False)
add_experiment_artifact = functools.partial(_noop, return_value=None)
add_work_unit_artifact = functools.partial(_noop, return_value=None)
curr_job_name = functools.partial(_noop, return_value='<unknown job name>')
url_to_python_only_logs = functools.partial(
    _noop, return_value='<unknown log url>'
)
set_citc_source = functools.partial(_noop, return_value=None)
