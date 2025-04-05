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

"""Publishes Orbax version."""

# A new PyPI release will be pushed everytime `__version__` is increased.
# Also modify version and date in CHANGELOG.
# LINT.IfChange
__version__ = '0.11.10'
# LINT.ThenChange(//depot//orbax/checkpoint/CHANGELOG.md)


# TODO: b/362813406 - Add latest change timestamp and commit number.
def get_details() -> str:
  """Returns the Orbax version details.

  It includes release version.
  """
  return f'orbax-checkpoint version: {__version__}'
