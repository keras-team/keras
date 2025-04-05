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

"""Alias of `etils.ecolab.lazy_imports`."""

from etils.ecolab import lazy_imports
from etils.ecolab.lazy_imports import *

# `import *` only import symbols defined in `__all__`, so manually import
# additional symbols
LAZY_MODULES = lazy_imports.LAZY_MODULES
print_current_imports = lazy_imports.print_current_imports

__dir__ = lazy_imports.__dir__
__all__ = lazy_imports.__all__
