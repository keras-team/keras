# Copyright 2024 The Flax Authors.
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

"""Flax API."""

# pylint: disable=g-import-not-at-top
# pyformat: disable

from flax import configurations
config: configurations.Config = configurations.config
del configurations

from flax import core
from flax import jax_utils
from flax import linen
from flax import serialization
from flax import traverse_util

from flax import version
__version__: str = version.__version__
del version

# DO NOT REMOVE - Marker for internal deprecated API.

# DO NOT REMOVE - Marker for internal logging.

# pyformat: enable
# pylint: enable=g-import-not-at-top
