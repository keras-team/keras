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

"""Tree utils."""

from etils.etree import backend
from etils.etree import tree_utils
from etils.etree.typing import Tree

# Expose 4 variants of the API depending on which backend is used.
jax = tree_utils.TreeAPI(backend.Jax())  # jax.tree_utils
optree = tree_utils.TreeAPI(backend.Optree())
tree = tree_utils.TreeAPI(backend.DmTree())  # tree (DeepMind)
nest = tree_utils.TreeAPI(backend.Nest())  # tf.nest
py = tree_utils.TreeAPI(backend.Python())  # Pure Python API

map = py.map  # pylint: disable=redefined-builtin
parallel_map = py.parallel_map
unzip = py.unzip
stack = py.stack
spec_like = py.spec_like
