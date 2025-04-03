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

"""A variable dict is a normal Python dictionary, which is a container for one
or more "variable collections", each of which are nested dictionaries whose
leaves are ``jax.numpy`` arrays.

The different variable collections share the same nested tree structure.

For example, consider the following variable dictionary::

  {
    "params": {
      "Conv1": { "weight": ..., "bias": ... },
      "BatchNorm1": { "scale": ..., "mean": ... },
      "Conv2": {...}
    },
    "batch_stats": {
      "BatchNorm1": { "moving_mean": ..., "moving_average": ...}
    }
  }

In this case, the ``"BatchNorm1"`` key lives in both the ``"params"`` and
```"batch_stats""`` collections. This reflects the fact that the submodule
named ``""BatchNorm1""`` has both trainable parameters (the ``"params"`` collection),
as well as other non-trainable variables (the ``"batch_stats"`` collection)

TODO: Make "variable dict" design note, and link to it from here.
"""

from .scope import Variable
