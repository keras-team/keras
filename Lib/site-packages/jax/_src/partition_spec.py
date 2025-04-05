# Copyright 2021 The JAX Authors.
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

from __future__ import annotations

class UnconstrainedSingleton:

  def __repr__(self):
    return "UNCONSTRAINED"


# Unconstrained sentinel value for PartitionSpec, representing a dimension for
# which the user wants XLA to assign the best partitioning.
# TODO(yashkatariya): May rename to AUTO.
_UNCONSTRAINED_PARTITION = UnconstrainedSingleton()


class PartitionSpec(tuple):
  """Tuple describing how to partition an array across a mesh of devices.

  Each element is either ``None``, a string, or a tuple of strings.
  See the documentation of :class:`jax.sharding.NamedSharding` for more details.

  This class exists so JAX's pytree utilities can distinguish a partition
  specifications from tuples that should be treated as pytrees.
  """

  # A sentinel value representing a dim is unconstrained.
  UNCONSTRAINED = _UNCONSTRAINED_PARTITION

  def __init__(self, *partitions):
    pass

  def __new__(cls, *partitions):
    return tuple.__new__(PartitionSpec, partitions)

  def __repr__(self):
    return "PartitionSpec%s" % tuple.__repr__(self)

  def __reduce__(self):
    return (PartitionSpec, tuple(self))

  def _normalized_spec(self, ndim: int) -> PartitionSpec:
    out = []  # type: ignore
    for p in self:
      if p is None:
        out.append(None)
      elif p == self.UNCONSTRAINED:
        out.append(p)
      elif isinstance(p, (list, tuple)):
        if len(p) == 1:
          out.append(p[0])
        else:
          out.append(tuple(p))
      else:
        out.append(p)
    if len(out) < ndim:
      out.extend([None] * (ndim - len(out)))
    return PartitionSpec(*out)
