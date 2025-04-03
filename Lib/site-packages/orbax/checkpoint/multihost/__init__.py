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

"""Defines exported symbols for package orbax.checkpoint.multihost."""

# pylint: disable=g-importing-member, g-bad-import-order

from orbax.checkpoint._src.multihost.multihost import broadcast_one_to_all
from orbax.checkpoint._src.multihost.multihost import is_primary_host
from orbax.checkpoint._src.multihost.multihost import reached_preemption
from orbax.checkpoint._src.multihost.multihost import sync_global_processes
from orbax.checkpoint._src.multihost.multihost import process_index

from orbax.checkpoint._src.multihost.multihost import BarrierSyncFn
from orbax.checkpoint._src.multihost.multihost import get_barrier_sync_fn

# TODO(cpgaffney) Export multislice symbols.

# The following symbols are temporary workarounds and WILL be removed in the
# future. Please do not use.
from orbax.checkpoint._src.multihost.multihost import is_runtime_to_distributed_ids_initialized
from orbax.checkpoint._src.multihost.multihost import initialize_runtime_to_distributed_ids
