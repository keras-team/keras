# Copyright 2022 The JAX Authors.
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
"""Interfaces to stages of the compiled execution process.

JAX transformations that compile just in time for execution, such as
``jax.jit`` and ``jax.pmap``, also support a common means of explicit
lowering and compilation *ahead of time*. This module defines types
that represent the stages of this process.

For more, see the `AOT walkthrough <https://jax.readthedocs.io/en/latest/aot.html>`_.
"""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.stages import (
  Compiled as Compiled,
  CompilerOptions as CompilerOptions,
  Lowered as Lowered,
  Wrapped as Wrapped,
  ArgInfo as ArgInfo,
  OutInfo as OutInfo,
  Traced as Traced,
)
