# Copyright 2023 The JAX Authors.
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

"""Modules for JAX extensions.

The :mod:`jax.extend` module provides modules for access to JAX
internal machinery. See
`JEP #15856 <https://jax.readthedocs.io/en/latest/jep/15856-jex.html>`_.

This module is not the only means by which JAX aims to be
extensible. For example, the main JAX API offers mechanisms for
`customizing derivatives
<https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_,
`registering custom pytree definitions
<https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees>`_,
and more.

API policy
----------

Unlike the
`public API <https://jax.readthedocs.io/en/latest/api_compatibility.html>`_,
this module offers **no compatibility guarantee** across releases.
Breaking changes will be announced via the
`JAX project changelog <https://jax.readthedocs.io/en/latest/changelog.html>`_.
"""

from jax.extend import (
    backend as backend,
    core as core,
    ffi as ffi,
    linear_util as linear_util,
    random as random,
    source_info_util as source_info_util,
)
