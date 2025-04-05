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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.errors import (
  JAXTypeError as JAXTypeError,
  JAXIndexError as JAXIndexError,
  ConcretizationTypeError as ConcretizationTypeError,
  NonConcreteBooleanIndexError as NonConcreteBooleanIndexError,
  TracerArrayConversionError as TracerArrayConversionError,
  TracerBoolConversionError as TracerBoolConversionError,
  TracerIntegerConversionError as TracerIntegerConversionError,
  UnexpectedTracerError as UnexpectedTracerError,
  KeyReuseError as KeyReuseError,
)

from jax._src.lib import xla_client as _xc
JaxRuntimeError = _xc.XlaRuntimeError
del _xc

from jax._src.traceback_util import SimplifiedTraceback as SimplifiedTraceback
