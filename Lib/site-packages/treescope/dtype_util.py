# Copyright 2024 The Treescope Authors.
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

"""Utilities for working with dtypes.

JAX extends the Numpy dtype system using the `ml_dtypes` package. Unfortunately,
these extended dtypes do not integrate directly with Numpy subdtype checks.
This module provides utilities to perform these checks that do not depend on
JAX being installed.
"""

import numpy as np
import numpy.typing


def is_integer_dtype(dtype: numpy.typing.DTypeLike) -> bool:
  """Returns whether the given dtype is an integer dtype.

  Supports both basic numpy dtypes and the extended dtypes in the `ml_dtypes`
  package (if installed).

  Args:
    dtype: The dtype to check.

  Returns:
    True if the given dtype is an integer dtype.
  """
  dtype = np.dtype(dtype)
  if np.issubdtype(dtype, np.integer):
    return True
  if isinstance(dtype.type, type) and dtype.type.__module__ == "ml_dtypes":
    import ml_dtypes  # pylint: disable=import-outside-toplevel

    try:
      _ = ml_dtypes.iinfo(dtype)
      return True
    except ValueError:
      return False
  return False


def is_floating_dtype(dtype: numpy.typing.DTypeLike) -> bool:
  """Returns whether the given dtype is a floating dtype.

  Supports both basic numpy dtypes and the extended dtypes in the `ml_dtypes`
  package (if installed).

  Args:
    dtype: The dtype to check.

  Returns:
    True if the given dtype is a floating dtype.
  """
  dtype = np.dtype(dtype)
  if np.issubdtype(dtype, np.floating):
    return True
  if isinstance(dtype.type, type) and dtype.type.__module__ == "ml_dtypes":
    import ml_dtypes  # pylint: disable=import-outside-toplevel

    try:
      _ = ml_dtypes.finfo(dtype)
      return True
    except ValueError:
      return False
  return False


def get_dtype_name(dtype: numpy.typing.DTypeLike) -> str:
  """Safely extracts a name for a dtype."""
  # Render scalar type objects as their literal names.
  if isinstance(dtype, type) and issubclass(dtype, np.generic):
    return dtype.__name__
  # Render any other dtype-like objects as the name of the concrete dtype they
  # convert to.
  try:
    return np.dtype(dtype).name
  except TypeError:
    return str(dtype)
