# Copyright 2022 The ml_dtypes Authors.
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

__version__ = "0.5.1"  # Keep in sync with pyproject.toml:version
__all__ = [
    "__version__",
    "bfloat16",
    "finfo",
    "float4_e2m1fn",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "iinfo",
    "int2",
    "int4",
    "uint2",
    "uint4",
]

from typing import Type

from ml_dtypes._finfo import finfo
from ml_dtypes._iinfo import iinfo
from ml_dtypes._ml_dtypes_ext import bfloat16
from ml_dtypes._ml_dtypes_ext import float4_e2m1fn
from ml_dtypes._ml_dtypes_ext import float6_e2m3fn
from ml_dtypes._ml_dtypes_ext import float6_e3m2fn
from ml_dtypes._ml_dtypes_ext import float8_e3m4
from ml_dtypes._ml_dtypes_ext import float8_e4m3
from ml_dtypes._ml_dtypes_ext import float8_e4m3b11fnuz
from ml_dtypes._ml_dtypes_ext import float8_e4m3fn
from ml_dtypes._ml_dtypes_ext import float8_e4m3fnuz
from ml_dtypes._ml_dtypes_ext import float8_e5m2
from ml_dtypes._ml_dtypes_ext import float8_e5m2fnuz
from ml_dtypes._ml_dtypes_ext import float8_e8m0fnu
from ml_dtypes._ml_dtypes_ext import int2
from ml_dtypes._ml_dtypes_ext import int4
from ml_dtypes._ml_dtypes_ext import uint2
from ml_dtypes._ml_dtypes_ext import uint4
import numpy as np

bfloat16: Type[np.generic]
float4_e2m1fn: Type[np.generic]
float6_e2m3fn: Type[np.generic]
float6_e3m2fn: Type[np.generic]
float8_e3m4: Type[np.generic]
float8_e4m3: Type[np.generic]
float8_e4m3b11fnuz: Type[np.generic]
float8_e4m3fn: Type[np.generic]
float8_e4m3fnuz: Type[np.generic]
float8_e5m2: Type[np.generic]
float8_e5m2fnuz: Type[np.generic]
float8_e8m0fnu: Type[np.generic]
int2: Type[np.generic]
int4: Type[np.generic]
uint2: Type[np.generic]
uint4: Type[np.generic]

del np, Type
