# Copyright 2023 The ml_dtypes Authors.
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

"""Overload of numpy.finfo to handle dtypes defined in ml_dtypes."""

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
import numpy as np

_bfloat16_dtype = np.dtype(bfloat16)
_float4_e2m1fn_dtype = np.dtype(float4_e2m1fn)
_float6_e2m3fn_dtype = np.dtype(float6_e2m3fn)
_float6_e3m2fn_dtype = np.dtype(float6_e3m2fn)
_float8_e3m4_dtype = np.dtype(float8_e3m4)
_float8_e4m3_dtype = np.dtype(float8_e4m3)
_float8_e4m3b11fnuz_dtype = np.dtype(float8_e4m3b11fnuz)
_float8_e4m3fn_dtype = np.dtype(float8_e4m3fn)
_float8_e4m3fnuz_dtype = np.dtype(float8_e4m3fnuz)
_float8_e5m2_dtype = np.dtype(float8_e5m2)
_float8_e5m2fnuz_dtype = np.dtype(float8_e5m2fnuz)
_float8_e8m0fnu_dtype = np.dtype(float8_e8m0fnu)


class _Bfloat16MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-126")
    self.smallest_normal = bfloat16(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-133")
    self.smallest_subnormal = bfloat16(smallest_subnormal)


class _Float4E2m1fnMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p0")
    self.smallest_normal = float4_e2m1fn(smallest_normal)
    smallest_subnormal = float.fromhex("0x0.8p0")
    self.smallest_subnormal = float4_e2m1fn(smallest_subnormal)


class _Float6E2m3fnMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p0")
    self.smallest_normal = float6_e2m3fn(smallest_normal)
    smallest_subnormal = float.fromhex("0x0.2p0")
    self.smallest_subnormal = float6_e2m3fn(smallest_subnormal)


class _Float6E3m2fnMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-2")
    self.smallest_normal = float6_e3m2fn(smallest_normal)
    smallest_subnormal = float.fromhex("0x0.4p-2")
    self.smallest_subnormal = float6_e3m2fn(smallest_subnormal)


class _Float8E3m4MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-2")
    self.smallest_normal = float8_e3m4(smallest_normal)
    smallest_subnormal = float.fromhex("0x0.1p-2")
    self.smallest_subnormal = float8_e3m4(smallest_subnormal)


class _Float8E4m3MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-6")
    self.smallest_normal = float8_e4m3(smallest_normal)
    smallest_subnormal = float.fromhex("0x0.2p-6")
    self.smallest_subnormal = float8_e4m3(smallest_subnormal)


class _Float8E4m3b11fnuzMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-10")
    self.smallest_normal = float8_e4m3b11fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-13")
    self.smallest_subnormal = float8_e4m3b11fnuz(smallest_subnormal)


class _Float8E4m3fnMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-6")
    self.smallest_normal = float8_e4m3fn(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-9")
    self.smallest_subnormal = float8_e4m3fn(smallest_subnormal)


class _Float8E4m3fnuzMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-7")
    self.smallest_normal = float8_e4m3fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-10")
    self.smallest_subnormal = float8_e4m3fnuz(smallest_subnormal)


class _Float8E5m2MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-14")
    self.smallest_normal = float8_e5m2(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-16")
    self.smallest_subnormal = float8_e5m2(smallest_subnormal)


class _Float8E5m2fnuzMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-15")
    self.smallest_normal = float8_e5m2fnuz(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-17")
    self.smallest_subnormal = float8_e5m2fnuz(smallest_subnormal)


class _Float8E8m0fnuMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-127")
    self.smallest_normal = float8_e8m0fnu(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-127")
    self.smallest_subnormal = float8_e8m0fnu(smallest_subnormal)


class finfo(np.finfo):  # pylint: disable=invalid-name,missing-class-docstring
  __doc__ = np.finfo.__doc__

  @staticmethod
  def _bfloat16_finfo():
    def float_to_str(f):
      return "%12.4e" % float(f)

    tiny = float.fromhex("0x1p-126")
    resolution = 0.01
    eps = float.fromhex("0x1p-7")
    epsneg = float.fromhex("0x1p-8")
    max_ = float.fromhex("0x1.FEp127")

    obj = object.__new__(np.finfo)
    obj.dtype = _bfloat16_dtype
    obj.bits = 16
    obj.eps = bfloat16(eps)
    obj.epsneg = bfloat16(epsneg)
    obj.machep = -7
    obj.negep = -8
    obj.max = bfloat16(max_)
    obj.min = bfloat16(-max_)
    obj.nexp = 8
    obj.nmant = 7
    obj.iexp = obj.nexp
    obj.maxexp = 128
    obj.minexp = -126
    obj.precision = 2
    obj.resolution = bfloat16(resolution)
    # pylint: disable=protected-access
    obj._machar = _Bfloat16MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = bfloat16(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float4_e2m1fn_finfo():
    eps = float.fromhex("0x0.8p0")  # 0.5
    max_ = float.fromhex("0x1.8p2")  # 6.0

    obj = object.__new__(np.finfo)
    obj.dtype = _float4_e2m1fn_dtype
    obj.bits = 4
    obj.eps = eps
    obj.epsneg = eps
    obj.machep = -1
    obj.negep = -1
    obj.max = float4_e2m1fn(max_)
    obj.min = float4_e2m1fn(-max_)
    obj.nexp = 2
    obj.nmant = 1
    obj.iexp = obj.nexp
    obj.maxexp = 3
    obj.minexp = 0
    obj.precision = 0
    obj.resolution = float4_e2m1fn(1.0)
    # pylint: disable=protected-access
    obj._machar = _Float4E2m1fnMachArLike()
    tiny = obj._machar.smallest_normal
    if not hasattr(obj, "tiny"):
      obj.tiny = tiny
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = tiny
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    float_to_str = str
    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(obj.max)
    obj._str_epsneg = float_to_str(obj.epsneg)
    obj._str_eps = float_to_str(obj.eps)
    obj._str_resolution = float_to_str(obj.resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float6_e2m3fn_finfo():
    eps = float.fromhex("0x0.2p0")  # 0.125
    max_ = float.fromhex("0x1.Ep2")  # 7.5

    obj = object.__new__(np.finfo)
    obj.dtype = _float6_e2m3fn_dtype
    obj.bits = 6
    obj.eps = eps
    obj.epsneg = eps
    obj.machep = -3
    obj.negep = -3
    obj.max = float6_e2m3fn(max_)
    obj.min = float6_e2m3fn(-max_)
    obj.nexp = 2
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 3
    obj.minexp = 0
    obj.precision = 0
    obj.resolution = float6_e2m3fn(1.0)
    # pylint: disable=protected-access
    obj._machar = _Float6E2m3fnMachArLike()
    tiny = obj._machar.smallest_normal
    if not hasattr(obj, "tiny"):
      obj.tiny = tiny
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = tiny
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    float_to_str = str
    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(obj.max)
    obj._str_epsneg = float_to_str(obj.epsneg)
    obj._str_eps = float_to_str(obj.eps)
    obj._str_resolution = float_to_str(obj.resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float6_e3m2fn_finfo():
    eps = float.fromhex("0x1p-2")  # 0.25
    max_ = float.fromhex("0x1.Cp4")  # 28

    obj = object.__new__(np.finfo)
    obj.dtype = _float6_e3m2fn_dtype
    obj.bits = 6
    obj.eps = eps
    obj.epsneg = eps / 2
    obj.machep = -2
    obj.negep = -3
    obj.max = float6_e3m2fn(max_)
    obj.min = float6_e3m2fn(-max_)
    obj.nexp = 3
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 5
    obj.minexp = -2
    obj.precision = 0
    obj.resolution = float6_e3m2fn(1.0)
    # pylint: disable=protected-access
    obj._machar = _Float6E3m2fnMachArLike()
    tiny = obj._machar.smallest_normal
    if not hasattr(obj, "tiny"):
      obj.tiny = tiny
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = tiny
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    float_to_str = str
    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(obj.max)
    obj._str_epsneg = float_to_str(obj.epsneg)
    obj._str_eps = float_to_str(obj.eps)
    obj._str_resolution = float_to_str(obj.resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e3m4_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-2")  # 1/4 min normal
    resolution = 0.1
    eps = float.fromhex("0x1p-4")  # 1/16
    epsneg = float.fromhex("0x1p-5")  # 1/32
    max_ = float.fromhex("0x1.Fp3")  # 15.5 max normal

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e3m4_dtype
    obj.bits = 8
    obj.eps = float8_e3m4(eps)
    obj.epsneg = float8_e3m4(epsneg)
    obj.machep = -4
    obj.negep = -5
    obj.max = float8_e3m4(max_)
    obj.min = float8_e3m4(-max_)
    obj.nexp = 3
    obj.nmant = 4
    obj.iexp = obj.nexp
    obj.maxexp = 4
    obj.minexp = -2
    obj.precision = 1
    obj.resolution = float8_e3m4(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E3m4MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e3m4(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-6")  # 1/64 min normal
    resolution = 0.1
    eps = float.fromhex("0x1p-3")  # 1/8
    epsneg = float.fromhex("0x1p-4")  # 1/16
    max_ = float.fromhex("0x1.Ep7")  # 240 max normal

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3_dtype
    obj.bits = 8
    obj.eps = float8_e4m3(eps)
    obj.epsneg = float8_e4m3(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3(max_)
    obj.min = float8_e4m3(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 8
    obj.minexp = -6
    obj.precision = 1
    obj.resolution = float8_e4m3(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3b11fnuz_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-10")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max_ = float.fromhex("0x1.Ep4")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3b11fnuz_dtype
    obj.bits = 8
    obj.eps = float8_e4m3b11fnuz(eps)
    obj.epsneg = float8_e4m3b11fnuz(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3b11fnuz(max_)
    obj.min = float8_e4m3b11fnuz(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 5
    obj.minexp = -10
    obj.precision = 1
    obj.resolution = float8_e4m3b11fnuz(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3b11fnuzMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3b11fnuz(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3fn_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-6")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max_ = float.fromhex("0x1.Cp8")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3fn_dtype
    obj.bits = 8
    obj.eps = float8_e4m3fn(eps)
    obj.epsneg = float8_e4m3fn(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3fn(max_)
    obj.min = float8_e4m3fn(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 9
    obj.minexp = -6
    obj.precision = 1
    obj.resolution = float8_e4m3fn(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3fnMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3fn(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3fnuz_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-7")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max_ = float.fromhex("0x1.Ep7")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3fnuz_dtype
    obj.bits = 8
    obj.eps = float8_e4m3fnuz(eps)
    obj.epsneg = float8_e4m3fnuz(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3fnuz(max_)
    obj.min = float8_e4m3fnuz(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 8
    obj.minexp = -7
    obj.precision = 1
    obj.resolution = float8_e4m3fnuz(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3fnuzMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3fnuz(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e5m2_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-14")
    resolution = 0.1
    eps = float.fromhex("0x1p-2")
    epsneg = float.fromhex("0x1p-3")
    max_ = float.fromhex("0x1.Cp15")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e5m2_dtype
    obj.bits = 8
    obj.eps = float8_e5m2(eps)
    obj.epsneg = float8_e5m2(epsneg)
    obj.machep = -2
    obj.negep = -3
    obj.max = float8_e5m2(max_)
    obj.min = float8_e5m2(-max_)
    obj.nexp = 5
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 16
    obj.minexp = -14
    obj.precision = 1
    obj.resolution = float8_e5m2(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E5m2MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e5m2(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e5m2fnuz_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-15")
    resolution = 0.1
    eps = float.fromhex("0x1p-2")
    epsneg = float.fromhex("0x1p-3")
    max_ = float.fromhex("0x1.Cp15")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e5m2fnuz_dtype
    obj.bits = 8
    obj.eps = float8_e5m2fnuz(eps)
    obj.epsneg = float8_e5m2fnuz(epsneg)
    obj.machep = -2
    obj.negep = -3
    obj.max = float8_e5m2fnuz(max_)
    obj.min = float8_e5m2fnuz(-max_)
    obj.nexp = 5
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 16
    obj.minexp = -15
    obj.precision = 1
    obj.resolution = float8_e5m2fnuz(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E5m2fnuzMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e5m2fnuz(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e8m0fnu_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-127")
    resolution = 0.1
    eps = float.fromhex("0x1p+0")
    epsneg = float.fromhex("0x1p-1")
    max_ = float.fromhex("0x1p+127")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e8m0fnu_dtype
    obj.bits = 8
    obj.eps = float8_e8m0fnu(eps)
    obj.epsneg = float8_e8m0fnu(epsneg)
    obj.machep = 0
    obj.negep = -1
    obj.max = float8_e8m0fnu(max_)
    obj.min = float8_e8m0fnu(tiny)
    obj.nexp = 8
    obj.nmant = 0
    obj.iexp = obj.nexp
    obj.maxexp = 128
    obj.minexp = -127
    obj.precision = 1
    obj.resolution = float8_e8m0fnu(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E8m0fnuMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e8m0fnu(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  _finfo_type_map = {
      _bfloat16_dtype: _bfloat16_finfo,
      _float4_e2m1fn_dtype: _float4_e2m1fn_finfo,
      _float6_e2m3fn_dtype: _float6_e2m3fn_finfo,
      _float6_e3m2fn_dtype: _float6_e3m2fn_finfo,
      _float8_e3m4_dtype: _float8_e3m4_finfo,
      _float8_e4m3_dtype: _float8_e4m3_finfo,
      _float8_e4m3fn_dtype: _float8_e4m3fn_finfo,
      _float8_e4m3fnuz_dtype: _float8_e4m3fnuz_finfo,
      _float8_e4m3b11fnuz_dtype: _float8_e4m3b11fnuz_finfo,
      _float8_e5m2_dtype: _float8_e5m2_finfo,
      _float8_e5m2fnuz_dtype: _float8_e5m2fnuz_finfo,
      _float8_e8m0fnu_dtype: _float8_e8m0fnu_finfo,
  }
  _finfo_name_map = {t.name: t for t in _finfo_type_map}
  _finfo_cache = {
      t: init_fn.__func__() for t, init_fn in _finfo_type_map.items()  # pytype: disable=attribute-error
  }

  def __new__(cls, dtype):
    if isinstance(dtype, str):
      key = cls._finfo_name_map.get(dtype)
    elif isinstance(dtype, np.dtype):
      key = dtype
    else:
      key = np.dtype(dtype)
    i = cls._finfo_cache.get(key)
    if i is not None:
      return i
    return super().__new__(cls, dtype)
