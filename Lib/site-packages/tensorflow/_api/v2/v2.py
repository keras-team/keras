# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""
Top-level module of TensorFlow. By convention, we refer to this module as
`tf` instead of `tensorflow`, following the common practice of importing
TensorFlow via the command `import tensorflow as tf`.

The primary function of this module is to import all of the public TensorFlow
interfaces into a single place. The interfaces themselves are located in
sub-modules, as described below.

Note that the file `__init__.py` in the TensorFlow source code tree is actually
only a placeholder to enable test cases to run. The TensorFlow build replaces
this file with a file generated from [`api_template.__init__.py`](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/api_template.__init__.py)
"""
# pylint: disable=g-bad-import-order,protected-access,g-import-not-at-top

import distutils as _distutils
import importlib
import inspect as _inspect
import os as _os
import site as _site
import sys as _sys

_os.environ.setdefault("ENABLE_RUNTIME_UPTIME_TELEMETRY", "1")

# Do not remove this line; See https://github.com/tensorflow/tensorflow/issues/42596
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import KerasLazyLoader as _KerasLazyLoader

# Make sure code inside the TensorFlow codebase can use tf2.enabled() at import.
_os.environ["TF2_BEHAVIOR"] = "1"
from tensorflow.python import tf2 as _tf2
_tf2.enable()

from tensorflow._api.v2 import __internal__
from tensorflow._api.v2 import __operators__
from tensorflow._api.v2 import audio
from tensorflow._api.v2 import autodiff
from tensorflow._api.v2 import autograph
from tensorflow._api.v2 import bitwise
from tensorflow._api.v2 import compat
from tensorflow._api.v2 import config
from tensorflow._api.v2 import data
from tensorflow._api.v2 import debugging
from tensorflow._api.v2 import distribute
from tensorflow._api.v2 import dtypes
from tensorflow._api.v2 import errors
from tensorflow._api.v2 import experimental
from tensorflow._api.v2 import feature_column
from tensorflow._api.v2 import graph_util
from tensorflow._api.v2 import image
from tensorflow._api.v2 import io
from tensorflow._api.v2 import linalg
from tensorflow._api.v2 import lite
from tensorflow._api.v2 import lookup
from tensorflow._api.v2 import math
from tensorflow._api.v2 import mlir
from tensorflow._api.v2 import nest
from tensorflow._api.v2 import nn
from tensorflow._api.v2 import profiler
from tensorflow._api.v2 import quantization
from tensorflow._api.v2 import queue
from tensorflow._api.v2 import ragged
from tensorflow._api.v2 import random
from tensorflow._api.v2 import raw_ops
from tensorflow._api.v2 import saved_model
from tensorflow._api.v2 import sets
from tensorflow._api.v2 import signal
from tensorflow._api.v2 import sparse
from tensorflow._api.v2 import strings
from tensorflow._api.v2 import summary
from tensorflow._api.v2 import sysconfig
from tensorflow._api.v2 import test
from tensorflow._api.v2 import tpu
from tensorflow._api.v2 import train
from tensorflow._api.v2 import types
from tensorflow._api.v2 import version
from tensorflow._api.v2 import xla
from tensorflow.python.ops.gen_array_ops import bitcast # line: 558
from tensorflow.python.ops.gen_array_ops import broadcast_to # line: 829
from tensorflow.python.ops.gen_array_ops import extract_volume_patches # line: 2569
from tensorflow.python.ops.gen_array_ops import identity_n # line: 4268
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse # line: 9182
from tensorflow.python.ops.gen_array_ops import scatter_nd # line: 9318
from tensorflow.python.ops.gen_array_ops import space_to_batch_nd # line: 10152
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add as tensor_scatter_nd_add # line: 11302
from tensorflow.python.ops.gen_array_ops import tensor_scatter_max as tensor_scatter_nd_max # line: 11480
from tensorflow.python.ops.gen_array_ops import tensor_scatter_min as tensor_scatter_nd_min # line: 11601
from tensorflow.python.ops.gen_array_ops import tensor_scatter_sub as tensor_scatter_nd_sub # line: 11711
from tensorflow.python.ops.gen_array_ops import tile # line: 12133
from tensorflow.python.ops.gen_array_ops import unravel_index # line: 12894
from tensorflow.python.ops.gen_control_flow_ops import no_op # line: 475
from tensorflow.python.ops.gen_data_flow_ops import dynamic_partition # line: 594
from tensorflow.python.ops.gen_data_flow_ops import dynamic_stitch # line: 736
from tensorflow.python.ops.gen_experimental_dataset_ops import check_pinned # line: 701
from tensorflow.python.ops.gen_linalg_ops import matrix_square_root # line: 1913
from tensorflow.python.ops.gen_logging_ops import timestamp # line: 886
from tensorflow.python.ops.gen_math_ops import acosh # line: 231
from tensorflow.python.ops.gen_math_ops import asin # line: 991
from tensorflow.python.ops.gen_math_ops import asinh # line: 1091
from tensorflow.python.ops.gen_math_ops import atan # line: 1184
from tensorflow.python.ops.gen_math_ops import atan2 # line: 1284
from tensorflow.python.ops.gen_math_ops import atanh # line: 1383
from tensorflow.python.ops.gen_math_ops import cos # line: 2521
from tensorflow.python.ops.gen_math_ops import cosh # line: 2615
from tensorflow.python.ops.gen_math_ops import greater # line: 4243
from tensorflow.python.ops.gen_math_ops import greater_equal # line: 4344
from tensorflow.python.ops.gen_math_ops import less # line: 5280
from tensorflow.python.ops.gen_math_ops import less_equal # line: 5381
from tensorflow.python.ops.gen_math_ops import logical_and # line: 5836
from tensorflow.python.ops.gen_math_ops import logical_not # line: 5975
from tensorflow.python.ops.gen_math_ops import logical_or # line: 6062
from tensorflow.python.ops.gen_math_ops import maximum # line: 6383
from tensorflow.python.ops.gen_math_ops import minimum # line: 6639
from tensorflow.python.ops.gen_math_ops import neg as negative # line: 6986
from tensorflow.python.ops.gen_math_ops import real_div as realdiv # line: 8141
from tensorflow.python.ops.gen_math_ops import sin # line: 10372
from tensorflow.python.ops.gen_math_ops import sinh # line: 10465
from tensorflow.python.ops.gen_math_ops import square # line: 12035
from tensorflow.python.ops.gen_math_ops import tan # line: 12425
from tensorflow.python.ops.gen_math_ops import tanh # line: 12519
from tensorflow.python.ops.gen_math_ops import truncate_div as truncatediv # line: 12674
from tensorflow.python.ops.gen_math_ops import truncate_mod as truncatemod # line: 12768
from tensorflow.python.ops.gen_nn_ops import approx_top_k # line: 33
from tensorflow.python.ops.gen_nn_ops import conv # line: 1061
from tensorflow.python.ops.gen_nn_ops import conv2d_backprop_filter_v2 # line: 1609
from tensorflow.python.ops.gen_nn_ops import conv2d_backprop_input_v2 # line: 1977
from tensorflow.python.ops.gen_ragged_array_ops import ragged_fill_empty_rows # line: 196
from tensorflow.python.ops.gen_ragged_array_ops import ragged_fill_empty_rows_grad # line: 305
from tensorflow.python.ops.gen_random_index_shuffle_ops import random_index_shuffle # line: 30
from tensorflow.python.ops.gen_spectral_ops import fftnd # line: 620
from tensorflow.python.ops.gen_spectral_ops import ifftnd # line: 991
from tensorflow.python.ops.gen_spectral_ops import irfftnd # line: 1347
from tensorflow.python.ops.gen_spectral_ops import rfftnd # line: 1707
from tensorflow.python.ops.gen_string_ops import as_string # line: 29
from tensorflow.python.data.ops.optional_ops import OptionalSpec # line: 205
from tensorflow.python.eager.backprop import GradientTape # line: 704
from tensorflow.python.eager.context import executing_eagerly # line: 2570
from tensorflow.python.eager.polymorphic_function.polymorphic_function import function # line: 1300
from tensorflow.python.framework.constant_op import constant # line: 177
from tensorflow.python.framework.device_spec import DeviceSpecV2 as DeviceSpec # line: 46
from tensorflow.python.framework.dtypes import DType # line: 51
from tensorflow.python.framework.dtypes import as_dtype # line: 793
from tensorflow.python.framework.dtypes import bfloat16 # line: 450
from tensorflow.python.framework.dtypes import bool # line: 414
from tensorflow.python.framework.dtypes import complex128 # line: 401
from tensorflow.python.framework.dtypes import complex64 # line: 394
from tensorflow.python.framework.dtypes import double # line: 388
from tensorflow.python.framework.dtypes import float16 # line: 373
from tensorflow.python.framework.dtypes import float32 # line: 380
from tensorflow.python.framework.dtypes import float64 # line: 386
from tensorflow.python.framework.dtypes import half # line: 374
from tensorflow.python.framework.dtypes import int16 # line: 354
from tensorflow.python.framework.dtypes import int32 # line: 360
from tensorflow.python.framework.dtypes import int64 # line: 366
from tensorflow.python.framework.dtypes import int8 # line: 348
from tensorflow.python.framework.dtypes import qint16 # line: 426
from tensorflow.python.framework.dtypes import qint32 # line: 432
from tensorflow.python.framework.dtypes import qint8 # line: 420
from tensorflow.python.framework.dtypes import quint16 # line: 444
from tensorflow.python.framework.dtypes import quint8 # line: 438
from tensorflow.python.framework.dtypes import resource # line: 312
from tensorflow.python.framework.dtypes import string # line: 408
from tensorflow.python.framework.dtypes import uint16 # line: 330
from tensorflow.python.framework.dtypes import uint32 # line: 336
from tensorflow.python.framework.dtypes import uint64 # line: 342
from tensorflow.python.framework.dtypes import uint8 # line: 324
from tensorflow.python.framework.dtypes import variant # line: 318
from tensorflow.python.framework.importer import import_graph_def # line: 358
from tensorflow.python.framework.indexed_slices import IndexedSlices # line: 54
from tensorflow.python.framework.indexed_slices import IndexedSlicesSpec # line: 203
from tensorflow.python.framework.load_library import load_library # line: 120
from tensorflow.python.framework.load_library import load_op_library # line: 31
from tensorflow.python.framework.ops import Graph # line: 1948
from tensorflow.python.framework.ops import Operation # line: 1068
from tensorflow.python.framework.ops import RegisterGradient # line: 1674
from tensorflow.python.framework.ops import control_dependencies # line: 4529
from tensorflow.python.framework.ops import device_v2 as device # line: 4420
from tensorflow.python.framework.ops import get_current_name_scope # line: 5685
from tensorflow.python.framework.ops import init_scope # line: 4733
from tensorflow.python.framework.ops import inside_function # line: 4871
from tensorflow.python.framework.ops import is_symbolic_tensor # line: 6197
from tensorflow.python.framework.ops import name_scope_v2 as name_scope # line: 5720
from tensorflow.python.framework.ops import no_gradient # line: 1723
from tensorflow.python.framework.sparse_tensor import SparseTensor # line: 48
from tensorflow.python.framework.sparse_tensor import SparseTensorSpec # line: 377
from tensorflow.python.framework.tensor import Tensor # line: 138
from tensorflow.python.framework.tensor import TensorSpec # line: 917
from tensorflow.python.framework.tensor_conversion import convert_to_tensor_v2_with_dispatch as convert_to_tensor # line: 96
from tensorflow.python.framework.tensor_conversion_registry import register_tensor_conversion_function # line: 80
from tensorflow.python.framework.tensor_shape import TensorShape # line: 747
from tensorflow.python.framework.tensor_util import constant_value as get_static_value # line: 903
from tensorflow.python.framework.tensor_util import is_tf_type as is_tensor # line: 1135
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray # line: 640
from tensorflow.python.framework.tensor_util import make_tensor_proto # line: 432
from tensorflow.python.framework.type_spec import TypeSpec # line: 49
from tensorflow.python.framework.type_spec import type_spec_from_value # line: 958
from tensorflow.python.framework.versions import COMPILER_VERSION as __compiler_version__ # line: 41
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as __cxx11_abi_flag__ # line: 48
from tensorflow.python.framework.versions import CXX_VERSION as __cxx_version__ # line: 54
from tensorflow.python.framework.versions import GIT_VERSION as __git_version__ # line: 35
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as __monolithic_build__ # line: 60
from tensorflow.python.framework.versions import VERSION as __version__ # line: 29
from tensorflow.python.module.module import Module # line: 29
from tensorflow.python.ops.array_ops import batch_to_space_v2 as batch_to_space # line: 3849
from tensorflow.python.ops.array_ops import boolean_mask_v2 as boolean_mask # line: 1538
from tensorflow.python.ops.array_ops import broadcast_dynamic_shape # line: 527
from tensorflow.python.ops.array_ops import broadcast_static_shape # line: 561
from tensorflow.python.ops.array_ops import concat # line: 1349
from tensorflow.python.ops.array_ops import edit_distance # line: 3523
from tensorflow.python.ops.array_ops import expand_dims_v2 as expand_dims # line: 392
from tensorflow.python.ops.array_ops import fill # line: 204
from tensorflow.python.ops.array_ops import fingerprint # line: 6366
from tensorflow.python.ops.array_ops import gather_v2 as gather # line: 4996
from tensorflow.python.ops.array_ops import gather_nd_v2 as gather_nd # line: 5428
from tensorflow.python.ops.array_ops import guarantee_const # line: 6705
from tensorflow.python.ops.array_ops import identity # line: 253
from tensorflow.python.ops.array_ops import meshgrid # line: 3377
from tensorflow.python.ops.array_ops import newaxis # line: 60
from tensorflow.python.ops.array_ops import one_hot # line: 3987
from tensorflow.python.ops.array_ops import ones # line: 2916
from tensorflow.python.ops.array_ops import ones_like_v2 as ones_like # line: 2848
from tensorflow.python.ops.array_ops import pad_v2 as pad # line: 3196
from tensorflow.python.ops.array_ops import parallel_stack # line: 1135
from tensorflow.python.ops.array_ops import rank # line: 878
from tensorflow.python.ops.array_ops import repeat # line: 6651
from tensorflow.python.ops.array_ops import required_space_to_batch_paddings # line: 3681
from tensorflow.python.ops.array_ops import reshape # line: 63
from tensorflow.python.ops.array_ops import reverse_sequence_v2 as reverse_sequence # line: 4723
from tensorflow.python.ops.array_ops import searchsorted # line: 6109
from tensorflow.python.ops.array_ops import sequence_mask # line: 4165
from tensorflow.python.ops.array_ops import shape_v2 as shape # line: 597
from tensorflow.python.ops.array_ops import shape_n # line: 731
from tensorflow.python.ops.array_ops import size_v2 as size # line: 761
from tensorflow.python.ops.array_ops import slice # line: 939
from tensorflow.python.ops.array_ops import space_to_batch_v2 as space_to_batch # line: 3784
from tensorflow.python.ops.array_ops import split # line: 1743
from tensorflow.python.ops.array_ops import squeeze_v2 as squeeze # line: 4287
from tensorflow.python.ops.array_ops import stop_gradient # line: 6724
from tensorflow.python.ops.array_ops import strided_slice # line: 995
from tensorflow.python.ops.array_ops import tensor_scatter_nd_update # line: 5536
from tensorflow.python.ops.array_ops import transpose_v2 as transpose # line: 1825
from tensorflow.python.ops.array_ops import unique # line: 1642
from tensorflow.python.ops.array_ops import unique_with_counts # line: 1690
from tensorflow.python.ops.array_ops import where_v2 as where # line: 4444
from tensorflow.python.ops.array_ops import zeros # line: 2598
from tensorflow.python.ops.array_ops import zeros_like_v2 as zeros_like # line: 2700
from tensorflow.python.ops.array_ops_stack import stack # line: 24
from tensorflow.python.ops.array_ops_stack import unstack # line: 88
from tensorflow.python.ops.batch_ops import batch_function as nondifferentiable_batch_function # line: 28
from tensorflow.python.ops.check_ops import assert_equal_v2 as assert_equal # line: 762
from tensorflow.python.ops.check_ops import assert_greater_v2 as assert_greater # line: 978
from tensorflow.python.ops.check_ops import assert_less_v2 as assert_less # line: 942
from tensorflow.python.ops.check_ops import assert_rank_v2 as assert_rank # line: 1064
from tensorflow.python.ops.check_ops import ensure_shape # line: 2226
from tensorflow.python.ops.clip_ops import clip_by_global_norm # line: 298
from tensorflow.python.ops.clip_ops import clip_by_norm # line: 152
from tensorflow.python.ops.clip_ops import clip_by_value # line: 34
from tensorflow.python.ops.cond import cond_for_tf_v2 as cond # line: 243
from tensorflow.python.ops.control_flow_assert import Assert # line: 62
from tensorflow.python.ops.control_flow_case import case_v2 as case # line: 33
from tensorflow.python.ops.control_flow_ops import group # line: 1958
from tensorflow.python.ops.control_flow_ops import tuple_v2 as tuple # line: 2037
from tensorflow.python.ops.control_flow_switch_case import switch_case # line: 181
from tensorflow.python.ops.critical_section_ops import CriticalSection # line: 121
from tensorflow.python.ops.custom_gradient import custom_gradient # line: 45
from tensorflow.python.ops.custom_gradient import grad_pass_through # line: 773
from tensorflow.python.ops.custom_gradient import recompute_grad # line: 600
from tensorflow.python.ops.functional_ops import foldl_v2 as foldl # line: 161
from tensorflow.python.ops.functional_ops import foldr_v2 as foldr # line: 358
from tensorflow.python.ops.functional_ops import scan_v2 as scan # line: 691
from tensorflow.python.ops.gradients_impl import gradients_v2 as gradients # line: 188
from tensorflow.python.ops.gradients_impl import HessiansV2 as hessians # line: 455
from tensorflow.python.ops.gradients_util import AggregationMethod # line: 943
from tensorflow.python.ops.histogram_ops import histogram_fixed_width # line: 103
from tensorflow.python.ops.histogram_ops import histogram_fixed_width_bins # line: 31
from tensorflow.python.ops.init_ops_v2 import Constant as constant_initializer # line: 204
from tensorflow.python.ops.init_ops_v2 import Ones as ones_initializer # line: 157
from tensorflow.python.ops.init_ops_v2 import RandomNormal as random_normal_initializer # line: 371
from tensorflow.python.ops.init_ops_v2 import RandomUniform as random_uniform_initializer # line: 302
from tensorflow.python.ops.init_ops_v2 import Zeros as zeros_initializer # line: 110
from tensorflow.python.ops.linalg_ops import eig # line: 382
from tensorflow.python.ops.linalg_ops import eigvals # line: 414
from tensorflow.python.ops.linalg_ops import eye # line: 196
from tensorflow.python.ops.linalg_ops import norm_v2 as norm # line: 561
from tensorflow.python.ops.logging_ops import print_v2 as print # line: 147
from tensorflow.python.ops.manip_ops import roll # line: 27
from tensorflow.python.ops.map_fn import map_fn_v2 as map_fn # line: 614
from tensorflow.python.ops.math_ops import abs # line: 361
from tensorflow.python.ops.math_ops import acos # line: 5815
from tensorflow.python.ops.math_ops import add # line: 3862
from tensorflow.python.ops.math_ops import add_n # line: 3943
from tensorflow.python.ops.math_ops import argmax_v2 as argmax # line: 264
from tensorflow.python.ops.math_ops import argmin_v2 as argmin # line: 318
from tensorflow.python.ops.math_ops import cast # line: 940
from tensorflow.python.ops.math_ops import complex # line: 695
from tensorflow.python.ops.math_ops import cumsum # line: 4194
from tensorflow.python.ops.math_ops import divide # line: 442
from tensorflow.python.ops.math_ops import equal # line: 1806
from tensorflow.python.ops.math_ops import exp # line: 5712
from tensorflow.python.ops.math_ops import floor # line: 5846
from tensorflow.python.ops.math_ops import linspace_nd as linspace # line: 113
from tensorflow.python.ops.math_ops import matmul # line: 3421
from tensorflow.python.ops.math_ops import multiply # line: 477
from tensorflow.python.ops.math_ops import not_equal # line: 1843
from tensorflow.python.ops.math_ops import pow # line: 665
from tensorflow.python.ops.math_ops import range # line: 1962
from tensorflow.python.ops.math_ops import reduce_all # line: 3110
from tensorflow.python.ops.math_ops import reduce_any # line: 3216
from tensorflow.python.ops.math_ops import reduce_logsumexp # line: 3321
from tensorflow.python.ops.math_ops import reduce_max # line: 2991
from tensorflow.python.ops.math_ops import reduce_mean # line: 2517
from tensorflow.python.ops.math_ops import reduce_min # line: 2863
from tensorflow.python.ops.math_ops import reduce_prod # line: 2691
from tensorflow.python.ops.math_ops import reduce_sum # line: 2173
from tensorflow.python.ops.math_ops import round # line: 910
from tensorflow.python.ops.math_ops import saturate_cast # line: 1025
from tensorflow.python.ops.math_ops import scalar_mul_v2 as scalar_mul # line: 656
from tensorflow.python.ops.math_ops import sigmoid # line: 4096
from tensorflow.python.ops.math_ops import sign # line: 743
from tensorflow.python.ops.math_ops import sqrt # line: 5673
from tensorflow.python.ops.math_ops import subtract # line: 541
from tensorflow.python.ops.math_ops import tensordot # line: 5188
from tensorflow.python.ops.math_ops import truediv # line: 1476
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map # line: 452
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor # line: 65
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec # line: 2319
from tensorflow.python.ops.script_ops import numpy_function # line: 807
from tensorflow.python.ops.script_ops import eager_py_func as py_function # line: 460
from tensorflow.python.ops.sort_ops import argsort # line: 86
from tensorflow.python.ops.sort_ops import sort # line: 29
from tensorflow.python.ops.special_math_ops import einsum # line: 618
from tensorflow.python.ops.tensor_array_ops import TensorArray # line: 971
from tensorflow.python.ops.tensor_array_ops import TensorArraySpec # line: 1363
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients # line: 22
from tensorflow.python.ops.variable_scope import variable_creator_scope # line: 2719
from tensorflow.python.ops.variables import Variable # line: 204
from tensorflow.python.ops.variables import VariableAggregationV2 as VariableAggregation # line: 92
from tensorflow.python.ops.variables import VariableSynchronization # line: 64
from tensorflow.python.ops.while_loop import while_loop_v2 as while_loop # line: 35
from tensorflow.python.platform.tf_logging import get_logger # line: 93




# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow.foo import bar" works.
# We're using bitwise, but there's nothing special about that.
_API_MODULE = _sys.modules[__name__].bitwise
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
_current_module = _sys.modules[__name__]

if not hasattr(_current_module, "__path__"):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)

# Hook external TensorFlow modules.

# Load tensorflow-io-gcs-filesystem if enabled
if (_os.getenv("TF_USE_MODULAR_FILESYSTEM", "0") == "true" or
    _os.getenv("TF_USE_MODULAR_FILESYSTEM", "0") == "1"):
  import tensorflow_io_gcs_filesystem as _tensorflow_io_gcs_filesystem

# Lazy-load Keras v2/3.
_tf_uses_legacy_keras = (
    _os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"))
setattr(_current_module, "keras", _KerasLazyLoader(globals()))
_module_dir = _module_util.get_parent_dir_for_name("keras._tf_keras.keras")
_current_module.__path__ = [_module_dir] + _current_module.__path__
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name("tf_keras.api._v2.keras")
else:
  _module_dir = _module_util.get_parent_dir_for_name("keras.api._v2.keras")
_current_module.__path__ = [_module_dir] + _current_module.__path__


# Enable TF2 behaviors
from tensorflow.python.compat import v2_compat as _compat
_compat.enable_v2_behavior()
_major_api_version = 2


# Load all plugin libraries from site-packages/tensorflow-plugins if we are
# running under pip.
# TODO(gunan): Find a better location for this code snippet.
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi

# Get sitepackages directories for the python installation.
_site_packages_dirs = []
if _site.ENABLE_USER_SITE and _site.USER_SITE is not None:
  _site_packages_dirs += [_site.USER_SITE]
_site_packages_dirs += [p for p in _sys.path if "site-packages" in p]
if "getsitepackages" in dir(_site):
  _site_packages_dirs += _site.getsitepackages()

if "sysconfig" in dir(_distutils):
  _site_packages_dirs += [_distutils.sysconfig.get_python_lib()]

_site_packages_dirs = list(set(_site_packages_dirs))

# Find the location of this exact file.
_current_file_location = _inspect.getfile(_inspect.currentframe())

def _running_from_pip_package():
  return any(
      _current_file_location.startswith(dir_) for dir_ in _site_packages_dirs)

if _running_from_pip_package():
  # TODO(gunan): Add sanity checks to loaded modules here.

  # Load first party dynamic kernels.
  _tf_dir = _os.path.dirname(_current_file_location)
  _kernel_dir = _os.path.join(_tf_dir, "core", "kernels")
  if _os.path.exists(_kernel_dir):
    _ll.load_library(_kernel_dir)

  # Load third party dynamic kernels.
  for _s in _site_packages_dirs:
    _plugin_dir = _os.path.join(_s, "tensorflow-plugins")
    if _os.path.exists(_plugin_dir):
      _ll.load_library(_plugin_dir)
      # Load Pluggable Device Library
      _ll.load_pluggable_device_library(_plugin_dir)

if _os.getenv("TF_PLUGGABLE_DEVICE_LIBRARY_PATH", ""):
  _ll.load_pluggable_device_library(
      _os.getenv("TF_PLUGGABLE_DEVICE_LIBRARY_PATH")
  )

# Add Keras module aliases
_losses = _KerasLazyLoader(globals(), submodule="losses", name="losses")
_metrics = _KerasLazyLoader(globals(), submodule="metrics", name="metrics")
_optimizers = _KerasLazyLoader(
    globals(), submodule="optimizers", name="optimizers")
_initializers = _KerasLazyLoader(
    globals(), submodule="initializers", name="initializers")
setattr(_current_module, "losses", _losses)
setattr(_current_module, "metrics", _metrics)
setattr(_current_module, "optimizers", _optimizers)
setattr(_current_module, "initializers", _initializers)


# Do an eager load for Keras' code so that any function/method that needs to
# happen at load time will trigger, eg registration of optimizers in the
# SavedModel registry.
# See b/196254385 for more details.
try:
  if _tf_uses_legacy_keras:
    importlib.import_module("tf_keras.src.optimizers")
  else:
    importlib.import_module("keras.src.optimizers")
except (ImportError, AttributeError):
  pass

del importlib

# Delete modules that should be hidden from dir().
# Don't fail if these modules are not available.
# For e.g. this file will be originally placed under tensorflow/_api/v1 which
# does not have "python", "core" directories. Then, it will be copied
# to tensorflow/ which does have these two directories.
try:
  del python
except NameError:
  pass
try:
  del core
except NameError:
  pass
try:
  del compiler
except NameError:
  pass


_names_with_underscore = ['__compiler_version__', '__cxx11_abi_flag__', '__cxx_version__', '__git_version__', '__internal__', '__monolithic_build__', '__operators__', '__version__']
__all__ = [_s for _s in dir() if not _s.startswith('_')]
__all__.extend([_s for _s in _names_with_underscore])
          
