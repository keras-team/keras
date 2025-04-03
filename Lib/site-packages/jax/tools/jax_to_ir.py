# Copyright 2019 The JAX Authors.
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

r"""Tool to convert a JAX function to serialized representations.

This script is meant to be used as part of a genrule that converts a JAX program
into an IR that can be consumed by another system (e.g. a compiler).

Convert to HLO
==============

For example, you can generate an HLO proto for the XLA compiler. The HLO proto
represents an XLA program, and can be run from e.g. a C++ program, without
involving any Python.

This lets you use JAX as a convenient frontend for writing "XLA programs".  From
another perspective, this script lets you make JAX into an ahead-of-time JAX ->
XLA compiler, although when you run the XLA program, it will still be compiled
just-in-time.

See tensorflow/compiler/xla/service/hlo_runner.h.

Usage:

  $ cat prog.py
  import jax.numpy as jnp

  def fn(x, y, z):
    return jnp.dot(x, y) / z

  $ python jax_to_ir.py \
    --fn prog.fn \
    --input_shapes '[("y", "f32[128,32]"), ("x", "f32[8,128]")]' \
    --constants '{"z": 3.14159}' \
    --ir_format HLO \
    --ir_human_dest /tmp/fn_hlo.txt \
    --ir_dest /tmp/fn_hlo.pb

Alternatively, you can use this script via a genrule.  This way bazel will
generate the hlo text/proto as part of compilation, and then e.g. a C++ program
can depend on this.  See jax_to_hlo macro in build_defs.bzl.

The order of elements in input_shapes determines the order of parameters in the
resulting HLO program.

Values of `constants` which are lists are converted to Numpy arrays using
jnp.asarray.  In addition, you can specify constants using the flag
--evaled_constants; values there that are strings are first evaluated using
ast.literal_eval.  --evaled_constants is primarily useful for genrules; Skylark
doesn't support floating-point types, so genrules need to deal in strings.

Note that XLA's backwards-compatibility guarantees for saved HLO are currently
(2019-06-13) best-effort.  It will mostly work, but it will occasionally break,
and the XLA team won't (and in fact will be unable to) help.  One way to be sure
it won't break is to use the same version of XLA to build the HLO as you use to
run it.  The genrule above makes this easy.
"""

from ast import literal_eval
import importlib
import functools
import re

from absl import app
from absl import flags
import jax
import jax.numpy as jnp

try:
  from jax.experimental import jax2tf
except ImportError:
  jax2tf = None  # type: ignore[assignment]

try:
  import tensorflow as tf
except ImportError:
  tf = None


_FN = flags.DEFINE_string(
    'fn', None, "Fully-qualified name of function that we're going to convert"
)
_INPUT_SHAPES = flags.DEFINE_string(
    'input_shapes', None, 'Python dict indicating XLA shapes of params'
)
_CONSTANTS = flags.DEFINE_string(
    'constants', '{}', 'Python dict giving constant values for some params'
)
_EVALED_CONSTANTS = flags.DEFINE_string(
    'evaled_constants',
    '{}',
    'Python dict giving constant values for some params.  '
    'Values in this dict that are of type str are evaluated '
    'using ast.literal_eval.',
)
_IR_FORMAT = flags.DEFINE_enum(
    'ir_format', 'HLO', ('HLO', 'TF'), 'Output format.'
)
_IR_DEST = flags.DEFINE_string('ir_dest', None, 'File to write IR to')
_IR_HUMAN_DEST = flags.DEFINE_string(
    'ir_human_dest', None, 'File to write human readable debug output'
)


def jax_to_ir(fn, input_shapes, *, constants=None, format):
  """Converts a JAX function to a serialized ir and a debug txt dump.

  Args:
    fn: Function to convert.
    input_shapes: List of tuples (arg name, jax.core.ShapedArray),
      indicating the shapes of the arguments to fn.  The order of parameters in
      the resulting XLA program will match the order in this list.
    constants: Dict mapping function argument name to a Python value.  Specified
      arguments these values as compile-time constants.
    format: Which IR format to use. Supported values are 'HLO' and 'TF'.

  Returns:
    A tuple of (compiler_suitable_ir, human_readable_ir).
  """
  if not constants:
    constants = {}

  overlapping_args = {arg_name for arg_name, _ in input_shapes} & set(
      constants.keys())
  if overlapping_args:
    raise ValueError(
        'Arguments appear in both `input_shapes` and `constants`: %s' %
        ', '.join(sorted(overlapping_args)))

  # TODO(tomhennigan): Ideally we could avoid creating actual values here.
  args = [jnp.zeros(s.shape, s.dtype) for _, s in input_shapes]

  # Curry `constants` into the function.
  fn_curried = functools.partial(fn, **constants)

  # Wrapper that takes in args in the order of `input_shapes` and converts them
  # to kwargs for calling `fn`.
  def ordered_wrapper(*args):
    arg_names = [arg_name for arg_name, _ in input_shapes]
    return fn_curried(**dict(zip(arg_names, args)))

  if format == 'HLO':
    comp = jax.jit(ordered_wrapper).lower(*args).compiler_ir('hlo')
    serialized_proto = comp.as_serialized_hlo_module_proto()
    debug_txt = comp.as_hlo_text()
  else:
    assert format == 'TF'
    if tf is None:
      raise ValueError(
          'Conversion to TF graph requires TensorFlow to be installed.')

    f = jax2tf.convert(ordered_wrapper)
    f = tf_wrap_with_input_names(f, input_shapes)
    f = tf.function(f, autograph=False)
    g = f.get_concrete_function(*args).graph.as_graph_def()
    serialized_proto = g.SerializeToString()
    debug_txt = str(g)

  return serialized_proto, debug_txt


def tf_wrap_with_input_names(f, input_shapes):
  def wrapper(*args):
    args = tuple(
        tf.identity(a, name=name) for a, (name, _) in zip(args, input_shapes))
    # NOTE: Output names already set via `jax2tf.convert(..)`.
    return f(*args)
  return wrapper

jax_to_hlo = functools.partial(jax_to_ir, format='HLO')
jax_to_tf = functools.partial(jax_to_ir, format='TF')


def main(argv):
  if len(argv) != 1:
    raise app.UsageError('No positional arguments are accepted.')

  if not _IR_DEST.value and not _IR_HUMAN_DEST.value:
    raise app.Error('At least one of --ir_dest and '
                    '--ir_human_dest is required.')

  module_name, fn_name = _FN.value.rsplit('.', 1)
  module = importlib.import_module(module_name)
  fn = getattr(module, fn_name)

  input_shapes = [(name, parse_shape_str(shape_str))
                  for name, shape_str in literal_eval(_INPUT_SHAPES.value)]

  # Parse --constants and --evaled_constants.
  constants = {}
  for k, v in literal_eval(_CONSTANTS.value).items():
    if isinstance(v, list):
      v = jnp.asarray(v)
    constants[k] = v

  for k, v in literal_eval(_EVALED_CONSTANTS.value).items():
    if isinstance(v, str):
      v = literal_eval(v)
    if isinstance(v, list):
      v = jnp.asarray(v)
    if k in constants:
      raise ValueError(
          'Argument appears in both --constants and --evaled_constants: %s' % k)
    constants[k] = v

  ir, debug_ir = jax_to_ir(fn, input_shapes, constants=constants,
                           format=_IR_FORMAT.value)

  if _IR_DEST.value:
    with open(_IR_DEST.value, 'wb') as f:
      f.write(ir)

  if _IR_HUMAN_DEST.value:
    with open(_IR_HUMAN_DEST.value, 'w') as f:
      f.write(debug_ir)


def parse_shape_str(s):
  match = _SHAPE_RE.match(s)
  if not match:
    raise ValueError(f'Invalid shape {s}. Valid example: "f32[1,2,3]".'
                     f'Note that dtype must be one of {list(_DT)}')
  dtype = _DT[match.group(1)]
  if match.group(2):
    shape = tuple(int(d.strip()) for d in match.group(2).split(","))
  else:
    shape = ()
  return jax.core.ShapedArray(shape, dtype)

_DT = {
    'pred': jnp.bool_,
    'u4': jnp.uint4, 'u8': jnp.uint8, 'u16': jnp.uint16, 'u32': jnp.uint32, 'u64': jnp.uint64,
    's4': jnp.int4, 's8': jnp.int8, 's16': jnp.int16, 's32': jnp.int32, 's64': jnp.int64,
    'bf16': jnp.bfloat16,
    'f16': jnp.float16, 'f32': jnp.float32, 'f64': jnp.float64,
    'c64': jnp.complex64, 'c128': jnp.complex128
}
if hasattr(jnp, 'int2'):
  _DT['s2'] = jnp.int2
if hasattr(jnp, 'uint2'):
  _DT['u2'] = jnp.uint2

_SHAPE_RE = re.compile(f"^({'|'.join(_DT)})\\[\\s*(\\d*[\\s*,\\d+]*)\\s*\\]$")


def set_up_flags():
  flags.mark_flag_as_required('fn')
  flags.mark_flag_as_required('input_shapes')


if __name__ == '__main__':
  set_up_flags()
  app.run(main)
