# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Utilities to patch JAX functions with faked implementations.

This module provides fake implementations of jax.jit and jax.pmap, which can be
patched over existing implementations for easier debugging.

See https://www.martinfowler.com/articles/mocksArentStubs.html
"""

import contextlib
import functools
import inspect
import os
import re
from typing import Any, Callable, Iterable, Optional, Union
from unittest import mock
from absl import flags
import jax
import jax.numpy as jnp


FLAGS = flags.FLAGS
flags.DEFINE_integer('chex_n_cpu_devices', 1,
                     'Number of CPU threads to use as devices in tests.')
flags.DEFINE_bool('chex_assert_multiple_cpu_devices', False,
                  'Whether to fail if a number of CPU devices is less than 2.')

_xla_device_count_flag_regexp = (
    r'[-]{0,2}xla_force_host_platform_device_count=(\d+)?(\s|$)')


def get_n_cpu_devices_from_xla_flags() -> int:
  """Parses number of CPUs from the XLA environment flags."""
  m = re.match(_xla_device_count_flag_regexp, os.getenv('XLA_FLAGS', ''))

  # At least one CPU device must be available.
  n_devices = int(m.group(1)) if m else 1
  return n_devices


def set_n_cpu_devices(n: Optional[int] = None) -> None:
  """Forces XLA to use `n` CPU threads as host devices.

  This allows `jax.pmap` to be tested on a single-CPU platform.
  This utility only takes effect before XLA backends are initialized, i.e.
  before any JAX operation is executed (including `jax.devices()` etc.).
  See https://github.com/google/jax/issues/1408.

  Args:
    n: A required number of CPU devices (``FLAGS.chex_n_cpu_devices`` is used by
      default).

  Raises:
    RuntimeError: If XLA backends were already initialized.
  """
  n = n or FLAGS['chex_n_cpu_devices'].value

  n_devices = get_n_cpu_devices_from_xla_flags()
  cpu_backend = (jax._src.xla_bridge._backends or {}).get('cpu', None)  # pylint: disable=protected-access
  if cpu_backend is not None and n_devices != n:
    raise RuntimeError(
        f'Attempted to set {n} devices, but {n_devices} CPUs already available:'
        ' ensure that `set_n_cpu_devices` is executed before any JAX operation.'
    )

  xla_flags = os.getenv('XLA_FLAGS', '')
  xla_flags = re.sub(_xla_device_count_flag_regexp, '', xla_flags)
  os.environ['XLA_FLAGS'] = ' '.join(
      [f'--xla_force_host_platform_device_count={n}'] + xla_flags.split())


def convert_to_varargs(sig, *args, **kwargs):
  """Converts varargs+kwargs function arguments into varargs only."""
  bound_args = sig.bind(*args, **kwargs)
  return bound_args.args


def _ignore_axis_index_groups(fn):
  """Wrapper that forces axis_index_groups to be None.

  This is to avoid problems within fake_pmap where parallel operations are
  performed with vmap, rather than pmap. Parallel operations where
  `axis_index_groups` is not `None` are not currently supported under vmap.

  Args:
    fn: the function to wrap

  Returns:
    a wrapped function that forces any keyword argument named
      `axis_index_groups` to be None
  """
  @functools.wraps(fn)
  def _fake(*args, axis_index_groups=None, **kwargs):
    del axis_index_groups
    return fn(*args, axis_index_groups=None, **kwargs)
  return _fake


_fake_all_gather = _ignore_axis_index_groups(jax.lax.all_gather)
_fake_all_to_all = _ignore_axis_index_groups(jax.lax.all_to_all)
_fake_psum = _ignore_axis_index_groups(jax.lax.psum)
_fake_pmean = _ignore_axis_index_groups(jax.lax.pmean)
_fake_pmax = _ignore_axis_index_groups(jax.lax.pmax)
_fake_pmin = _ignore_axis_index_groups(jax.lax.pmin)
_fake_pswapaxes = _ignore_axis_index_groups(jax.lax.pswapaxes)


@functools.wraps(jax.pmap)
def _fake_pmap(fn,
               axis_name: Optional[Any] = None,
               *,
               in_axes=0,
               static_broadcasted_argnums: Union[int, Iterable[int]] = (),
               jit_result: bool = False,
               fake_parallel_axis: bool = False,
               **unused_kwargs):
  """Fake implementation of pmap using vmap."""

  if isinstance(static_broadcasted_argnums, int):
    static_broadcasted_argnums = (static_broadcasted_argnums,)
  if static_broadcasted_argnums and isinstance(in_axes, dict):
    raise NotImplementedError(
        'static_broadcasted_argnums with dict in_axes not supported.')

  fn_signature = inspect.signature(
      fn,
      # Disable 'follow wrapped' because we want the exact signature of fn,
      # not the signature of any function it might wrap.
      follow_wrapped=False)

  @functools.wraps(fn)
  def wrapped_fn(*args, **kwargs):
    # Convert kwargs to varargs
    # This is a workaround for vmapped functions not working with kwargs
    call_args = convert_to_varargs(fn_signature, *args, **kwargs)

    if static_broadcasted_argnums:
      # Make sure vmap does not try to map over `static_broadcasted_argnums`.
      if isinstance(in_axes, int):
        vmap_in_axes = [in_axes] * len(call_args)
      else:
        vmap_in_axes = list(in_axes)
      for argnum in static_broadcasted_argnums:
        vmap_in_axes[argnum] = None

      # To protect the arguments from `static_broadcasted_argnums`,
      # from turning into tracers (because of vmap), we capture the original
      # `call_args` and replace the passed in tracers with original values.
      original_call_args = call_args

      # A function passed to vmap, that will simply replace the static args
      # with their original values.
      def fn_without_statics(*args):
        args_with_original_statics = [
            orig_arg if i in static_broadcasted_argnums else arg
            for i, (arg, orig_arg) in enumerate(zip(args, original_call_args))
        ]
        return fn(*args_with_original_statics)

      # Make sure to avoid turning static args into tracers: Some python objects
      # might not survive vmap. Just replace with an unused constant.
      call_args = [
          1 if i in static_broadcasted_argnums else arg
          for i, arg in enumerate(call_args)
      ]

    else:
      vmap_in_axes = in_axes
      fn_without_statics = fn

    vmapped_fn = jax.vmap(
        fn_without_statics, in_axes=vmap_in_axes, axis_name=axis_name
    )
    if jit_result:
      vmapped_fn = jax.jit(vmapped_fn)

    if fake_parallel_axis:
      call_args = jax.tree_util.tree_map(
          lambda x: jnp.expand_dims(x, axis=0), call_args)

    output = vmapped_fn(*call_args)

    if fake_parallel_axis:
      output = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), output)

    return output

  return wrapped_fn


# pylint:disable=unnecessary-dunder-call
class FakeContext(contextlib.ExitStack):

  def start(self):
    self.__enter__()

  def stop(self):
    self.__exit__(None, None, None)


# pylint:enable=unnecessary-dunder-call


def fake_jit(enable_patching: bool = True) -> FakeContext:
  """Context manager for patching `jax.jit` with the identity function.

  This is intended to be used as a debugging tool to programmatically enable or
  disable JIT compilation.

  Can be used either as a context managed scope:

  .. code-block:: python

    with chex.fake_jit():
      @jax.jit
      def foo(x):
        ...

  or by calling `start` and `stop`:

  .. code-block:: python

    fake_jit_context = chex.fake_jit()
    fake_jit_context.start()

    @jax.jit
      def foo(x):
            ...

    fake_jit_context.stop()

  Args:
    enable_patching: Whether to patch `jax.jit`.

  Returns:
    Context where `jax.jit` is patched with the identity function jax is
    configured to avoid jitting internally whenever possible in functions
    such as `jax.lax.scan`, etc.
  """
  stack = FakeContext()
  stack.enter_context(jax.disable_jit(disable=enable_patching))
  return stack


def fake_pmap(
    enable_patching: bool = True,
    jit_result: bool = False,
    ignore_axis_index_groups: bool = False,
    fake_parallel_axis: bool = False,
) -> FakeContext:
  """Context manager for patching `jax.pmap` with `jax.vmap`.

  This is intended to be used as a debugging tool to programmatically replace
  pmap transformations with a non-parallel vmap transformation.

  Can be used either as a context managed scope:

  .. code-block:: python

    with chex.fake_pmap():
      @jax.pmap
      def foo(x):
        ...

  or by calling `start` and `stop`:

  .. code-block:: python

    fake_pmap_context = chex.fake_pmap()
    fake_pmap_context.start()
    @jax.pmap
      def foo(x):
        ...
    fake_pmap_context.stop()

  Args:
    enable_patching: Whether to patch `jax.pmap`.
    jit_result: Whether the transformed function should be jitted despite not
      being pmapped.
    ignore_axis_index_groups: Whether to force any parallel operation within the
      context to set `axis_index_groups` to be None. This is a compatibility
      option to allow users of the axis_index_groups parameter to run under the
      fake_pmap context. This feature is not currently supported in vmap, and
      will fail, so we force the parameter to be `None`.
      *Warning*: This will produce different results to running under `jax.pmap`
    fake_parallel_axis: Fake a parallel axis

  Returns:
    Context where `jax.pmap` is patched with `jax.vmap`.
  """
  stack = FakeContext()
  if enable_patching:
    patched_pmap = functools.partial(
        _fake_pmap,
        jit_result=jit_result,
        fake_parallel_axis=fake_parallel_axis)

    stack.enter_context(mock.patch('jax.pmap', patched_pmap))

    if ignore_axis_index_groups:
      stack.enter_context(mock.patch('jax.lax.all_gather', _fake_all_gather))
      stack.enter_context(mock.patch('jax.lax.all_to_all', _fake_all_to_all))
      stack.enter_context(mock.patch('jax.lax.psum', _fake_psum))
      stack.enter_context(mock.patch('jax.lax.pmean', _fake_pmean))
      stack.enter_context(mock.patch('jax.lax.pmax', _fake_pmax))
      stack.enter_context(mock.patch('jax.lax.pmin', _fake_pmin))
      stack.enter_context(mock.patch('jax.lax.pswapaxes', _fake_pswapaxes))
    else:
      # Use default implementations
      pass

  return stack


def fake_pmap_and_jit(enable_pmap_patching: bool = True,
                      enable_jit_patching: bool = True) -> FakeContext:
  """Context manager for patching `jax.jit` and `jax.pmap`.

  This is a convenience function, equivalent to nested `chex.fake_pmap` and
  `chex.fake_jit` contexts.

  Note that calling (the true implementation of) `jax.pmap` will compile the
  function, so faking `jax.jit` in this case will not stop the function from
  being compiled.

  Args:
    enable_pmap_patching: Whether to patch `jax.pmap`.
    enable_jit_patching: Whether to patch `jax.jit`.

  Returns:
    Context where jax.pmap and jax.jit are patched with jax.vmap and the
    identity function
  """
  stack = FakeContext()
  stack.enter_context(fake_pmap(enable_pmap_patching))
  stack.enter_context(fake_jit(enable_jit_patching))
  return stack


class OnCallOfTransformedFunction():
  """Injects a callback into any transformed function.

  A typical use-case is jax.jit or jax.pmap which is often hidden deep inside
  the code. This context manager allows to inject a callback function into
  functions which are transformed by the user-specified transformation.
  The callback will receive the transformed function and its arguments.

  The function can be useful to debug, profile and check the calls of any
  transformed function in a program

  For instance:

  with chex.OnCallOfTransformedFunction('jax.jit', print):
    [...]

  would print all calls to any function which was jit-compiled within this
  context.

  We can also automatically create profiles on the first call of all the
  jit compiled functions in the program:

  class profile_once():
    def __init__(self):
      self._first_call = True

    def __call__(self, fn, *args, **kwargs):
      if self._first_call:
        self._first_call = False
        print(profile_from_HLO(fn.lower(*args, **kwargs))

  with chex.OnCallOfTransformedFunction('jax.jit', profile_once()):
    [...]
  """

  def __init__(self, fn_transformation: str, callback_fn: Callable[..., Any]):
    """Creates a new OnCallOfTransformedFunction context manager.

    Args:
      fn_transformation: identifier of the function transformation e.g.
        'jax.jit', 'jax.pmap', ...
      callback_fn: A callback function which receives the transformed function
        and its arguments on every call.
    """
    self._fn_transformation = fn_transformation
    self._callback_fn = callback_fn
    self._patch: mock._patch[Callable[[Any], Any]] = None  # pylint: disable=unsubscriptable-object
    self._original_fn_transformation = None

  def __enter__(self):

    def _new_fn_transformation(fn, *args, **kwargs):
      """Returns a transformed version of the given function."""
      transformed_fn = self._original_fn_transformation(fn, *args, **kwargs)

      @functools.wraps(transformed_fn)
      def _new_transformed_fn(*args, **kwargs):
        """Returns result of the returned function and calls the callback."""
        self._callback_fn(transformed_fn, *args, **kwargs)
        return transformed_fn(*args, **kwargs)

      return _new_transformed_fn

    self._patch = mock.patch(self._fn_transformation, _new_fn_transformation)
    self._original_fn_transformation, unused_local = self._patch.get_original()
    self._patch.start()

  def __exit__(self, *unused_args):
    self._patch.stop()
