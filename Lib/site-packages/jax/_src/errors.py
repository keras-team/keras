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
from __future__ import annotations

from jax._src import core
from jax._src.util import set_module

export = set_module('jax.errors')


class _JAXErrorMixin:
  """Mixin for JAX-specific errors"""
  _error_page = 'https://jax.readthedocs.io/en/latest/errors.html'
  _module_name = "jax.errors"

  def __init__(self, message: str):
    error_page = self._error_page
    module_name = self._module_name
    class_name = self.__class__.__name__
    error_msg = f'{message}\nSee {error_page}#{module_name}.{class_name}'
    # https://github.com/python/mypy/issues/5887
    super().__init__(error_msg)  # type: ignore


@export
class JAXTypeError(_JAXErrorMixin, TypeError):
  pass


@export
class JAXIndexError(_JAXErrorMixin, IndexError):
  pass


@export
class ConcretizationTypeError(JAXTypeError):
  """
  This error occurs when a JAX Tracer object is used in a context where a
  concrete value is required (see :ref:`faq-different-kinds-of-jax-values`
  for more on what a Tracer is). In some situations, it can be easily fixed by
  marking problematic values as static; in others, it may indicate that your
  program is doing operations that are not directly supported by JAX's JIT
  compilation model.

  Examples:

  Traced value where static value is expected
    One common cause of this error is using a traced value where a static value
    is required. For example:

      >>> from functools import partial
      >>> from jax import jit
      >>> import jax.numpy as jnp
      >>> @jit
      ... def func(x, axis):
      ...   return x.min(axis)

      >>> func(jnp.arange(4), 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete
      value is expected: axis argument to jnp.min().

    This can often be fixed by marking the problematic argument as static::

        >>> @partial(jit, static_argnums=1)
        ... def func(x, axis):
        ...   return x.min(axis)

        >>> func(jnp.arange(4), 0)
        Array(0, dtype=int32)

  Shape depends on Traced Value
    Such an error may also arise when a shape in your JIT-compiled computation
    depends on the values within a traced quantity. For example::

      >>> @jit
      ... def func(x):
      ...     return jnp.where(x < 0)

      >>> func(jnp.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected:
      The error arose in jnp.nonzero.

    This is an example of an operation that is incompatible with JAX's JIT
    compilation model, which requires array sizes to be known at compile-time.
    Here the size of the returned array depends on the contents of `x`, and such
    code cannot be JIT compiled.

    In many cases it is possible to work around this by modifying the logic used
    in the function; for example here is code with a similar issue::

      >>> @jit
      ... def func(x):
      ...     indices = jnp.where(x > 1)
      ...     return x[indices].sum()

      >>> func(jnp.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete
      value is expected: The error arose in jnp.nonzero.

    And here is how you might express the same operation in a way that avoids
    creation of a dynamically-sized index array::

      >>> @jit
      ... def func(x):
      ...   return jnp.where(x > 1, x, 0).sum()

      >>> func(jnp.arange(4))
      Array(5, dtype=int32)

  To understand more subtleties having to do with tracers vs. regular values,
  and concrete vs. abstract values, you may want to read
  :ref:`faq-different-kinds-of-jax-values`.
  """
  def __init__(self, tracer: core.Tracer, context: str = ""):
    super().__init__(
        "Abstract tracer value encountered where concrete value is expected: "
        f"{tracer._error_repr()}\n{context}{tracer._origin_msg()}\n")


@export
class NonConcreteBooleanIndexError(JAXIndexError):
  """
  This error occurs when a program attempts to use non-concrete boolean indices
  in a traced indexing operation. Under JIT compilation, JAX arrays must have
  static shapes (i.e. shapes that are known at compile-time) and so boolean
  masks must be used carefully. Some logic implemented via boolean masking is
  simply not possible in a :func:`jax.jit` function; in other cases, the logic
  can be re-expressed in a JIT-compatible way, often using the three-argument
  version of :func:`~jax.numpy.where`.

  Following are a few examples of when this error might arise.

  Constructing arrays via boolean masking
    This most commonly arises when attempting to create an array via a boolean
    mask within a JIT context. For example::

      >>> import jax
      >>> import jax.numpy as jnp

      >>> @jax.jit
      ... def positive_values(x):
      ...   return x[x > 0]

      >>> positive_values(jnp.arange(-5, 5))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[10])

    This function is attempting to return only the positive values in the input
    array; the size of this returned array cannot be determined at compile-time
    unless `x` is marked as static, and so operations like this cannot be
    performed under JIT compilation.

  Reexpressible Boolean Logic
    Although creating dynamically sized arrays is not supported directly, in
    many cases it is possible to re-express the logic of the computation in
    terms of a JIT-compatible operation. For example, here is another function
    that fails under JIT for the same reason::

      >>> @jax.jit
      ... def sum_of_positive(x):
      ...   return x[x > 0].sum()

      >>> sum_of_positive(jnp.arange(-5, 5))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[10])

    In this case, however, the problematic array is only an intermediate value,
    and we can instead express the same logic in terms of the JIT-compatible
    three-argument version of :func:`jax.numpy.where`::

      >>> @jax.jit
      ... def sum_of_positive(x):
      ...   return jnp.where(x > 0, x, 0).sum()

      >>> sum_of_positive(jnp.arange(-5, 5))
      Array(10, dtype=int32)

    This pattern of replacing boolean masking with three-argument
    :func:`~jax.numpy.where` is a common solution to this sort of problem.

  Boolean indexing into JAX arrays
    The other situation where this error often arises is when using boolean
    indices, such as with :code:`.at[...].set(...)`. Here is a simple example::

      >>> @jax.jit
      ... def manual_clip(x):
      ...   return x.at[x < 0].set(0)

      >>> manual_clip(jnp.arange(-2, 2))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[4])

    This function is attempting to set values smaller than zero to a scalar fill
    value. As above, this can be addressed by re-expressing the logic in terms
    of :func:`~jax.numpy.where`::

      >>> @jax.jit
      ... def manual_clip(x):
      ...   return jnp.where(x < 0, 0, x)

      >>> manual_clip(jnp.arange(-2, 2))
      Array([0, 0, 0, 1], dtype=int32)
  """
  def __init__(self, tracer: core.Tracer):
    super().__init__(
        f"Array boolean indices must be concrete; got {tracer}\n")


@export
class TracerArrayConversionError(JAXTypeError):
  """
  This error occurs when a program attempts to convert a JAX Tracer object into
  a standard NumPy array (see :ref:`faq-different-kinds-of-jax-values` for more
  on what a Tracer is). It typically occurs in one of a few situations.

  Using non-JAX functions in JAX transformations
    This error can occur if you attempt to use a non-JAX library like ``numpy``
    or ``scipy`` inside a JAX transformation (:func:`~jax.jit`, :func:`~jax.grad`,
    :func:`jax.vmap`, etc.). For example::

      >>> from jax import jit
      >>> import numpy as np

      >>> @jit
      ... def func(x):
      ...   return np.sin(x)

      >>> func(np.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerArrayConversionError: The numpy.ndarray conversion method
      __array__() was called on traced array with shape int32[4]

    In this case, you can fix the issue by using :func:`jax.numpy.sin` in place of
    :func:`numpy.sin`::

      >>> import jax.numpy as jnp
      >>> @jit
      ... def func(x):
      ...   return jnp.sin(x)

      >>> func(jnp.arange(4))
      Array([0.        , 0.84147096, 0.9092974 , 0.14112   ], dtype=float32)

    See also `External Callbacks`_ for options for calling back to host-side computations
    from transformed JAX code.

  Indexing a numpy array with a tracer
    If this error arises on a line that involves array indexing, it may be that
    the array being indexed ``x`` is a standard numpy.ndarray while the indices
    ``idx`` are traced JAX arrays. For example::

      >>> x = np.arange(10)

      >>> @jit
      ... def func(i):
      ...   return x[i]

      >>> func(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerArrayConversionError: The numpy.ndarray conversion method
      __array__() was called on traced array with shape int32[0]

    Depending on the context, you may fix this by converting the numpy array
    into a JAX array::

      >>> @jit
      ... def func(i):
      ...   return jnp.asarray(x)[i]

      >>> func(0)
      Array(0, dtype=int32)

    or by declaring the index as a static argument::

      >>> from functools import partial
      >>> @partial(jit, static_argnums=(0,))
      ... def func(i):
      ...   return x[i]

      >>> func(0)
      Array(0, dtype=int32)

  To understand more subtleties having to do with tracers vs. regular values,
  and concrete vs. abstract values, you may want to read
  :ref:`faq-different-kinds-of-jax-values`.

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  def __init__(self, tracer: core.Tracer):
    super().__init__(
        "The numpy.ndarray conversion method __array__() was called on "
        f"{tracer._error_repr()}{tracer._origin_msg()}")


@export
class TracerIntegerConversionError(JAXTypeError):
  """
  This error can occur when a JAX Tracer object is used in a context where a
  Python integer is expected (see :ref:`faq-different-kinds-of-jax-values` for
  more on what a Tracer is). It typically occurs in a few situations.

  Passing a tracer in place of an integer
    This error can occur if you attempt to pass a traced value to a function
    that requires a static integer argument; for example::

      >>> from jax import jit
      >>> import numpy as np

      >>> @jit
      ... def func(x, axis):
      ...   return np.split(x, 2, axis)

      >>> func(np.arange(4), 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerIntegerConversionError: The __index__() method was called on
      traced array with shape int32[0]

    When this happens, the solution is often to mark the problematic argument as
    static::

      >>> from functools import partial
      >>> @partial(jit, static_argnums=1)
      ... def func(x, axis):
      ...   return np.split(x, 2, axis)

      >>> func(np.arange(10), 0)
      [Array([0, 1, 2, 3, 4], dtype=int32),
       Array([5, 6, 7, 8, 9], dtype=int32)]

    An alternative is to apply the transformation to a closure that encapsulates
    the arguments to be protected, either manually as below or by using
    :func:`functools.partial`::

      >>> jit(lambda arr: np.split(arr, 2, 0))(np.arange(4))
      [Array([0, 1], dtype=int32), Array([2, 3], dtype=int32)]

    **Note a new closure is created at every invocation, which defeats the
    compilation caching mechanism, which is why static_argnums is preferred.**

  Indexing a list with a Tracer
    This error can occur if you attempt to index a Python list with a traced
    quantity.
    For example::

      >>> import jax.numpy as jnp
      >>> from jax import jit

      >>> L = [1, 2, 3]

      >>> @jit
      ... def func(i):
      ...   return L[i]

      >>> func(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerIntegerConversionError: The __index__() method was called on
      traced array with shape int32[0]

    Depending on the context, you can generally fix this either by converting
    the list to a JAX array::

      >>> @jit
      ... def func(i):
      ...   return jnp.array(L)[i]

      >>> func(0)
      Array(1, dtype=int32)

    or by declaring the index as a static argument::

      >>> from functools import partial
      >>> @partial(jit, static_argnums=0)
      ... def func(i):
      ...   return L[i]

      >>> func(0)
      Array(1, dtype=int32, weak_type=True)

  To understand more subtleties having to do with tracers vs. regular values,
  and concrete vs. abstract values, you may want to read
  :ref:`faq-different-kinds-of-jax-values`.
  """
  def __init__(self, tracer: core.Tracer):
    super().__init__(
        f"The __index__() method was called on {tracer._error_repr()}"
        f"{tracer._origin_msg()}")


@export
class TracerBoolConversionError(ConcretizationTypeError):
  """
  This error occurs when a traced value in JAX is used in a context where a
  boolean value is expected (see :ref:`faq-different-kinds-of-jax-values`
  for more on what a Tracer is).

  The boolean cast may be an explicit (e.g. ``bool(x)``) or implicit, through use of
  control flow (e.g. ``if x > 0`` or ``while x``), use of Python boolean
  operators (e.g. ``z = x and y``, ``z = x or y``, ``z = not x``) or functions
  that use them (e.g. ``z = max(x, y)``, ``z = min(x, y)`` etc.).

  In some situations, this problem can be easily fixed by marking traced values as
  static; in others, it may indicate that your program is doing operations that are
  not directly supported by JAX's JIT compilation model.

  Examples:

  Traced value used in control flow
    One case where this often arises is when a traced value is used in
    Python control flow. For example::

      >>> from jax import jit
      >>> import jax.numpy as jnp
      >>> @jit
      ... def func(x, y):
      ...   return x if x.sum() < y.sum() else y

      >>> func(jnp.ones(4), jnp.zeros(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerBoolConversionError: Attempted boolean conversion of JAX Tracer [...]

    We could mark both inputs ``x`` and ``y`` as static, but that would defeat
    the purpose of using :func:`jax.jit` here. Another option is to re-express
    the if statement in terms of the three-term :func:`jax.numpy.where`::

      >>> @jit
      ... def func(x, y):
      ...   return jnp.where(x.sum() < y.sum(), x, y)

      >>> func(jnp.ones(4), jnp.zeros(4))
      Array([0., 0., 0., 0.], dtype=float32)

    For more complicated control flow including loops, see
    :ref:`lax-control-flow`.

  Control flow on traced values
    Another common cause of this error is if you inadvertently trace over a boolean
    flag. For example::

      >>> @jit
      ... def func(x, normalize=True):
      ...   if normalize:
      ...     return x / x.sum()
      ...   return x

      >>> func(jnp.arange(5), True)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerBoolConversionError: Attempted boolean conversion of JAX Tracer ...

    Here because the flag ``normalize`` is traced, it cannot be used in Python
    control flow. In this situation, the best solution is probably to mark this
    value as static::

      >>> from functools import partial
      >>> @partial(jit, static_argnames=['normalize'])
      ... def func(x, normalize=True):
      ...   if normalize:
      ...     return x / x.sum()
      ...   return x

      >>> func(jnp.arange(5), True)
      Array([0. , 0.1, 0.2, 0.3, 0.4], dtype=float32)

    For more on ``static_argnums``, see the documentation of :func:`jax.jit`.

  Using non-JAX aware functions
    Another common cause of this error is using non-JAX aware functions within JAX
    code. For example:

      >>> @jit
      ... def func(x):
      ...   return min(x, 0)

      >>> func(2)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerBoolConversionError: Attempted boolean conversion of JAX Tracer ...

    In this case, the error occurs because Python's built-in ``min`` function is not
    compatible with JAX transforms. This can be fixed by replacing it with
    ``jnp.minumum``:

      >>> @jit
      ... def func(x):
      ...   return jnp.minimum(x, 0)

      >>> print(func(2))
      0

  To understand more subtleties having to do with tracers vs. regular values,
  and concrete vs. abstract values, you may want to read
  :ref:`faq-different-kinds-of-jax-values`.
  """
  def __init__(self, tracer: core.Tracer):
    JAXTypeError.__init__(self,
        f"Attempted boolean conversion of {tracer._error_repr()}."
        f"{tracer._origin_msg()}")


@export
class UnexpectedTracerError(JAXTypeError):
  """
  This error occurs when you use a JAX value that has leaked out of a function.
  What does it mean to leak a value? If you use a JAX transformation on a
  function ``f`` that stores, in some scope outside of ``f``, a reference to
  an intermediate value, that value is considered to have been leaked.
  Leaking values is a side effect. (Read more about avoiding side effects in
  `Pure Functions <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_)

  JAX detects leaks when you then use the leaked value in another
  operation later on, at which point it raises an ``UnexpectedTracerError``.
  To fix this, avoid side effects: if a function computes a value needed
  in an outer scope, return that value from the transformed function explicitly.

  Specifically, a ``Tracer`` is JAX's internal representation of a function's
  intermediate values during transformations, e.g. within :func:`~jax.jit`,
  :func:`~jax.pmap`, :func:`~jax.vmap`, etc. Encountering a ``Tracer`` outside
  of a transformation implies a leak.

  Life-cycle of a leaked value
    Consider the following example of a transformed function which leaks a value
    to an outer scope::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit                   # 1
      ... def side_effecting(x):
      ...   y = x + 1            # 3
      ...   outs.append(y)       # 4

      >>> x = 1
      >>> side_effecting(x)      # 2
      >>> outs[0] + 1            # 5  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      UnexpectedTracerError: Encountered an unexpected tracer.

    In this example we leak a Traced value from an inner transformed scope to an
    outer scope. We get an ``UnexpectedTracerError`` when the leaked value is
    used, not when the value is leaked.

    This example also demonstrates the life-cycle of a leaked value:

      1. A function is transformed (in this case, by :func:`~jax.jit`)
      2. The transformed function is called (initiating an abstract trace of the
         function and turning ``x`` into a ``Tracer``)
      3. The intermediate value ``y``, which will later be leaked, is created
         (an intermediate value of a traced function is also a ``Tracer``)
      4. The value is leaked (appended to a list in an outer scope, escaping
         the function through a side-channel)
      5. The leaked value is used, and an UnexpectedTracerError is raised.

    The UnexpectedTracerError message tries to point to these locations in your
    code by including information about each stage. Respectively:

      1. The name of the transformed function (``side_effecting``) and which
         transform kicked off the trace  :func:`~jax.jit`).
      2. A reconstructed stack trace of where the leaked Tracer was created,
         which includes where the transformed function was called.
         (``When the Tracer was created, the final 5 stack frames were...``).
      3. From the reconstructed stack trace, the line of code that created
         the leaked Tracer.
      4. The leak location is not included in the error message because it is
         difficult to pin down! JAX can only tell you what the leaked value
         looks like (what shape it has and where it was created) and what
         boundary it was leaked over (the name of the transformation and the
         name of the transformed function).
      5. The current error's stack trace points to where the value is used.

    The error can be fixed by the returning the value out of the
    transformed function::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit
      ... def not_side_effecting(x):
      ...   y = x+1
      ...   return y

      >>> x = 1
      >>> y = not_side_effecting(x)
      >>> outs.append(y)
      >>> outs[0] + 1  # all good! no longer a leaked value.
      Array(3, dtype=int32, weak_type=True)

  Leak checker
    As discussed in point 2 and 3 above, JAX shows a reconstructed stack trace
    which points to where the leaked value was created.  This is because
    JAX only raises an error when the leaked value is used, not when the
    value is leaked. This is not the most useful place to raise this error,
    because you need to know the location where the Tracer was leaked to fix the
    error.

    To make this location easier to track down, you can use the leak checker.
    When the leak checker is enabled, an error is raised as soon as a ``Tracer``
    is leaked. (To be more exact, it will raise an error when the transformed
    function from which the ``Tracer`` is leaked returns)

    To enable the leak checker you can use the ``JAX_CHECK_TRACER_LEAKS``
    environment variable or the ``with jax.checking_leaks()`` context manager.

    .. note::
      Note that this tool is experimental and may report false positives. It
      works by disabling some JAX caches, so it will have a negative effect on
      performance and should only be used when debugging.

    Example usage::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit
      ... def side_effecting(x):
      ...   y = x+1
      ...   outs.append(y)

      >>> x = 1
      >>> with jax.checking_leaks():
      ...   y = side_effecting(x)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      Exception: Leaked Trace

  """

  def __init__(self, msg: str):
    super().__init__(msg)


@export
class KeyReuseError(JAXTypeError):
  """
  This error occurs when a PRNG key is reused in an unsafe manner.
  Key reuse is checked only when `jax_debug_key_reuse` is
  set to `True`.

  Here is a simple example of code that would lead to such an error::

    >>> with jax.debug_key_reuse(True):  # doctest: +SKIP
    ...   key = jax.random.key(0)
    ...   value = jax.random.uniform(key)
    ...   new_value = jax.random.uniform(key)
    ...
    ---------------------------------------------------------------------------
    KeyReuseError                             Traceback (most recent call last)
    ...
    KeyReuseError: Previously-consumed key passed to jit-compiled function at index 0

  This sort of key reuse is problematic because the JAX PRNG is stateless, and keys
  must be manually split; For more information on this see `the Pseudorandom Numbers
  tutorial <https://jax.readthedocs.io/en/latest/random-numbers.html>`_.
  """
  pass
