# Copyright 2018 The JAX Authors.
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

from functools import partial
import operator
from typing import cast, Any

import numpy as np

import jax.numpy as jnp
from jax import jit
from jax import jvp
from jax import vmap
from jax import lax

from jax._src import core
from jax._src import custom_derivatives
from jax._src import deprecations
from jax._src import dtypes
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact, promote_dtypes_inexact
from jax._src.ops import special as ops_special
from jax._src.third_party.scipy.betaln import betaln as _betaln_impl
from jax._src.typing import Array, ArrayLike
from jax._src.nn.functions import softmax as nn_softmax
from jax._src.nn.functions import log_softmax as nn_log_softmax


def gammaln(x: ArrayLike) -> Array:
  r"""Natural log of the absolute value of the gamma function.

  JAX implementation of :obj:`scipy.special.gammaln`.

  .. math::

     \mathrm{gammaln}(x) = \log(|\Gamma(x)|)

  Where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function.

  Args:
    x: arraylike, real valued.

  Returns:
    array containing the values of the log-gamma function

  See Also:
    - :func:`jax.scipy.special.gammaln`: the natural log of the gamma function
    - :func:`jax.scipy.special.gammasgn`: the sign of the gamma function

  Notes:
    ``gammaln`` does not support complex-valued inputs.
  """
  x, = promote_args_inexact("gammaln", x)
  return lax.lgamma(x)


@jit
def gammasgn(x: ArrayLike) -> Array:
  r"""Sign of the gamma function.

  JAX implementation of :obj:`scipy.special.gammasgn`.

  .. math::

    \mathrm{gammasgn}(x) = \begin{cases}
      +1 & \Gamma(x) > 0 \\
      -1 & \Gamma(x) < 0
    \end{cases}

  Where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function.
  Because :math:`\Gamma(x)` is never zero, no condition is required for this case.

  * if :math:`x = -\infty`, NaN is returned.
  * if :math:`x = \pm 0`, :math:`\pm 1` is returned.
  * if :math:`x` is a negative integer, NaN is returned. The sign of gamma
    at a negative integer depends on from which side the pole is approached.
  * if :math:`x = \infty`, :math:`1` is returned.
  * if :math:`x` is NaN, NaN is returned.

  Args:
    x: arraylike, real valued.

  Returns:
    array containing the sign of the gamma function

  See Also:
    - :func:`jax.scipy.special.gamma`: the gamma function
    - :func:`jax.scipy.special.gammaln`: the natural log of the gamma function
  """
  x, = promote_args_inexact("gammasgn", x)
  typ = x.dtype.type
  floor_x = lax.floor(x)
  x_negative = x < 0
  return jnp.select(
    [(x_negative & (x == floor_x)) | jnp.isnan(x),
     (x_negative & (floor_x % 2 != 0)) | ((x == 0) & jnp.signbit(x))],
    [typ(np.nan), typ(-1.0)],
    typ(1.0))


def gamma(x: ArrayLike) -> Array:
  r"""The gamma function.

  JAX implementation of :obj:`scipy.special.gamma`.

  The gamma function is defined for :math:`\Re(z)>0` as

  .. math::

     \mathrm{gamma}(z) = \Gamma(z) = \int_0^\infty t^{z-1}e^{-t}\mathrm{d}t

  and is extended by analytic continuation to arbitrary complex values `z`.
  For positive integers `n`, the gamma function is related to the
  :func:`~jax.scipy.special.factorial` function via the following identity:

  .. math::

     \Gamma(n) = (n - 1)!

  * if :math:`z = -\infty`, NaN is returned.
  * if :math:`x = \pm 0`, :math:`\pm \infty` is returned.
  * if :math:`x` is a negative integer, NaN is returned. The sign of gamma
    at a negative integer depends on from which side the pole is approached.
  * if :math:`x = \infty`, :math:`\infty` is returned.
  * if :math:`x` is NaN, NaN is returned.

  Args:
    x: arraylike, real valued.

  Returns:
    array containing the values of the gamma function

  See Also:
    - :func:`jax.scipy.special.factorial`: the factorial function.
    - :func:`jax.scipy.special.gammaln`: the natural log of the gamma function
    - :func:`jax.scipy.special.gammasgn`: the sign of the gamma function

  Notes:
    Unlike the scipy version, JAX's ``gamma`` does not support complex-valued
    inputs.
  """
  x, = promote_args_inexact("gamma", x)
  return gammasgn(x) * lax.exp(lax.lgamma(x))


def betaln(a: ArrayLike, b: ArrayLike) -> Array:
  r"""Natural log of the absolute value of the beta function

  JAX implementation of :obj:`scipy.special.betaln`.

  .. math::

     \mathrm{betaln}(a, b) = \log B(a, b)

  where :math:`B` is the :func:`~jax.scipy.special.beta` function.

  Args:
    a: arraylike, real-valued.  Parameter *a* of the beta distribution.
    b: arraylike, real-valued.  Parameter *b* of the beta distribution.

  Returns:
    array containing the values of the log-beta function

  See Also:
    :func:`jax.scipy.special.beta`
  """
  a, b = promote_args_inexact("betaln", a, b)
  return _betaln_impl(a, b)


def factorial(n: ArrayLike, exact: bool = False) -> Array:
  r"""Factorial function

  JAX implementation of :obj:`scipy.special.factorial`

  .. math::

     \mathrm{factorial}(n) = n! = \prod_{k=1}^n k

  Args:
    n: arraylike, values for which factorial will be computed elementwise
    exact: bool, only ``exact=False`` is supported.

  Returns:
    array containing values of the factorial.

  Notes:
    This computes the float-valued factorial via the :func:`~jax.scipy.special.gamma`
    function. JAX does not support exact factorials, because it is not particularly
    useful: above ``n=20``, the exact result cannot be represented by 64-bit integers,
    which are the largest integers available to JAX.

  See Also:
    :func:`jax.scipy.special.gamma`
  """
  if exact:
    raise NotImplementedError("factorial with exact=True")
  n, = promote_args_inexact("factorial", n)
  return jnp.where(n < 0, 0, lax.exp(lax.lgamma(n + 1)))


def beta(a: ArrayLike, b: ArrayLike) -> Array:
  r"""The beta function

  JAX implementation of :obj:`scipy.special.beta`.

  .. math::

     \mathrm{beta}(a, b) = B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a + b)}

  where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function.

  Args:
    a: arraylike, real-valued. Parameter *a* of the beta distribution.
    b: arraylike, real-valued. Parameter *b* of the beta distribution.

  Returns:
    array containing the values of the beta function.

  See Also:
    - :func:`jax.scipy.special.gamma`
    - :func:`jax.scipy.special.betaln`
  """
  a, b = promote_args_inexact("beta", a, b)
  sign = gammasgn(a) * gammasgn(b) * gammasgn(a + b)
  return sign * lax.exp(betaln(a, b))


def betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> Array:
  r"""The regularized incomplete beta function.

  JAX implementation of :obj:`scipy.special.betainc`.

  .. math::

     \mathrm{betainc}(a, b, x) = B(a, b)\int_0^x t^{a-1}(1-t^{b-1})\mathrm{d}t

  where :math:`B(a, b)` is the :func:`~jax.scipy.special.beta` function.

  Args:
    a: arraylike, real-valued. Parameter *a* of the beta distribution.
    b: arraylike, real-valued. Parameter *b* of the beta distribution.
    x: arraylike, real-valued. Upper limit of the integration.

  Returns:
    array containing values of the betainc function

  See Also:
    - :func:`jax.scipy.special.beta`
    - :func:`jax.scipy.special.betaln`
  """
  a, b, x = promote_args_inexact("betainc", a, b, x)
  return lax.betainc(a, b, x)


def digamma(x: ArrayLike) -> Array:
  r"""The digamma function

  JAX implementation of :obj:`scipy.special.digamma`.

  .. math::

     \mathrm{digamma}(z) = \psi(z) = \frac{\mathrm{d}}{\mathrm{d}z}\log \Gamma(z)

  where :math:`\Gamma(z)` is the :func:`~jax.scipy.special.gamma` function.

  Args:
    x: arraylike, real-valued.

  Returns:
    array containing values of the digamma function.

  Notes:
    The JAX version of `digamma` accepts real-valued inputs.

  See also:
    - :func:`jax.scipy.special.gamma`
    - :func:`jax.scipy.special.polygamma`
  """
  x, = promote_args_inexact("digamma", x)
  return lax.digamma(x)


def gammainc(a: ArrayLike, x: ArrayLike) -> Array:
  r"""The regularized lower incomplete gamma function.

  JAX implementation of :obj:`scipy.special.gammainc`.

  .. math::

     \mathrm{gammainc}(x; a) = \frac{1}{\Gamma(a)}\int_0^x t^{a-1}e^{-t}\mathrm{d}t

  where :math:`\Gamma(a)` is the :func:`~jax.scipy.special.gamma` function.

  Args:
    a: arraylike, real-valued. Positive shape parameter of the gamma distribution.
    x: arraylike, real-valued. Non-negative upper limit of integration

  Returns:
    array containing values of the gammainc function.

  See Also:
    - :func:`jax.scipy.special.gamma`
    - :func:`jax.scipy.special.gammaincc`
  """
  a, x = promote_args_inexact("gammainc", a, x)
  return lax.igamma(a, x)


def gammaincc(a: ArrayLike, x: ArrayLike) -> Array:
  r"""The regularized upper incomplete gamma function.

  JAX implementation of :obj:`scipy.special.gammaincc`.

  .. math::

     \mathrm{gammaincc}(x; a) = \frac{1}{\Gamma(a)}\int_x^\infty t^{a-1}e^{-t}\mathrm{d}t

  where :math:`\Gamma(a)` is the :func:`~jax.scipy.special.gamma` function.

  Args:
    a: arraylike, real-valued. Positive shape parameter of the gamma distribution.
    x: arraylike, real-valued. Non-negative lower limit of integration

  Returns:
    array containing values of the gammaincc function.

  See Also:
    - :func:`jax.scipy.special.gamma`
    - :func:`jax.scipy.special.gammainc`
  """
  a, x = promote_args_inexact("gammaincc", a, x)
  return lax.igammac(a, x)


def erf(x: ArrayLike) -> Array:
  r"""The error function

  JAX implementation of :obj:`scipy.special.erf`.

  .. math::

     \mathrm{erf}(x) = \frac{2}{\sqrt\pi} \int_{0}^x e^{-t^2} \mathrm{d}t

  Args:
    x: arraylike, real-valued.

  Returns:
    array containing values of the error function.

  Notes:
     The JAX version only supports real-valued inputs.

  See also:
    - :func:`jax.scipy.special.erfc`
    - :func:`jax.scipy.special.erfinv`
  """
  x, = promote_args_inexact("erf", x)
  return lax.erf(x)


def erfc(x: ArrayLike) -> Array:
  r"""The complement of the error function

  JAX implementation of :obj:`scipy.special.erfc`.

  .. math::

     \mathrm{erfc}(x) = \frac{2}{\sqrt\pi} \int_{x}^\infty e^{-t^2} \mathrm{d}t

  This is the complement of the error function :func:`~jax.scipy.special.erf`,
  ``erfc(x) = 1 - erf(x)``.

  Args:
    x: arraylike, real-valued.

  Returns:
    array containing values of the complement of the error function.

  Notes:
     The JAX version only supports real-valued inputs.

  See also:
    - :func:`jax.scipy.special.erf`
    - :func:`jax.scipy.special.erfinv`
  """
  x, = promote_args_inexact("erfc", x)
  return lax.erfc(x)


def erfinv(x: ArrayLike) -> Array:
  """The inverse of the error function

  JAX implementation of :obj:`scipy.special.erfinv`.

  Returns the inverse of :func:`~jax.scipy.special.erf`.

  Args:
    x: arraylike, real-valued.

  Returns:
    array containing values of the inverse error function.

  Notes:
     The JAX version only supports real-valued inputs.

  See also:
    - :func:`jax.scipy.special.erf`
    - :func:`jax.scipy.special.erfc`
  """
  x, = promote_args_inexact("erfinv", x)
  return lax.erf_inv(x)


@custom_derivatives.custom_jvp
def logit(x: ArrayLike) -> Array:
  r"""The logit function

  JAX implementation of :obj:`scipy.special.logit`.

  .. math::

     \mathrm{logit}(p) = \log\frac{p}{1 - p}

  Args:
    x: arraylike, real-valued.

  Returns:
    array containing values of the logit function.
  """
  x, = promote_args_inexact("logit", x)
  return lax.log(lax.div(x, lax.sub(_lax_const(x, 1), x)))
logit.defjvps(
    lambda g, ans, x: lax.div(g, lax.mul(x, lax.sub(_lax_const(x, 1), x))))


def expit(x: ArrayLike) -> Array:
  r"""The logistic sigmoid (expit) function

  JAX implementation of :obj:`scipy.special.expit`.

  .. math::

     \mathrm{expit}(x) = \frac{1}{1 + e^{-x}}

  Args:
    x: arraylike, real-valued.

  Returns:
    array containing values of the expit function.
  """
  x, = promote_args_inexact("expit", x)
  return lax.logistic(x)


logsumexp = ops_special.logsumexp


@custom_derivatives.custom_jvp
def xlogy(x: ArrayLike, y: ArrayLike) -> Array:
  """Compute x*log(y), returning 0 for x=0.

  JAX implementation of :obj:`scipy.special.xlogy`.

  This is defined to return zero when :math:`(x, y) = (0, 0)`, with a custom
  derivative rule so that automatic differentiation is well-defined at this point.

  Args:
    x: arraylike, real-valued.
    y: arraylike, real-valued.

  Returns:
    array containing xlogy values.

  See also:
    :func:`jax.scipy.special.xlog1py`
  """
  # Note: xlogy(0, 0) should return 0 according to the function documentation.
  x, y = promote_args_inexact("xlogy", x, y)
  x_ok = x != 0.
  return jnp.where(x_ok, lax.mul(x, lax.log(y)), jnp.zeros_like(x))

def _xlogy_jvp(primals, tangents):
  (x, y) = primals
  (x_dot, y_dot) = tangents
  result = xlogy(x, y)
  return result, (x_dot * lax.log(y) + y_dot * x / y).astype(result.dtype)
xlogy.defjvp(_xlogy_jvp)


@custom_derivatives.custom_jvp
def xlog1py(x: ArrayLike, y: ArrayLike) -> Array:
  """Compute x*log(1 + y), returning 0 for x=0.

  JAX implementation of :obj:`scipy.special.xlog1py`.

  This is defined to return 0 when :math:`(x, y) = (0, -1)`, with a custom
  derivative rule so that automatic differentiation is well-defined at this point.

  Args:
    x: arraylike, real-valued.
    y: arraylike, real-valued.

  Returns:
    array containing xlog1py values.

  See also:
    :func:`jax.scipy.special.xlogy`
  """
  # Note: xlog1py(0, -1) should return 0 according to the function documentation.
  x, y = promote_args_inexact("xlog1py", x, y)
  x_ok = x != 0.
  return jnp.where(x_ok, lax.mul(x, lax.log1p(y)), jnp.zeros_like(x))

def _xlog1py_jvp(primals, tangents):
  (x, y) = primals
  (x_dot, y_dot) = tangents
  result = xlog1py(x, y)
  return result, (x_dot * lax.log1p(y) + y_dot * x / (1 + y)).astype(result.dtype)
xlog1py.defjvp(_xlog1py_jvp)

@custom_derivatives.custom_jvp
def _xlogx(x):
  """Compute x log(x) with well-defined derivatives."""
  return xlogy(x, x)

def _xlogx_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  return  _xlogx(x), x_dot * (lax.log(x) + 1)
_xlogx.defjvp(_xlogx_jvp)


def entr(x: ArrayLike) -> Array:
  r"""The entropy function

  JAX implementation of :obj:`scipy.special.entr`.

  .. math::

     \mathrm{entr}(x) = \begin{cases}
       -x\log(x) & x > 0 \\
       0 & x = 0\\
       -\infty & \mathrm{otherwise}
     \end{cases}

  Args:
    x: arraylike, real-valued.

  Returns:
    array containing entropy values.

  See also:
    - :func:`jax.scipy.special.kl_div`
    - :func:`jax.scipy.special.rel_entr`
  """
  x, = promote_args_inexact("entr", x)
  return lax.select(lax.lt(x, _lax_const(x, 0)),
                    lax.full_like(x, -np.inf),
                    lax.neg(_xlogx(x)))


def multigammaln(a: ArrayLike, d: ArrayLike) -> Array:
  r"""The natural log of the multivariate gamma function.

  JAX implementation of :func:`scipy.special.multigammaln`.

  .. math::

     \mathrm{multigammaln}(a, d) = \log\Gamma_d(a)

  where

  .. math::

     \Gamma_d(a) = \pi^{d(d-1)/4}\prod_{i=1}^d\Gamma(a-(i-1)/2)

  and :math:`\Gamma(x)` is the :func:`~jax.scipy.special.gamma` function.

  Args:
    a: arraylike, real-valued.
    d: int, the dimension of the integration space.

  Returns:
    array containing values of the log-multigamma function.

  See also:
    - :func:`jax.scipy.special.gamma`
  """
  d = core.concrete_or_error(int, d, "d argument of multigammaln")
  a, d_ = promote_args_inexact("multigammaln", a, d)

  constant = lax.mul(lax.mul(lax.mul(_lax_const(a, 0.25), d_),
                             lax.sub(d_, _lax_const(a, 1))),
                     lax.log(_lax_const(a, np.pi)))
  b = lax.div(jnp.arange(d, dtype=d_.dtype), _lax_const(a, 2))
  res = jnp.sum(gammaln(jnp.expand_dims(a, axis=-1) -
                        jnp.expand_dims(b, axis=tuple(range(a.ndim)))),
                axis=-1)
  return res + constant


def kl_div(
    p: ArrayLike,
    q: ArrayLike,
) -> Array:
  r"""The Kullback-Leibler divergence.

  JAX implementation of :obj:`scipy.special.kl_div`.

  .. math::

     \mathrm{kl\_div}(p, q) = \begin{cases}
       p\log(p/q)-p+q & p>0,q>0\\
       q & p=0,q\ge 0\\
       \infty & \mathrm{otherwise}
    \end{cases}

  Args:
    p: arraylike, real-valued.
    q: arraylike, real-valued.

  Returns:
    array of KL-divergence values

  See also:
    - :func:`jax.scipy.special.entr`
    - :func:`jax.scipy.special.rel_entr`
  """
  p, q = promote_args_inexact("kl_div", p, q)
  return rel_entr(p, q) - p + q


def rel_entr(
    p: ArrayLike,
    q: ArrayLike,
) -> Array:
  r"""The relative entropy function.

  JAX implementation of :obj:`scipy.special.rel_entr`.

  .. math::

     \mathrm{rel\_entr}(p, q) = \begin{cases}
       p\log(p/q) & p>0,q>0\\
       0 & p=0,q\ge 0\\
       \infty & \mathrm{otherwise}
    \end{cases}

  Args:
    p: arraylike, real-valued.
    q: arraylike, real-valued.

  Returns:
    array of relative entropy values.

  See also:
    - :func:`jax.scipy.special.entr`
    - :func:`jax.scipy.special.kl_div`
  """
  p, q = promote_args_inexact("rel_entr", p, q)
  zero = _lax_const(p, 0.0)
  both_gt_zero_mask = lax.bitwise_and(lax.gt(p, zero), lax.gt(q, zero))
  one_zero_mask = lax.bitwise_and(lax.eq(p, zero), lax.ge(q, zero))

  safe_p = jnp.where(both_gt_zero_mask, p, 1)
  safe_q = jnp.where(both_gt_zero_mask, q, 1)
  log_val = lax.sub(_xlogx(safe_p), xlogy(safe_p, safe_q))
  result = jnp.where(
      both_gt_zero_mask, log_val, jnp.where(one_zero_mask, zero, jnp.inf)
  )
  return result

# coefs of (2k)! / B_{2k} where B are bernoulli numbers
# those numbers are obtained using https://www.wolframalpha.com
_BERNOULLI_COEFS = [
    12,
    -720,
    30240,
    -1209600,
    47900160,
    -1307674368000 / 691,
    74724249600,
    -10670622842880000 / 3617,
    5109094217170944000 / 43867,
    -802857662698291200000 / 174611,
    14101100039391805440000 / 77683,
    -1693824136731743669452800000 / 236364091,
    186134520519971831808000000 / 657931,
    -37893265687455865519472640000000 / 3392780147,
    759790291646040068357842010112000000 / 1723168255201,
    -134196726836183700385281186201600000000 / 7709321041217,
]


@custom_derivatives.custom_jvp
def zeta(x: ArrayLike, q: ArrayLike | None = None) -> Array:
  r"""The Hurwitz zeta function.

  JAX implementation of :func:`scipy.special.zeta`. JAX does not implement
  the Riemann zeta function (i.e. ``q = None``).

  .. math::

     \zeta(x, q) = \sum_{n=0}^\infty \frac{1}{(n + q)^x}

  Args:
    x: arraylike, real-valued
    q: arraylike, real-valued

  Returns:
    array of zeta function values
  """
  if q is None:
    raise NotImplementedError(
      "Riemann zeta function not implemented; pass q != None to compute the Hurwitz Zeta function.")
  x, q = promote_args_inexact("zeta", x, q)
  return lax.zeta(x, q)


# There is no general closed-form derivative for the zeta function, so we compute
# derivatives via a series expansion
def _zeta_series_expansion(x: ArrayLike, q: ArrayLike | None = None) -> Array:
  if q is None:
    raise NotImplementedError(
      "Riemann zeta function not implemented; pass q != None to compute the Hurwitz Zeta function.")
  # Reference: Johansson, Fredrik.
  # "Rigorous high-precision computation of the Hurwitz zeta function and its derivatives."
  # Numerical Algorithms 69.2 (2015): 253-270.
  # https://arxiv.org/abs/1309.2877 - formula (5)
  # here we keep the same notation as in reference
  s, a = promote_args_inexact("zeta", x, q)
  dtype = lax.dtype(a).type
  s_, a_ = jnp.expand_dims(s, -1), jnp.expand_dims(a, -1)
  # precision ~ N, M
  N = M = dtype(8) if lax.dtype(a) == jnp.float32 else dtype(16)
  assert M <= len(_BERNOULLI_COEFS)
  k = jnp.expand_dims(np.arange(N, dtype=N.dtype), tuple(range(a.ndim)))
  S = jnp.sum((a_ + k) ** -s_, -1)
  I = lax.div((a + N) ** (dtype(1) - s), s - dtype(1))
  T0 = (a + N) ** -s
  m = jnp.expand_dims(np.arange(2 * M, dtype=M.dtype), tuple(range(s.ndim)))
  s_over_a = (s_ + m) / (a_ + N)
  T1 = jnp.cumprod(s_over_a, -1)[..., ::2]
  T1 = jnp.clip(T1, max=jnp.finfo(dtype).max)
  coefs = np.expand_dims(np.array(_BERNOULLI_COEFS[:T1.shape[-1]], dtype=dtype),
                         tuple(range(a.ndim)))
  T1 = T1 / coefs
  T = T0 * (dtype(0.5) + T1.sum(-1))
  return S + I + T

zeta.defjvp(partial(jvp, _zeta_series_expansion))


def polygamma(n: ArrayLike, x: ArrayLike) -> Array:
  r"""The polygamma function.

  JAX implementation of :func:`scipy.special.polygamma`.

  .. math::

     \mathrm{polygamma}(n, x) = \psi^{(n)}(x) = \frac{\mathrm{d}^n}{\mathrm{d}x^n}\log \Gamma(x)

  where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function.

  Args:
    n: arraylike, integer-valued. The order of the derivative.
    x: arraylike, real-valued. The value at which to evaluate the function.

  Returns:
    array

  See also:
    - :func:`jax.scipy.special.gamma`
    - :func:`jax.scipy.special.digamma`
  """
  assert jnp.issubdtype(lax.dtype(n), jnp.integer)
  n_arr, x_arr = promote_args_inexact("polygamma", n, x)
  return lax.polygamma(n_arr, x_arr)


# Normal distributions

# Functions "ndtr" and "ndtri" are derived from calculations made in:
# https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
# The "spence" function is also based on the Cephes library with
# the corresponding spence.c file located in the tarball:
# https://netlib.org/cephes/misc.tgz
# In the following email exchange, the author gives his consent to redistribute
# derived works under an Apache 2.0 license.
#
# From: Stephen Moshier <steve@moshier.net>
# Date: Sat, Jun 9, 2018 at 2:36 PM
# Subject: Re: Licensing cephes under Apache (BSD-like) license.
# To: rif <rif@google.com>
#
#
#
# Hello Rif,
#
# Yes, Google may distribute Cephes files under the Apache 2 license.
#
# If clarification is needed, I do not favor BSD over other free licenses.
# I would agree that Apache 2 seems to cover the concern you mentioned
# about sublicensees.
#
# Best wishes for good luck with your projects!
# Steve Moshier
#
#
#
# On Thu, 31 May 2018, rif wrote:
#
# > Hello Steve.
# > My name is Rif. I work on machine learning software at Google.
# >
# > Your cephes software continues to be incredibly useful and widely used. I
# > was wondering whether it would be permissible for us to use the Cephes code
# > under the Apache 2.0 license, which is extremely similar in permissions to
# > the BSD license (Wikipedia comparisons). This would be quite helpful to us
# > in terms of avoiding multiple licenses on software.
# >
# > I'm sorry to bother you with this (I can imagine you're sick of hearing
# > about this by now), but I want to be absolutely clear we're on the level and
# > not misusing your important software. In former conversation with Eugene
# > Brevdo (ebrevdo@google.com), you wrote "If your licensing is similar to BSD,
# > the formal way that has been handled is simply to add a statement to the
# > effect that you are incorporating the Cephes software by permission of the
# > author." I wanted to confirm that (a) we could use the Apache license, (b)
# > that we don't need to (and probably you don't want to) keep getting
# > contacted about individual uses, because your intent is generally to allow
# > this software to be reused under "BSD-like" license, and (c) you're OK
# > letting incorporators decide whether a license is sufficiently BSD-like?
# >
# > Best,
# >
# > rif
# >
# >
# >

# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.)
_LOGNDTR_FLOAT64_LOWER = np.array(-20, np.float64)
_LOGNDTR_FLOAT32_LOWER = np.array(-10, np.float32)

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
_LOGNDTR_FLOAT64_UPPER = np.array(8, np.float64)
_LOGNDTR_FLOAT32_UPPER = np.array(5, np.float32)


def ndtr(x: ArrayLike) -> Array:
  r"""Normal distribution function.

  JAX implementation of :obj:`scipy.special.ndtr`.

  Returns the area under the Gaussian probability density function, integrated
  from minus infinity to x:

  .. math::
    \begin{align}
    \mathrm{ndtr}(x) =&
      \ \frac{1}{\sqrt{2 \pi}}\int_{-\infty}^{x} e^{-\frac{1}{2}t^2} dt \\
    =&\ \frac{1}{2} (1 + \mathrm{erf}(\frac{x}{\sqrt{2}})) \\
    =&\ \frac{1}{2} \mathrm{erfc}(\frac{x}{\sqrt{2}})
    \end{align}

  Args:
    x: An array of type `float32`, `float64`.

  Returns:
    An array with `dtype=x.dtype`.

  Raises:
    TypeError: if `x` is not floating-type.
  """
  x = jnp.asarray(x)
  dtype = lax.dtype(x)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        "x.dtype={} is not supported, see docstring for supported types."
        .format(dtype))
  return _ndtr(x)


def _ndtr(x: ArrayLike) -> Array:
  """Implements ndtr core logic."""
  dtype = lax.dtype(x).type
  half_sqrt_2 = dtype(0.5) * np.sqrt(2., dtype=dtype)
  w = x * half_sqrt_2
  z = lax.abs(w)
  y = lax.select(lax.lt(z, half_sqrt_2),
                      dtype(1.) + lax.erf(w),
                      lax.select(lax.gt(w, dtype(0.)),
                                      dtype(2.) - lax.erfc(z),
                                      lax.erfc(z)))
  return dtype(0.5) * y


def ndtri(p: ArrayLike) -> Array:
  r"""The inverse of the CDF of the Normal distribution function.

  JAX implementation of :obj:`scipy.special.ndtri`.

  Returns `x` such that the area under the PDF from :math:`-\infty` to `x` is equal
  to `p`.

  A piece-wise rational approximation is done for the function.
  This is based on the implementation in netlib.

  Args:
    p: an array of type `float32`, `float64`.

  Returns:
    an array with `dtype=p.dtype`.

  Raises:
    TypeError: if `p` is not floating-type.
  """
  dtype = lax.dtype(p)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        "x.dtype={} is not supported, see docstring for supported types."
        .format(dtype))
  return _ndtri(p)


def _ndtri(p: ArrayLike) -> Array:
  """Implements ndtri core logic."""

  # Constants used in piece-wise rational approximations. Taken from the cephes
  # library:
  # https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
  p0 = list(reversed([-5.99633501014107895267E1,
                      9.80010754185999661536E1,
                      -5.66762857469070293439E1,
                      1.39312609387279679503E1,
                      -1.23916583867381258016E0]))
  q0 = list(reversed([1.0,
                      1.95448858338141759834E0,
                      4.67627912898881538453E0,
                      8.63602421390890590575E1,
                      -2.25462687854119370527E2,
                      2.00260212380060660359E2,
                      -8.20372256168333339912E1,
                      1.59056225126211695515E1,
                      -1.18331621121330003142E0]))
  p1 = list(reversed([4.05544892305962419923E0,
                      3.15251094599893866154E1,
                      5.71628192246421288162E1,
                      4.40805073893200834700E1,
                      1.46849561928858024014E1,
                      2.18663306850790267539E0,
                      -1.40256079171354495875E-1,
                      -3.50424626827848203418E-2,
                      -8.57456785154685413611E-4]))
  q1 = list(reversed([1.0,
                      1.57799883256466749731E1,
                      4.53907635128879210584E1,
                      4.13172038254672030440E1,
                      1.50425385692907503408E1,
                      2.50464946208309415979E0,
                      -1.42182922854787788574E-1,
                      -3.80806407691578277194E-2,
                      -9.33259480895457427372E-4]))
  p2 = list(reversed([3.23774891776946035970E0,
                      6.91522889068984211695E0,
                      3.93881025292474443415E0,
                      1.33303460815807542389E0,
                      2.01485389549179081538E-1,
                      1.23716634817820021358E-2,
                      3.01581553508235416007E-4,
                      2.65806974686737550832E-6,
                      6.23974539184983293730E-9]))
  q2 = list(reversed([1.0,
                      6.02427039364742014255E0,
                      3.67983563856160859403E0,
                      1.37702099489081330271E0,
                      2.16236993594496635890E-1,
                      1.34204006088543189037E-2,
                      3.28014464682127739104E-4,
                      2.89247864745380683936E-6,
                      6.79019408009981274425E-9]))

  dtype = lax.dtype(p).type
  shape = jnp.shape(p)

  def _create_polynomial(var, coeffs):
    """Compute n_th order polynomial via Horner's method."""
    coeffs = np.array(coeffs, dtype)
    if not coeffs.size:
      return jnp.zeros_like(var)
    return coeffs[0] + _create_polynomial(var, coeffs[1:]) * var


  maybe_complement_p = jnp.where(p > dtype(-np.expm1(-2.)), dtype(1.) - p, p)
  # Write in an arbitrary value in place of 0 for p since 0 will cause NaNs
  # later on. The result from the computation when p == 0 is not used so any
  # number that doesn't result in NaNs is fine.
  sanitized_mcp = jnp.where(
      maybe_complement_p == dtype(0.),
      jnp.full(shape, dtype(0.5)),
      maybe_complement_p)

  # Compute x for p > exp(-2): x/sqrt(2pi) = w + w**3 P0(w**2)/Q0(w**2).
  w = sanitized_mcp - dtype(0.5)
  ww = lax.square(w)
  x_for_big_p = w + w * ww * (_create_polynomial(ww, p0)
                              / _create_polynomial(ww, q0))
  x_for_big_p *= -dtype(np.sqrt(2. * np.pi))

  # Compute x for p <= exp(-2): x = z - log(z)/z - (1/z) P(1/z) / Q(1/z),
  # where z = sqrt(-2. * log(p)), and P/Q are chosen between two different
  # arrays based on whether p < exp(-32).
  z = lax.sqrt(dtype(-2.) * lax.log(sanitized_mcp))
  first_term = z - lax.log(z) / z
  second_term_small_p = (
      _create_polynomial(dtype(1.) / z, p2) /
      _create_polynomial(dtype(1.) / z, q2) / z)
  second_term_otherwise = (
      _create_polynomial(dtype(1.) / z, p1) /
      _create_polynomial(dtype(1.) / z, q1) / z)
  x_for_small_p = first_term - second_term_small_p
  x_otherwise = first_term - second_term_otherwise

  x = jnp.where(sanitized_mcp > dtype(np.exp(-2.)),
                x_for_big_p,
                jnp.where(z >= dtype(8.0), x_for_small_p, x_otherwise))

  x = jnp.where(p > dtype(1. - np.exp(-2.)), x, -x)
  infinity = jnp.full(shape, dtype(np.inf))
  x_fix_boundaries = jnp.where(
      p == dtype(0.0), -infinity, jnp.where(p == dtype(1.0), infinity, x))
  return x_fix_boundaries


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(1,))
def log_ndtr(x: ArrayLike, series_order: int = 3) -> Array:
  r"""Log Normal distribution function.

  JAX implementation of :obj:`scipy.special.log_ndtr`.

  For details of the Normal distribution function see `ndtr`.

  This function calculates :math:`\log(\mathrm{ndtr}(x))` by either calling
  :math:`\log(\mathrm{ndtr}(x))` or using an asymptotic series. Specifically:

  - For `x > upper_segment`, use the approximation `-ndtr(-x)` based on
    :math:`\log(1-x) \approx -x, x \ll 1`.
  - For `lower_segment < x <= upper_segment`, use the existing `ndtr` technique
    and take a log.
  - For `x <= lower_segment`, we use the series approximation of `erf` to compute
    the log CDF directly.

  The `lower_segment` is set based on the precision of the input:

  .. math::
    \begin{align}
    \mathit{lower\_segment} =&
      \ \begin{cases}
        -20 &  x.\mathrm{dtype}=\mathit{float64} \\
        -10 &  x.\mathrm{dtype}=\mathit{float32} \\
        \end{cases} \\
    \mathit{upper\_segment} =&
      \ \begin{cases}
        8&  x.\mathrm{dtype}=\mathit{float64} \\
        5&  x.\mathrm{dtype}=\mathit{float32} \\
        \end{cases}
    \end{align}


  When `x < lower_segment`, the `ndtr` asymptotic series approximation is:

  .. math::
    \begin{align}
     \mathrm{ndtr}(x) =&\  \mathit{scale} * (1 + \mathit{sum}) + R_N \\
     \mathit{scale}   =&\  \frac{e^{-0.5 x^2}}{-x \sqrt{2 \pi}} \\
     \mathit{sum}     =&\  \sum_{n=1}^N {-1}^n (2n-1)!! / (x^2)^n \\
     R_N     =&\  O(e^{-0.5 x^2} (2N+1)!! / |x|^{2N+3})
    \end{align}

  where :math:`(2n-1)!! = (2n-1) (2n-3) (2n-5) ...  (3) (1)` is a
  `double-factorial
  <https://en.wikipedia.org/wiki/Double_factorial>`_ operator.


  Args:
    x: an array of type `float32`, `float64`.
    series_order: Positive Python integer. Maximum depth to
      evaluate the asymptotic expansion. This is the `N` above.

  Returns:
    an array with `dtype=x.dtype`.

  Raises:
    TypeError: if `x.dtype` is not handled.
    TypeError: if `series_order` is a not Python `integer.`
    ValueError:  if `series_order` is not in `[0, 30]`.
  """
  if not isinstance(series_order, int):
    raise TypeError("series_order must be a Python integer.")
  if series_order < 0:
    raise ValueError("series_order must be non-negative.")
  if series_order > 30:
    raise ValueError("series_order must be <= 30.")

  x_arr = jnp.asarray(x)
  dtype = lax.dtype(x_arr)

  if dtype == jnp.float64:
    lower_segment: np.ndarray = _LOGNDTR_FLOAT64_LOWER
    upper_segment: np.ndarray = _LOGNDTR_FLOAT64_UPPER
  elif dtype == jnp.float32:
    lower_segment = _LOGNDTR_FLOAT32_LOWER
    upper_segment = _LOGNDTR_FLOAT32_UPPER
  else:
    raise TypeError(f"x.dtype={np.dtype(dtype)} is not supported.")

  # The basic idea here was ported from:
  #   https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
  # We copy the main idea, with a few changes
  # * For x >> 1, and X ~ Normal(0, 1),
  #     Log[P[X < x]] = Log[1 - P[X < -x]] approx -P[X < -x],
  #     which extends the range of validity of this function.
  # * We use one fixed series_order for all of 'x', rather than adaptive.
  # * Our docstring properly reflects that this is an asymptotic series, not a
  #   Taylor series. We also provided a correct bound on the remainder.
  # * We need to use the max/min in the _log_ndtr_lower arg to avoid nan when
  #   x=0. This happens even though the branch is unchosen because when x=0
  #   the gradient of a select involves the calculation 1*dy+0*(-inf)=nan
  #   regardless of whether dy is finite. Note that the minimum is a NOP if
  #   the branch is chosen.
  return jnp.where(
      lax.gt(x_arr, upper_segment),
      -_ndtr(-x_arr),  # log(1-x) ~= -x, x << 1
      jnp.where(lax.gt(x_arr, lower_segment),
                       lax.log(_ndtr(lax.max(x_arr, lower_segment))),
                       _log_ndtr_lower(lax.min(x_arr, lower_segment),
                                       series_order)))

def _log_ndtr_jvp(series_order, primals, tangents):
  (x,), (t,) = primals, tangents
  ans = log_ndtr(x, series_order=series_order)
  t_out = lax.mul(t, lax.exp(lax.sub(_norm_logpdf(x), ans)))
  return ans, t_out
log_ndtr.defjvp(_log_ndtr_jvp)

def _log_ndtr_lower(x, series_order):
  """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
  dtype = lax.dtype(x).type
  x_2 = lax.square(x)
  # Log of the term multiplying (1 + sum)
  log_scale = -dtype(0.5) * x_2 - lax.log(-x) - dtype(0.5 * np.log(2. * np.pi))
  return log_scale + lax.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
  """Calculates the asymptotic series used in log_ndtr."""
  dtype = lax.dtype(x).type
  if series_order <= 0:
    return np.array(1, dtype)
  x_2 = lax.square(x)
  even_sum = jnp.zeros_like(x)
  odd_sum = jnp.zeros_like(x)
  x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.
  for n in range(1, series_order + 1):
    y = np.array(_double_factorial(2 * n - 1), dtype) / x_2n
    if n % 2:
      odd_sum += y
    else:
      even_sum += y
    x_2n *= x_2
  return dtype(1.) + even_sum - odd_sum


def _double_factorial(n: int) -> np.ndarray:
  """The double factorial function for small Python integer `n`."""
  return np.prod(np.arange(n, 1, -2))


_norm_logpdf_constant = np.log(np.sqrt(2 * np.pi))

def _norm_logpdf(x):
  neg_half = _lax_const(x, -0.5)
  log_normalizer = _lax_const(x, _norm_logpdf_constant)
  return lax.sub(lax.mul(neg_half, lax.square(x)), log_normalizer)


def i0e(x: ArrayLike) -> Array:
  r"""Exponentially scaled modified bessel function of zeroth order.

  JAX implementation of :obj:`scipy.special.i0e`.

  .. math::

     \mathrm{i0e}(x) = e^{-|x|} I_0(x)

  where :math:`I_0(x)` is the modified Bessel function :func:`~jax.scipy.special.i0`.

  Args:
    x: array, real-valued

  Returns:
    array of bessel function values.

  See also:
    - :func:`jax.scipy.special.i0`
    - :func:`jax.scipy.special.i1`
    - :func:`jax.scipy.special.i1e`
  """
  x, = promote_args_inexact("i0e", x)
  return lax.bessel_i0e(x)


def i0(x: ArrayLike) -> Array:
  r"""Modified bessel function of zeroth order.

  JAX implementation of :obj:`scipy.special.i0`.

  .. math::

     \mathrm{i0}(x) = I_0(x) = \sum_{k=0}^\infty \frac{(x^2/4)^k}{(k!)^2}

  Args:
    x: array, real-valued

  Returns:
    array of bessel function values.

  See also:
    - :func:`jax.scipy.special.i0e`
    - :func:`jax.scipy.special.i1`
    - :func:`jax.scipy.special.i1e`
  """
  x, = promote_args_inexact("i0", x)
  return lax.mul(lax.exp(lax.abs(x)), lax.bessel_i0e(x))


def i1e(x: ArrayLike) -> Array:
  r"""Exponentially scaled modified bessel function of first order.

  JAX implementation of :obj:`scipy.special.i1e`.

  .. math::

     \mathrm{i1e}(x) = e^{-|x|} I_1(x)

  where :math:`I_1(x)` is the modified Bessel function :func:`~jax.scipy.special.i1`.

  Args:
    x: array, real-valued

  Returns:
    array of bessel function values

  See also:
    - :func:`jax.scipy.special.i0`
    - :func:`jax.scipy.special.i0e`
    - :func:`jax.scipy.special.i1`
  """
  x, = promote_args_inexact("i1e", x)
  return lax.bessel_i1e(x)


def i1(x: ArrayLike) -> Array:
  r"""Modified bessel function of first order.

  JAX implementation of :obj:`scipy.special.i1`.

  .. math::

     \mathrm{i1}(x) = I_1(x) = \frac{1}{2}x\sum_{k=0}^\infty\frac{(x^2/4)^k}{k!(k+1)!}

  Args:
    x: array, real-valued

  Returns:
    array of bessel function values

  See also:
    - :func:`jax.scipy.special.i0`
    - :func:`jax.scipy.special.i0e`
    - :func:`jax.scipy.special.i1e`
  """
  x, = promote_args_inexact("i1", x)
  return lax.mul(lax.exp(lax.abs(x)), lax.bessel_i1e(x))

def _bessel_jn_scan_body_fun(carry, k):
  f0, f1, bs, z = carry
  f = 2.0 * (k + 1.0) * f1 / z - f0

  def true_fn_update_bs(u):
    bs, f = u
    return bs + 2.0 * f

  def false_fn_update_bs(u):
    bs, _ = u
    return bs

  bs = lax.cond(jnp.mod(k, 2) == 0, true_fn_update_bs,
                false_fn_update_bs, operand=(bs, f))

  f0 = f1
  f1 = f
  return (f0, f1, bs, z), f


def _bessel_jn(z: ArrayLike, *, v: int, n_iter: int=50) -> Array:
  f0 = _lax_const(z, 0.0)
  f1 = _lax_const(z, 1E-16)
  f = _lax_const(z, 0.0)
  bs = _lax_const(z, 0.0)

  (_, _, bs, _), j_vals = lax.scan(
      f=_bessel_jn_scan_body_fun, init=(f0, f1, bs, z),
      xs=lax.iota(lax.dtype(z), n_iter+1), reverse=True)

  f = j_vals[0]  # Use the value at the last iteration.
  j_vals = j_vals[:v+1]
  j_vals = j_vals / (bs - f)

  return j_vals


@partial(jit, static_argnames=["v", "n_iter"])
def bessel_jn(z: ArrayLike, *, v: int, n_iter: int=50) -> Array:
  """Bessel function of the first kind of integer order and real argument.

  Reference:
  Shanjie Zhang and Jian-Ming Jin. Computation of special functions.
  Wiley-Interscience, 1996.

  Args:
    z: The sampling point(s) at which the Bessel function of the first kind are
      computed.
    v: The order (int) of the Bessel function.
    n_iter: The number of iterations required for updating the function
      values. As a rule of thumb, `n_iter` is the smallest nonnegative integer
      that satisfies the condition
      `int(0.5 * log10(6.28 + n_iter) - n_iter *  log10(1.36 + abs(z) / n_iter)) > 20`.
      Details in `BJNDD` (https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.f)

  Returns:
    An array of shape `(v+1, *z.shape)` containing the values of the Bessel
    function of orders 0, 1, ..., v. The return type matches the type of `z`.

  Raises:
    TypeError if `v` is not integer.
    ValueError if elements of array `z` are not float.
  """
  z = jnp.asarray(z)
  z, = promote_dtypes_inexact(z)
  z_dtype = lax.dtype(z)
  if dtypes.issubdtype(z_dtype, complex):
    raise ValueError("complex input not supported.")

  v = core.concrete_or_error(operator.index, v, 'Argument v of bessel_jn.')
  n_iter = core.concrete_or_error(int, n_iter, 'Argument n_iter of bessel_jn.')

  bessel_jn_fun = partial(_bessel_jn, v=v, n_iter=n_iter)
  for _ in range(z.ndim):
    bessel_jn_fun = vmap(bessel_jn_fun)
  return jnp.moveaxis(bessel_jn_fun(z), -1, 0)


def _gen_recurrence_mask(
    l_max: int, is_normalized: bool, dtype: Any
) -> tuple[Array, Array]:
  """Generates a mask for recurrence relation on the remaining entries.

  The remaining entries are with respect to the diagonal and offdiagonal
  entries.

  Args:
    l_max: see `gen_normalized_legendre`.
    is_normalized: True if the recurrence mask is used by normalized associated
      Legendre functions.

  Returns:
    Arrays representing the mask used by the recurrence relations.
  """

  # Computes all coefficients.
  m_mat, l_mat = jnp.meshgrid(
    jnp.arange(l_max + 1, dtype=dtype),
    jnp.arange(l_max + 1, dtype=dtype),
    indexing='ij')
  if is_normalized:
    c0 = l_mat * l_mat
    c1 = m_mat * m_mat
    c2 = 2.0 * l_mat
    c3 = (l_mat - 1.0) * (l_mat - 1.0)
    d0 = jnp.sqrt((4.0 * c0 - 1.0) / (c0 - c1))
    d1 = jnp.sqrt(((c2 + 1.0) * (c3 - c1)) / ((c2 - 3.0) * (c0 - c1)))
  else:
    d0 = (2.0 * l_mat - 1.0) / (l_mat - m_mat)
    d1 = (l_mat + m_mat - 1.0) / (l_mat - m_mat)

  d0_mask_indices = jnp.triu_indices(l_max + 1, 1)
  d1_mask_indices = jnp.triu_indices(l_max + 1, 2)
  d_zeros = jnp.zeros((l_max + 1, l_max + 1), dtype=dtype)
  d0_mask = d_zeros.at[d0_mask_indices].set(d0[d0_mask_indices])
  d1_mask = d_zeros.at[d1_mask_indices].set(d1[d1_mask_indices])

  # Creates a 3D mask that contains 1s on the diagonal plane and 0s elsewhere.
  # i = jnp.arange(l_max + 1)[:, None, None]
  # j = jnp.arange(l_max + 1)[None, :, None]
  # k = jnp.arange(l_max + 1)[None, None, :]
  i, j, k = jnp.ogrid[:l_max + 1, :l_max + 1, :l_max + 1]
  mask = (i + j - k == 0).astype(dtype)

  d0_mask_3d = jnp.einsum('jk,ijk->ijk', d0_mask, mask)
  d1_mask_3d = jnp.einsum('jk,ijk->ijk', d1_mask, mask)

  return (d0_mask_3d, d1_mask_3d)


@partial(jit, static_argnums=(2))
def _gen_derivatives(p: Array,
                     x: Array,
                     is_normalized: bool) -> Array:
  """Generates derivatives of associated Legendre functions of the first kind.

  Args:
    p: The 3D array containing the values of associated Legendre functions; the
      dimensions are in the sequence of order (m), degree (l), and evaluation
      points.
    x: A vector of type `float32` or `float64` containing the sampled points.
    is_normalized: True if the associated Legendre functions are normalized.
  Returns:
    The 3D array representing the derivatives of associated Legendre functions
    of the first kind.
  """

  num_m, num_l, num_x = p.shape

  # p_{l-1}^m.
  p_m_lm1 = jnp.pad(p, ((0, 0), (1, 0), (0, 0)))[:, :num_l, :]

  # p_{l-1}^{m+2}.
  p_mp2_lm1 = jnp.pad(p_m_lm1, ((0, 2), (0, 0), (0, 0)))[2:num_m + 2, :, :]

  # p_{l-1}^{m-2}.
  p_mm2_lm1 = jnp.pad(p_m_lm1, ((2, 0), (0, 0), (0, 0)))[:num_m, :, :]

  # Derivative computation requires negative orders.
  if is_normalized:
    raise NotImplementedError(
        'Negative orders for normalization is not implemented yet.')
  else:
    if num_l > 1:
      l_vec = jnp.arange(1, num_l - 1, dtype=x.dtype)
      p_p1 = p[1, 1:num_l - 1, :]
      coeff = -1.0 / ((l_vec + 1) * l_vec)
      update_p_p1 = jnp.einsum('i,ij->ij', coeff, p_p1)
      p_mm2_lm1 = p_mm2_lm1.at[1, 2:num_l, :].set(update_p_p1)

    if num_l > 2:
      l_vec = jnp.arange(2, num_l - 1, dtype=x.dtype)
      p_p2 = p[2, 2:num_l - 1, :]
      coeff = 1.0 / ((l_vec + 2) * (l_vec + 1) * l_vec * (l_vec - 1))
      update_p_p2 = jnp.einsum('i,ij->ij', coeff, p_p2)
      p_mm2_lm1 = p_mm2_lm1.at[0, 3:num_l, :].set(update_p_p2)

  m_mat, l_mat = jnp.meshgrid(
    jnp.arange(num_m, dtype=x.dtype),
    jnp.arange(num_l, dtype=x.dtype),
    indexing='ij')

  coeff_zeros = jnp.zeros((num_m, num_l), dtype=x.dtype)
  upper_0_indices = jnp.triu_indices(num_m, 0, num_l)
  zero_vec = jnp.zeros((num_l,), dtype=x.dtype)

  a0 = -0.5 / (m_mat - 1.0)
  a0_masked = coeff_zeros.at[upper_0_indices].set(a0[upper_0_indices])
  a0_masked = a0_masked.at[1, :].set(zero_vec)

  b0 = l_mat + m_mat
  c0 = a0 * (b0 - 2.0) * (b0 - 1.0)
  c0_masked = coeff_zeros.at[upper_0_indices].set(c0[upper_0_indices])
  c0_masked = c0_masked.at[1, :].set(zero_vec)

  # p_l^{m-1}.
  p_mm1_l = (jnp.einsum('ij,ijk->ijk', a0_masked, p_m_lm1) +
             jnp.einsum('ij,ijk->ijk', c0_masked, p_mm2_lm1))

  d0 = -0.5 / (m_mat + 1.0)
  d0_masked = coeff_zeros.at[upper_0_indices].set(d0[upper_0_indices])
  e0 = d0 * b0 * (b0 + 1.0)
  e0_masked = coeff_zeros.at[upper_0_indices].set(e0[upper_0_indices])

  # p_l^{m+1}.
  p_mp1_l = (jnp.einsum('ij,ijk->ijk', d0_masked, p_mp2_lm1) +
             jnp.einsum('ij,ijk->ijk', e0_masked, p_m_lm1))

  f0 = b0 * (l_mat - m_mat + 1.0) / 2.0
  f0_masked = coeff_zeros.at[upper_0_indices].set(f0[upper_0_indices])
  p_derivative = jnp.einsum('ij,ijk->ijk', f0_masked, p_mm1_l) - 0.5 * p_mp1_l

  # Special treatment of the singularity at m = 1.
  if num_m > 1:
    l_vec = jnp.arange(num_l, dtype=p.dtype)
    g0 = jnp.einsum('i,ij->ij', (l_vec + 1) * l_vec, p[0, :, :])
    if num_l > 2:
      g0 = g0 -  p[2, :, :]
    p_derivative_m0 = jnp.einsum('j,ij->ij', 0.5 / jnp.sqrt(1 - x * x), g0)
    p_derivative = p_derivative.at[1, :, :].set(p_derivative_m0)
    p_derivative = p_derivative.at[1, 0, :].set(0)

  return p_derivative


@partial(jit, static_argnums=(0, 2))
def _gen_associated_legendre(l_max: int,
                             x: Array,
                             is_normalized: bool) -> Array:
  r"""Computes associated Legendre functions (ALFs) of the first kind.

  The ALFs of the first kind are used in spherical harmonics. The spherical
  harmonic of degree `l` and order `m` can be written as
  `Y_l^m(θ, φ) = N_l^m * P_l^m(cos(θ)) * exp(i m φ)`, where `N_l^m` is the
  normalization factor and θ and φ are the colatitude and longitude,
  respectively. `N_l^m` is chosen in the way that the spherical harmonics form
  a set of orthonormal basis functions of L^2(S^2). For the computational
  efficiency of spherical harmonics transform, the normalization factor is
  used in the computation of the ALFs. In addition, normalizing `P_l^m`
  avoids overflow/underflow and achieves better numerical stability. Three
  recurrence relations are used in the computation.

  Args:
    l_max: The maximum degree of the associated Legendre function. Both the
      degrees and orders are `[0, 1, 2, ..., l_max]`.
    x: A vector of type `float32`, `float64` containing the sampled points in
      spherical coordinates, at which the ALFs are computed; `x` is essentially
      `cos(θ)`. For the numerical integration used by the spherical harmonics
      transforms, `x` contains the quadrature points in the interval of
      `[-1, 1]`. There are several approaches to provide the quadrature points:
      Gauss-Legendre method (`scipy.special.roots_legendre`), Gauss-Chebyshev
      method (`scipy.special.roots_chebyu`), and Driscoll & Healy
      method (Driscoll, James R., and Dennis M. Healy. "Computing Fourier
      transforms and convolutions on the 2-sphere." Advances in applied
      mathematics 15, no. 2 (1994): 202-250.). The Gauss-Legendre quadrature
      points are nearly equal-spaced along θ and provide exact discrete
      orthogonality, (P^m)^T W P_m = I, where `T` represents the transpose
      operation, `W` is a diagonal matrix containing the quadrature weights,
      and `I` is the identity matrix. The Gauss-Chebyshev points are equally
      spaced, which only provide approximate discrete orthogonality. The
      Driscoll & Healy quadrature points are equally spaced and provide the
      exact discrete orthogonality. The number of sampling points is required to
      be twice as the number of frequency points (modes) in the Driscoll & Healy
      approach, which enables FFT and achieves a fast spherical harmonics
      transform.
    is_normalized: True if the associated Legendre functions are normalized.
      With normalization, `N_l^m` is applied such that the spherical harmonics
      form a set of orthonormal basis functions of L^2(S^2).

  Returns:
    The 3D array of shape `(l_max + 1, l_max + 1, len(x))` containing the values
    of the ALFs at `x`; the dimensions in the sequence of order, degree, and
    evaluation points.
  """
  p = jnp.zeros((l_max + 1, l_max + 1, x.shape[0]), dtype=x.dtype)

  a_idx = jnp.arange(1, l_max + 1, dtype=x.dtype)
  b_idx = jnp.arange(l_max, dtype=x.dtype)
  if is_normalized:
    initial_value: ArrayLike = 0.5 / jnp.sqrt(jnp.pi)  # The initial value p(0,0).
    f_a = jnp.cumprod(-1 * jnp.sqrt(1.0 + 0.5 / a_idx))
    f_b = jnp.sqrt(2.0 * b_idx + 3.0)
  else:
    initial_value = 1.0  # The initial value p(0,0).
    f_a = jnp.cumprod(1.0 - 2.0 * a_idx)
    f_b = 2.0 * b_idx + 1.0

  p = p.at[(0, 0)].set(initial_value)

  # Compute the diagonal entries p(l,l) with recurrence.
  y = jnp.cumprod(
      jnp.broadcast_to(jnp.sqrt(1.0 - x * x), (l_max, x.shape[0])),
      axis=0)
  p_diag = initial_value * jnp.einsum('i,ij->ij', f_a, y)
  diag_indices = jnp.diag_indices(l_max + 1)
  p = p.at[(diag_indices[0][1:], diag_indices[1][1:])].set(p_diag)

  # Compute the off-diagonal entries with recurrence.
  p_offdiag = jnp.einsum('ij,ij->ij',
                         jnp.einsum('i,j->ij', f_b, x),
                         p[jnp.diag_indices(l_max)])
  offdiag_indices = (diag_indices[0][:l_max], diag_indices[1][:l_max] + 1)
  p = p.at[offdiag_indices].set(p_offdiag)

  # Compute the remaining entries with recurrence.
  d0_mask_3d, d1_mask_3d = _gen_recurrence_mask(
      l_max, is_normalized=is_normalized, dtype=x.dtype)

  def body_fun(i, p_val):
    coeff_0 = d0_mask_3d[i]
    coeff_1 = d1_mask_3d[i]
    h = (jnp.einsum('ij,ijk->ijk',
                    coeff_0,
                    jnp.einsum(
                        'ijk,k->ijk', jnp.roll(p_val, shift=1, axis=1), x)) -
         jnp.einsum('ij,ijk->ijk', coeff_1, jnp.roll(p_val, shift=2, axis=1)))
    p_val = p_val + h
    return p_val

  # TODO(jakevdp): use some sort of fixed-point procedure here instead?
  p = p.astype(jnp.result_type(p, x, d0_mask_3d))
  if l_max > 1:
    p = lax.fori_loop(lower=2, upper=l_max+1, body_fun=body_fun, init_val=p)

  return p


def lpmn(m: int, n: int, z: Array) -> tuple[Array, Array]:
  """The associated Legendre functions (ALFs) of the first kind.

  Args:
    m: The maximum order of the associated Legendre functions.
    n: The maximum degree of the associated Legendre function, often called
      `l` in describing ALFs. Both the degrees and orders are
      `[0, 1, 2, ..., l_max]`, where `l_max` denotes the maximum degree.
    z: A vector of type `float32` or `float64` containing the sampling
      points at which the ALFs are computed.

  Returns:
    A 2-tuple of 3D arrays of shape `(l_max + 1, l_max + 1, len(z))` containing
    the values and derivatives of the associated Legendre functions of the
    first kind. The return type matches the type of `z`.

  Raises:
    TypeError if elements of array `z` are not in (float32, float64).
    ValueError if array `z` is not 1D.
    NotImplementedError if `m!=n`.
  """
  dtype = lax.dtype(z)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        'z.dtype={} is not supported, see docstring for supported types.'
        .format(dtype))

  if z.ndim != 1:
    raise ValueError('z must be a 1D array.')

  m = core.concrete_or_error(int, m, 'Argument m of lpmn.')
  n = core.concrete_or_error(int, n, 'Argument n of lpmn.')

  if m != n:
    raise NotImplementedError('Computations for m!=n are not yet supported.')

  l_max = n
  is_normalized = False
  p_vals = _gen_associated_legendre(l_max, z, is_normalized)
  p_derivatives = _gen_derivatives(p_vals, z, is_normalized)

  return (p_vals, p_derivatives)


def lpmn_values(m: int, n: int, z: Array, is_normalized: bool) -> Array:
  r"""The associated Legendre functions (ALFs) of the first kind.

  Unlike `lpmn`, this function only computes the values of ALFs.
  The ALFs of the first kind can be used in spherical harmonics. The
  spherical harmonic of degree `l` and order `m` can be written as
  :math:`Y_l^m(\theta, \phi) = N_l^m * P_l^m(\cos \theta) * \exp(i m \phi)`,
  where :math:`N_l^m` is the normalization factor and θ and φ are the
  colatitude and longitude, respectively. :math:`N_l^m` is chosen in the
  way that the spherical harmonics form a set of orthonormal basis function
  of :math:`L^2(S^2)`. Normalizing :math:`P_l^m` avoids overflow/underflow
  and achieves better numerical stability.

  Args:
    m: The maximum order of the associated Legendre functions.
    n: The maximum degree of the associated Legendre function, often called
      `l` in describing ALFs. Both the degrees and orders are
      `[0, 1, 2, ..., l_max]`, where `l_max` denotes the maximum degree.
    z: A vector of type `float32` or `float64` containing the sampling
      points at which the ALFs are computed.
    is_normalized: True if the associated Legendre functions are normalized.
      With normalization, :math:`N_l^m` is applied such that the spherical
      harmonics form a set of orthonormal basis functions of :math:`L^2(S^2)`.

  Returns:
    A 3D array of shape `(l_max + 1, l_max + 1, len(z))` containing
    the values of the associated Legendre functions of the first kind. The
    return type matches the type of `z`.

  Raises:
    TypeError if elements of array `z` are not in (float32, float64).
    ValueError if array `z` is not 1D.
    NotImplementedError if `m!=n`.
  """
  dtype = lax.dtype(z)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        'z.dtype={} is not supported, see docstring for supported types.'
        .format(dtype))

  if z.ndim != 1:
    raise ValueError('z must be a 1D array.')

  m = core.concrete_or_error(int, m, 'Argument m of lpmn.')
  n = core.concrete_or_error(int, n, 'Argument n of lpmn.')

  if m != n:
    raise NotImplementedError('Computations for m!=n are not yet supported.')

  l_max = n

  return _gen_associated_legendre(l_max, z, is_normalized)



@partial(jit, static_argnums=(4,))
def _sph_harm(n: Array,
              m: Array,
              theta: Array,
              phi: Array,
              n_max: int) -> Array:
  """Computes the spherical harmonics."""

  cos_colatitude = jnp.cos(theta)

  legendre = _gen_associated_legendre(n_max, cos_colatitude, True)
  legendre_val = legendre.at[abs(m), n, jnp.arange(len(n))].get(mode="clip")

  angle = abs(m) * phi
  vandermonde = lax.complex(jnp.cos(angle), jnp.sin(angle))
  harmonics = lax.complex(legendre_val * jnp.real(vandermonde),
                          legendre_val * jnp.imag(vandermonde))

  # Negative order.
  harmonics = jnp.where(m < 0,
                        (-1.0)**abs(m) * jnp.conjugate(harmonics),
                        harmonics)

  return harmonics


def sph_harm_y(n: Array,
               m: Array,
               theta: Array,
               phi: Array,
               diff_n: int | None = None,
               n_max: int | None = None) -> Array:
  r"""Computes the spherical harmonics.

  The JAX version has one extra argument `n_max`, the maximum value in `n`.

  The spherical harmonic of degree `n` and order `m` can be written as
  :math:`Y_n^m(\theta, \phi) = N_n^m * P_n^m(\cos \theta) * \exp(i m \phi)`,
  where :math:`N_n^m = \sqrt{\frac{\left(2n+1\right) \left(n-m\right)!}
  {4 \pi \left(n+m\right)!}}` is the normalization factor and :math:`\theta` and
  :math:`\phi` are the colatitude and longitude, respectively. :math:`N_n^m` is
  chosen in the way that the spherical harmonics form a set of orthonormal basis
  functions of :math:`L^2(S^2)`.

  Args:
    n: The degree of the harmonic; must have `n >= 0`. The standard notation for
      degree in descriptions of spherical harmonics is `l (lower case L)`. We
      use `n` here to be consistent with `scipy.special.sph_harm_y`. Return
      values for `n < 0` are undefined.
    m: The order of the harmonic; must have `|m| <= n`. Return values for
      `|m| > n` are undefined.
    theta: The polar (colatitudinal) coordinate; must be in [0, pi].
    phi: The azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
    diff_n: Unsupported by JAX.
    n_max: The maximum degree `max(n)`. If the supplied `n_max` is not the true
      maximum value of `n`, the results are clipped to `n_max`. For example,
      `sph_harm(m=jnp.array([2]), n=jnp.array([10]), theta, phi, n_max=6)`
      actually returns
      `sph_harm(m=jnp.array([2]), n=jnp.array([6]), theta, phi, n_max=6)`
  Returns:
    A 1D array containing the spherical harmonics at (m, n, theta, phi).
  """
  if diff_n is not None:
    raise NotImplementedError(
        "The 'diff_n' argument to jax.scipy.special.sph_harm_y is not supported.")

  if jnp.isscalar(theta):
    theta = jnp.array([theta])

  if n_max is None:
    n_max = np.max(n)
  n_max = core.concrete_or_error(
      int, n_max, 'The `n_max` argument of `jnp.scipy.special.sph_harm` must '
      'be statically specified to use `sph_harm` within JAX transformations.')

  return _sph_harm(n, m, theta, phi, n_max)


def sph_harm(m: Array,
             n: Array,
             theta: Array,
             phi: Array,
             n_max: int | None = None) -> Array:
  r"""Computes the spherical harmonics.

  Note:
    This function is deprecated, and :func:`~jax.scipy.special.sph_harm_y`
    should be used instead, noting that the order of ``m`` and ``n`` are
    reversed, and definitions of ``theta`` and ``phi`` are swapped.

  The JAX version has one extra argument `n_max`, the maximum value in `n`.

  The spherical harmonic of degree `n` and order `m` can be written as
  :math:`Y_n^m(\theta, \phi) = N_n^m * P_n^m(\cos \phi) * \exp(i m \theta)`,
  where :math:`N_n^m = \sqrt{\frac{\left(2n+1\right) \left(n-m\right)!}
  {4 \pi \left(n+m\right)!}}` is the normalization factor and :math:`\phi` and
  :math:`\theta` are the colatitude and longitude, respectively. :math:`N_n^m` is
  chosen in the way that the spherical harmonics form a set of orthonormal basis
  functions of :math:`L^2(S^2)`.

  Args:
    m: The order of the harmonic; must have `|m| <= n`. Return values for
      `|m| > n` are undefined.
    n: The degree of the harmonic; must have `n >= 0`. The standard notation for
      degree in descriptions of spherical harmonics is `l (lower case L)`. We
      use `n` here to be consistent with `scipy.special.sph_harm`. Return
      values for `n < 0` are undefined.
    theta: The azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
    phi: The polar (colatitudinal) coordinate; must be in [0, pi].
    n_max: The maximum degree `max(n)`. If the supplied `n_max` is not the true
      maximum value of `n`, the results are clipped to `n_max`. For example,
      `sph_harm(m=jnp.array([2]), n=jnp.array([10]), theta, phi, n_max=6)`
      actually returns
      `sph_harm(m=jnp.array([2]), n=jnp.array([6]), theta, phi, n_max=6)`
  Returns:
    A 1D array containing the spherical harmonics at (m, n, theta, phi).
  """
  # Added 2025-01-06.
  # TODO(dfm): Remove after deprecation period.
  deprecations.warn(
      "jax-scipy-special-sph-harm",
      ("jax.scipy.special.sph_harm is deprecated. Please use "
       "jax.scipy.special.sph_harm_y instead, noting that the order of `m` and "
       "`n` are reversed, and definitions of `theta` and `phi` are swapped."),
      stacklevel=2,
  )
  return sph_harm_y(n, m, phi, theta, n_max=n_max)


# exponential integrals
# these algorithms are ported over from the files ei.c and expn.c in the Cephes mathematical library.
# https://fossies.org/dox/cephes-math-28/ei_8c_source.html
# https://fossies.org/dox/cephes-math-28/expn_8c_source.html


def _expint1(x: Array) -> Array:
  # 0 < x <= 2
  A = [
    -5.350447357812542947283e0,
    2.185049168816613393830e2,
    -4.176572384826693777058e3,
    5.541176756393557601232e4,
    -3.313381331178144034309e5,
    1.592627163384945414220e6,
  ]
  B = [
    1.0,
    -5.250547959112862969197e1,
    1.259616186786790571525e3,
    -1.756549581973534652631e4,
    1.493062117002725991967e5,
    -7.294949239640527645655e5,
    1.592627163384945429726e6,
  ]
  A_arr = jnp.array(A, dtype=x.dtype)
  B_arr = jnp.array(B, dtype=x.dtype)
  f = jnp.polyval(A_arr, x) / jnp.polyval(B_arr, x)
  return x * f + jnp.euler_gamma + jnp.log(x)


def _eval_expint_k(A: list[float], B: list[float], x: Array) -> Array:
  # helper function for all subsequent intervals
  A_arr = jnp.array(A, dtype=x.dtype)
  B_arr = jnp.array(B, dtype=x.dtype)
  one = _lax_const(x, 1.0)
  w = one / x
  f = jnp.polyval(A_arr, w) / jnp.polyval(B_arr, w)
  f = w * f + one
  return jnp.exp(x) * w * f


def _expint2(x: Array) -> Array:
  # 2 <= x < 4
  A = [
    1.981808503259689673238e-2,
    -1.271645625984917501326e0,
    -2.088160335681228318920e0,
    2.755544509187936721172e0,
    -4.409507048701600257171e-1,
    4.665623805935891391017e-2,
    -1.545042679673485262580e-3,
    7.059980605299617478514e-5,
  ]
  B = [
    1.0,
    1.476498670914921440652e0,
    5.629177174822436244827e-1,
    1.699017897879307263248e-1,
    2.291647179034212017463e-2,
    4.450150439728752875043e-3,
    1.727439612206521482874e-4,
    3.953167195549672482304e-5,
  ]
  return _eval_expint_k(A, B, x)


def _expint3(x: Array) -> Array:
  # 4 <= x <= 8
  A = [
    -1.373215375871208729803e0,
    -7.084559133740838761406e-1,
    1.580806855547941010501e0,
    -2.601500427425622944234e-1,
    2.994674694113713763365e-2,
    -1.038086040188744005513e-3,
    4.371064420753005429514e-5,
    2.141783679522602903795e-6,
  ]
  B = [
    1.0,
    8.585231423622028380768e-1,
    4.483285822873995129957e-1,
    7.687932158124475434091e-2,
    2.449868241021887685904e-2,
    8.832165941927796567926e-4,
    4.590952299511353531215e-4,
    -4.729848351866523044863e-6,
    2.665195537390710170105e-6,
  ]
  return _eval_expint_k(A, B, x)


def _expint4(x: Array) -> Array:
  # 8 <= x <= 16
  A = [
    -2.106934601691916512584e0,
    1.732733869664688041885e0,
    -2.423619178935841904839e-1,
    2.322724180937565842585e-2,
    2.372880440493179832059e-4,
    -8.343219561192552752335e-5,
    1.363408795605250394881e-5,
    -3.655412321999253963714e-7,
    1.464941733975961318456e-8,
    6.176407863710360207074e-10,
  ]
  B = [
    1.0,
    -2.298062239901678075778e-1,
    1.105077041474037862347e-1,
    -1.566542966630792353556e-2,
    2.761106850817352773874e-3,
    -2.089148012284048449115e-4,
    1.708528938807675304186e-5,
    -4.459311796356686423199e-7,
    1.394634930353847498145e-8,
    6.150865933977338354138e-10,
  ]
  return _eval_expint_k(A, B, x)


def _expint5(x):
  # 16 <= x <= 32
  A = [
    -2.458119367674020323359e-1,
    -1.483382253322077687183e-1,
    7.248291795735551591813e-2,
    -1.348315687380940523823e-2,
    1.342775069788636972294e-3,
    -7.942465637159712264564e-5,
    2.644179518984235952241e-6,
    -4.239473659313765177195e-8,
  ]
  B = [
    1.0,
    -1.044225908443871106315e-1,
    -2.676453128101402655055e-1,
    9.695000254621984627876e-2,
    -1.601745692712991078208e-2,
    1.496414899205908021882e-3,
    -8.462452563778485013756e-5,
    2.728938403476726394024e-6,
    -4.239462431819542051337e-8,
  ]
  return _eval_expint_k(A, B, x)


def _expint6(x):
  # 32 <= x <= 64
  A = [
    1.212561118105456670844e-1,
    -5.823133179043894485122e-1,
    2.348887314557016779211e-1,
    -3.040034318113248237280e-2,
    1.510082146865190661777e-3,
    -2.523137095499571377122e-5,
  ]
  B = [
    1.0,
    -1.002252150365854016662e0,
    2.928709694872224144953e-1,
    -3.337004338674007801307e-2,
    1.560544881127388842819e-3,
    -2.523137093603234562648e-5,
  ]
  return _eval_expint_k(A, B, x)


def _expint7(x):
  # x > 64
  A = [
    -7.657847078286127362028e-1,
    6.886192415566705051750e-1,
    -2.132598113545206124553e-1,
    3.346107552384193813594e-2,
    -3.076541477344756050249e-3,
    1.747119316454907477380e-4,
    -6.103711682274170530369e-6,
    1.218032765428652199087e-7,
    -1.086076102793290233007e-9,
  ]
  B = [
    1.0,
    -1.888802868662308731041e0,
    1.066691687211408896850e0,
    -2.751915982306380647738e-1,
    3.930852688233823569726e-2,
    -3.414684558602365085394e-3,
    1.866844370703555398195e-4,
    -6.345146083130515357861e-6,
    1.239754287483206878024e-7,
    -1.086076102793126632978e-9,
  ]
  return _eval_expint_k(A, B, x)


def _expi_pos(x: Array) -> Array:
  # x >= 0
  _c = _lax_const
  conds = [(_c(x, 0) < x) & (x <= _c(x, 2))] + [
    (_c(x, 2 ** i) < x) & (x <= _c(x, 2 ** (i + 1))) for i in range(1, 6)
  ]
  return jnp.piecewise(
    x,
    conds,
    [_expint1, _expint2, _expint3, _expint4, _expint5, _expint6, _expint7],
  )

def _expi_neg(x: Array) -> Array:
  # x < 0
  return -exp1(-x)

@custom_derivatives.custom_jvp
@jit
def expi(x: ArrayLike) -> Array:
  r"""Exponential integral function.

  JAX implementation of :obj:`scipy.special.expi`

  .. math::

     \mathrm{expi}(x) = \int_{-\infty}^x \frac{e^t}{t} \mathrm{d}t

  Args:
    x: arraylike, real-valued

  Returns:
    array of expi values

  See also:
    - :func:`jax.scipy.special.expn`
    - :func:`jax.scipy.special.exp1`
  """
  x_arr, = promote_args_inexact("expi", x)
  return jnp.piecewise(x_arr, [x_arr < 0], [_expi_neg, _expi_pos])


@expi.defjvp
@jit
def expi_jvp(primals, tangents):
  (x,) = primals
  (x_dot,) = tangents
  return expi(x), jnp.exp(x) / x * x_dot


def _expn1(n: Array, x: Array) -> Array:
  # exponential integral En
  _c = _lax_const
  MACHEP = jnp.finfo(x.dtype).eps

  zero = _c(x, 0.0)
  one = _c(x, 1.0)
  psi = -jnp.euler_gamma - jnp.log(x)
  psi = lax.fori_loop(_c(n, 1), n, lambda i, psi: psi + one / i, psi)
  n1 = jnp.where(n == _c(n, 1), one + one, n)
  init = dict(
    x=x,
    z=-x,
    xk=zero,
    yk=one,
    pk=one - n,
    ans=jnp.where(n == _c(n, 1), zero, one / (one - n1)),
    t=jnp.inf,
  )

  def body(d):
    d["xk"] += one
    d["yk"] *= d["z"] / d["xk"]
    d["pk"] += one
    d["ans"] += jnp.where(d["pk"] != zero, d["yk"] / d["pk"], zero)
    d["t"] = jnp.where(d["ans"] != zero, abs(d["yk"] / d["ans"]), one)
    return d

  def cond(d):
    return (d["x"] > _c(d["x"], 0.0)) & (d["t"] > MACHEP)

  d = lax.while_loop(cond, body, init)
  t = n
  r = n - _c(n, 1)
  return d["z"] ** r * psi / jnp.exp(gammaln(t)) - d["ans"]


def _expn2(n: Array, x: Array) -> Array:
  # x > 1.
  _c = _lax_const
  BIG = _c(x, 1.44115188075855872e17)
  MACHEP = jnp.finfo(BIG.dtype).eps  # ?
  zero = _c(x, 0.0)
  one = _c(x, 1.0)

  init = dict(
    k=_c(n, 1),
    pkm2=one,
    qkm2=x,
    pkm1=one,
    qkm1=x + n,
    ans=one / (x + n),
    t=_c(x, jnp.inf),
    r=zero,
    x=x,
  )

  def body(d):
    x = d["x"]
    d["k"] += _c(d["k"], 1)
    k = d["k"]
    odd = k % _c(k, 2) == _c(k, 1)
    yk = jnp.where(odd, one, x)
    xk = jnp.where(odd, n + (k - _c(k, 1)) / _c(k, 2), k / _c(k, 2))
    pk = d["pkm1"] * yk + d["pkm2"] * xk
    qk = d["qkm1"] * yk + d["qkm2"] * xk
    nz = qk != zero
    d["r"] = r = jnp.where(nz, pk / qk, d["r"])
    d["t"] = jnp.where(nz, abs((d["ans"] - r) / r), one)
    d["ans"] = jnp.where(nz, r, d["ans"])
    d["pkm2"] = d["pkm1"]
    d["pkm1"] = pk
    d["qkm2"] = d["qkm1"]
    d["qkm1"] = qk
    is_big = abs(pk) > BIG
    for s in "pq":
      for i in "12":
        key = s + "km" + i
        d[key] = jnp.where(is_big, d[key] / BIG, d[key])
    return d

  def cond(d):
    return (d["x"] > _c(d["k"], 0)) & (d["t"] > MACHEP)

  d = lax.while_loop(cond, body, init)
  return d["ans"] * jnp.exp(-x)


def _expn3(n: Array, x: Array) -> Array:
  # n >= 5000
  _c = _lax_const
  one = _c(x, 1.0)
  xk = x + n
  yk = one / (xk * xk)
  t = n
  ans = yk * t * (_c(x, 6) * x * x - _c(x, 8) * t * x + t * t)
  ans = yk * (ans + t * (t - _c(x, 2) * x))
  ans = yk * (ans + t)
  return (ans + one) * jnp.exp(-x) / xk


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(0,))
@jnp.vectorize
@jit
def expn(n: ArrayLike, x: ArrayLike) -> Array:
  r"""Generalized exponential integral function.

  JAX implementation of :obj:`scipy.special.expn`.

  .. math::

     \mathrm{expn}(x) = E_n(x) = x^{n-1}\int_x^\infty\frac{e^{-t}}{t^n}\mathrm{d}t

  Args:
    n: arraylike, real-valued
    x: arraylike, real-valued

  Returns:
    array of expn values

  See also:
    - :func:`jax.scipy.special.expi`
    - :func:`jax.scipy.special.exp1`
  """
  n, x = promote_args_inexact("expn", n, x)
  _c = _lax_const
  zero = _c(x, 0)
  one = _c(x, 1)
  conds = [
    (n < _c(n, 0)) | (x < zero),
    (x == zero) & (n < _c(n, 2)),
    (x == zero) & (n >= _c(n, 2)),
    (n == _c(n, 0)) & (x >= zero),
    (n >= _c(n, 5000)),
    (x > one),
  ]
  n1 = jnp.where(n == _c(n, 1), n + n, n)
  vals = [
    jnp.nan,
    jnp.inf,
    one / n1,  # prevent div by zero
    jnp.exp(-x) / x,
    partial(_expn3, n),
    partial(_expn2, n),
    partial(_expn1, n),
  ]
  ret = jnp.piecewise(x, conds, vals)
  return ret


@expn.defjvp
@jit
def expn_jvp(n, primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return expn(n, x), lax.mul(
    lax.neg(x_dot), expn(lax.sub(n, _lax_const(n, 1)), x)
  )


def exp1(x: ArrayLike) -> Array:
  r"""Exponential integral function.

  JAX implementation of :obj:`scipy.special.exp1`

  .. math::

     \mathrm{exp1}(x) = E_1(x) = x^{n-1}\int_x^\infty\frac{e^{-t}}{t}\mathrm{d}t


  Args:
    x: arraylike, real-valued

  Returns:
    array of exp1 values

  See also:
    - :func:`jax.scipy.special.expi`
    - :func:`jax.scipy.special.expn`
  """
  x, = promote_args_inexact("exp1", x)
  # Casting because custom_jvp generic does not work correctly with mypy.
  return cast(Array, expn(1, x))


def _spence_poly(w: Array) -> Array:
  A = jnp.array([4.65128586073990045278E-5,
                  7.31589045238094711071E-3,
                  1.33847639578309018650E-1,
                  8.79691311754530315341E-1,
                  2.71149851196553469920E0,
                  4.25697156008121755724E0,
                  3.29771340985225106936E0,
                  1.00000000000000000126E0,
                  ], dtype=w.dtype)

  B = jnp.array([6.90990488912553276999E-4,
                  2.54043763932544379113E-2,
                  2.82974860602568089943E-1,
                  1.41172597751831069617E0,
                  3.63800533345137075418E0,
                  5.03278880143316990390E0,
                  3.54771340985225096217E0,
                  9.99999999999999998740E-1,
                  ],dtype=w.dtype)

  return -w * jnp.polyval(A, w) / jnp.polyval(B, w)


def _spence_calc(x: Array) -> Array:
  x2_bool = x > 2.0
  x = jnp.piecewise(x, [x2_bool],
                    [lambda x: 1.0 / x, lambda x: x])

  x1_5_bool = x > 1.5
  x_5_bool = x < 0.5
  x2_bool = x2_bool | x1_5_bool

  w = jnp.piecewise(x,
                    [x1_5_bool, x_5_bool],
                    [lambda x: 1.0 / x - 1.0,
                      lambda x: -x,
                      lambda x: x - 1.0])

  y = _spence_poly(w)
  y_flag_one = jnp.pi ** 2 / 6.0 - jnp.log(x) * jnp.log(1.0 - x) - y
  y = jnp.where(x_5_bool, y_flag_one, y)
  y_flag_two = -0.5 * jnp.log(x) ** 2 - y
  return jnp.where(x2_bool, y_flag_two, y)


def _spence(x: Array) -> Array:
  return jnp.piecewise(x,
                       [x < 0.0, x == 1.0, x == 0.0],
                       [jnp.nan, 0, jnp.pi ** 2 / 6, _spence_calc])


def spence(x: Array) -> Array:
  r"""Spence's function, also known as the dilogarithm for real values.

  JAX implementation of :obj:`scipy.special.spence`.

  It is defined to be:

  .. math::
    \mathrm{spence}(x) = \begin{equation}
    \int_1^x \frac{\log(t)}{1 - t}dt
    \end{equation}

  Unlike the SciPy implementation, this is only defined for positive
  real values of `z`. For negative values, `NaN` is returned.

  Args:
    z: An array of type `float32`, `float64`.

  Returns:
    An array with `dtype=z.dtype`.
    computed values of Spence's function.

  Raises:
    TypeError: if elements of array `z` are not in (float32, float64).

  Notes:
  There is a different convention which defines Spence's function by the
  integral:

  .. math::
    \begin{equation}
    -\int_0^z \frac{\log(1 - t)}{t}dt
    \end{equation}

  This is our spence(1 - z).
  """
  x = jnp.asarray(x)
  dtype = lax.dtype(x)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
      f"x.dtype={dtype} is not supported, see docstring for supported types.")
  return _spence(x)


def bernoulli(n: int) -> Array:
  """Generate the first N Bernoulli numbers.

  JAX implementation of :func:`scipy.special.bernoulli`.

  Args:
    n: integer, the number of Bernoulli terms to generate.

  Returns:
    Array containing the first ``n`` Bernoulli numbers.

  Notes:
    ``bernoulli`` generates numbers using the :math:`B_n^-` convention,
    such that :math:`B_1=-1/2`.
  """
  # Generate Bernoulli numbers using the Chowla and Hartung algorithm.
  n = core.concrete_or_error(operator.index, n, "Argument n of bernoulli")
  if n < 0:
    raise ValueError("n must be a non-negative integer.")
  b3 = jnp.array([1, -1/2, 1/6])
  if n < 3:
    return b3[:n + 1]
  bn = jnp.zeros(n + 1).at[:3].set(b3)
  m = jnp.arange(4, n + 1, 2, dtype=bn.dtype)
  q1 = (1. / jnp.pi ** 2) * jnp.cumprod(-(m - 1) * m / 4 / jnp.pi ** 2)
  k = jnp.arange(2, 50, dtype=bn.dtype)  # Choose 50 because 2 ** -50 < 1E-15
  q2 = jnp.sum(k[:, None] ** -m[None, :], axis=0)
  return bn.at[4::2].set(q1 * (1 + q2))


@custom_derivatives.custom_jvp
def poch(z: ArrayLike, m: ArrayLike) -> Array:
  r"""The Pochammer symbol.

  JAX implementation of :obj:`scipy.special.poch`.

  .. math::

     \mathrm{poch}(z, m) = (z)_m = \frac{\Gamma(z + m)}{\Gamma(z)}

  where :math:`\Gamma(z)` is the :func:`~jax.scipy.special.gamma` function.

  Args:
    z: arraylike, real-valued
    m: arraylike, real-valued

  Returns:
    array of Pochammer values.

  Notes:
    The JAX version supports only real-valued inputs.
  """
  z, m = promote_args_inexact("poch", z, m)

  return jnp.where(m == 0., jnp.array(1, dtype=z.dtype), gamma(z + m) / gamma(z))


def _poch_z_derivative(z, m):
  """
  Defined in :
  https://functions.wolfram.com/GammaBetaErf/Pochhammer/20/01/01/
  """

  return (digamma(z + m) - digamma(z)) * poch(z, m)


def _poch_m_derivative(z, m):
  """
  Defined in :
  https://functions.wolfram.com/GammaBetaErf/Pochhammer/20/01/02/
  """

  return digamma(z + m) * poch(z, m)


poch.defjvps(
  lambda z_dot, primal_out, z, m:  _poch_z_derivative(z, m) * z_dot,
  lambda m_dot, primal_out, z, m: _poch_m_derivative(z, m) * m_dot,
)


def _hyp1f1_serie(a, b, x):
  """
  Compute the 1F1 hypergeometric function using the taylor expansion
  See Eq. 3.2 and associated method (a) from PEARSON, OLVER & PORTER 2014
  https://doi.org/10.48550/arXiv.1407.7786
  """

  precision = jnp.finfo(x.dtype).eps

  def body(state):
    serie, k, term = state
    serie += term
    term *= (a + k) / (b + k) * x / (k + 1)
    k += 1

    return serie, k, term

  def cond(state):
    serie, k, term = state

    return (k < 250) & (lax.abs(term) / lax.abs(serie) > precision)

  init = 1, 1, a / b * x

  return lax.while_loop(cond, body, init)[0]


def _hyp1f1_asymptotic(a, b, x):
  """
  Compute the 1F1 hypergeometric function using asymptotic expansion
  See Eq. 3.8 and simplification for real inputs from PEARSON, OLVER & PORTER 2014
  https://doi.org/10.48550/arXiv.1407.7786
  """

  precision = jnp.finfo(x.dtype).eps

  def body(state):
    serie, k, term = state
    serie += term
    term *= (b - a + k) * (1 - a + k) / (k + 1) / x
    k += 1

    return serie, k, term

  def cond(state):
    serie, k, term = state

    return (k < 250) & (lax.abs(term) / lax.abs(serie) > precision)

  init = 1, 1, (b - a) * (1 - a) / x
  serie = lax.while_loop(cond, body, init)[0]

  return gamma(b) / gamma(a) * lax.exp(x) * x ** (a - b) * serie


@jit
@jnp.vectorize
def _hyp1f1_a_derivative(a, b, x):
  """
  Define it as a serie using :
  https://functions.wolfram.com/HypergeometricFunctions/Hypergeometric1F1/20/01/01/
  """

  precision = jnp.finfo(x.dtype).eps

  def body(state):
    serie, k, term = state
    serie += term * (digamma(a + k) - digamma(a))
    term *= (a + k) / (b + k) * x / (k + 1)
    k += 1

    return serie, k, term

  def cond(state):
    serie, k, term = state

    return (k < 250) & (lax.abs(term) / lax.abs(serie) > precision)

  init = 0, 1, a / b * x

  return lax.while_loop(cond, body, init)[0]


@jit
@jnp.vectorize
def _hyp1f1_b_derivative(a, b, x):
  """
  Define it as a serie using :
  https://functions.wolfram.com/HypergeometricFunctions/Hypergeometric1F1/20/01/02/
  """

  precision = jnp.finfo(x.dtype).eps

  def body(state):
    serie, k, term = state
    serie += term * (digamma(b) - digamma(b + k))
    term *= (a + k) / (b + k) * x / (k + 1)
    k += 1

    return serie, k, term

  def cond(state):
    serie, k, term = state

    return (k < 250) & (lax.abs(term) / lax.abs(serie) > precision)

  init = 0, 1, a / b * x

  return lax.while_loop(cond, body, init)[0]


@jit
def _hyp1f1_x_derivative(a, b, x):
  """
  Define it as a serie using :
  https://functions.wolfram.com/HypergeometricFunctions/Hypergeometric1F1/20/01/04/
  """

  return a / b * hyp1f1(a + 1, b + 1, x)


@custom_derivatives.custom_jvp
@jit
@jnp.vectorize
def hyp1f1(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> Array:
  r"""The 1F1 hypergeometric function.

  JAX implementation of :obj:`scipy.special.hyp1f1`.

  .. math::

     \mathrm{hyp1f1}(a, b, x) = {}_1F_1(x;a, b) = \sum_{k=0}^\infty \frac{(a)_k}{(b)_kk!}x^k

  where :math:`(\cdot)_k` is the Pochammer symbol (refer to :func:`~jax.scipy.special.poch`).

  The JAX version only accepts positive and real inputs. Values of ``a``, ``b``,
  and ``x``, leading to high values of 1F1 may lead to erroneous results;
  consider enabling double precision in this case. The convention for
  ``a = b = 0`` is ``1``, unlike in scipy's implementation.

  Args:
    a: arraylike, real-valued
    b: arraylike, real-valued
    x: arraylike, real-valued

  Returns:
    array of 1F1 values.
  """
  # This is backed by https://doi.org/10.48550/arXiv.1407.7786
  # There is room for improvement in the implementation using recursion to
  # evaluate lower values of hyp1f1 when a or b or both are > 60-80
  a, b, x = promote_args_inexact('hyp1f1', a, b, x)

  result = lax.cond(lax.abs(x) < 100, _hyp1f1_serie, _hyp1f1_asymptotic, a, b, x)
  index = (a == 0) * 1 + ((a == b) & (a != 0)) * 2 + ((b == 0) & (a != 0)) * 3

  return lax.select_n(index,
                      result,
                      jnp.array(1, dtype=x.dtype),
                      jnp.exp(x),
                      jnp.array(jnp.inf, dtype=x.dtype))


hyp1f1.defjvps(
  lambda a_dot, primal_out, a, b, x: _hyp1f1_a_derivative(a, b, x) * a_dot,
  lambda b_dot, primal_out, a, b, x: _hyp1f1_b_derivative(a, b, x) * b_dot,
  lambda x_dot, primal_out, a, b, x: _hyp1f1_x_derivative(a, b, x) * x_dot
)


def softmax(x: ArrayLike,
            /,
            *,
            axis: int | tuple[int, ...] | None = None,
            ) -> Array:
  r"""Softmax function.

  JAX implementation of :func:`scipy.special.softmax`.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x : input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.

  Returns:
    An array of the same shape as ``x``.

  Note:
    If any input values are ``+inf``, the result will be all ``NaN``: this
    reflects the fact that ``inf / inf`` is not well-defined in the context of
    floating-point math.

  See also:
    :func:`log_softmax`
  """
  return nn_softmax(x, axis=axis)


def log_softmax(x: ArrayLike,
                /,
                *,
                axis: int | tuple[int, ...] | None = None,
                ) -> Array:
  r"""Log-Softmax function.

  JAX implementation of :func:`scipy.special.log_softmax`

  Computes the logarithm of the :code:`softmax` function, which rescales
  elements to the range :math:`[-\infty, 0)`.

  .. math ::
    \mathrm{log\_softmax}(x)_i = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    \right)

  Args:
    x : input array
    axis: the axis or axes along which the :code:`log_softmax` should be
      computed.

  Returns:
    An array of the same shape as ``x``

  Note:
    If any input values are ``+inf``, the result will be all ``NaN``: this
    reflects the fact that ``inf / inf`` is not well-defined in the context of
    floating-point math.

  See also:
    :func:`softmax`
  """
  return nn_log_softmax(x, axis=axis)
