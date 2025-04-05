from __future__ import annotations

import jax.numpy as jnp
from jax import jit

from jax._src import custom_derivatives, dtypes
from jax._src.numpy.lax_numpy import complexfloating
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


@jit
def sincospisquaredhalf(
  x: Array,
) -> tuple[Array, Array]:
  """
  Accurate evaluation of sin(pi * x**2 / 2) and cos(pi * x**2 / 2).

  As based on the sinpi and cospi functions from SciPy, see:
  - https://github.com/scipy/scipy/blob/v1.14.0/scipy/special/special/cephes/trig.h
  """
  x = jnp.abs(x)
  # define s = x % 2, y = x - s, then
  # r = (x * x / 2) % 2
  #   = [(y + s)*(y + s)/2] % 2
  #   = [y*y/2 + s*y + s*s/2] % 2
  #   = [(y*y/2)%2 + (s*y + s*s/2)%2]%2
  #   = [0 + (s*(y+s/2))%2]%2
  #   = [s*(x-s/2)]%2
  s = jnp.fmod(x, 2.0)
  r = jnp.fmod(s * (x - s / 2), 2.0)

  sinpi = jnp.where(
    r < 0.5,
    jnp.sin(jnp.pi * r),
    jnp.where(
      r > 1.5,
      jnp.sin(jnp.pi * (r - 2.0)),
      -jnp.sin(jnp.pi * (r - 1.0)),
    ),
  )
  cospi = jnp.where(
    r == 0.5,
    0.0,
    jnp.where(r < 1.0, -jnp.sin(jnp.pi * (r - 0.5)), jnp.sin(jnp.pi * (r - 1.5))),
  )

  return sinpi, cospi


@custom_derivatives.custom_jvp
def fresnel(x: ArrayLike) -> tuple[Array, Array]:
  r"""The Fresnel integrals

  JAX implementation of :obj:`scipy.special.fresnel`.

  The Fresnel integrals are defined as
    .. math::
       S(x) &= \int_0^x \sin(\pi t^2 /2) dt \\
       C(x) &= \int_0^x \cos(\pi t^2 /2) dt.

  Args:
    x: arraylike, real-valued.

  Returns:
    Arrays containing the values of the Fresnel integrals.

  Notes:
     The JAX version only supports real-valued inputs, and
     is based on the SciPy C++ implementation, see
     `here
     <https://github.com/scipy/scipy/blob/v1.14.0/scipy/special/special/cephes/fresnl.h>`_.
     For ``float32`` dtypes, the implementation is directly based
     on the Cephes implementation ``fresnlf``.

     As for the original Cephes implementation, the accuracy
     is only guaranteed in the domain [-10, 10]. Outside of
     that domain, one could observe divergence between the
     theoretical derivatives and the custom JVP implementation,
     especially for large input values.

     Finally, for half-precision data types, ``float16``
     and ``bfloat16``, the array elements are upcasted to
     ``float32`` as the Cephes coefficients used in
     series expansions would otherwise lead to poor results.
     Other data types, like ``float8``, are not supported.
  """

  xxa, = promote_args_inexact("fresnel", x)
  original_dtype = xxa.dtype

  # This part is mostly a direct translation of SciPy's C++ code,
  # and the original Cephes implementation for single precision.

  if dtypes.issubdtype(xxa.dtype, complexfloating):
    raise NotImplementedError(
        'Support for complex-valued inputs is not implemented yet.')
  elif xxa.dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
    # Single-precision Cephes coefficients

    # For half-precision, series expansions have either
    # produce overflow or poor accuracy.
    # Upcasting to single-precision is hence needed.
    xxa = xxa.astype(jnp.float32)  # No-op for float32

    fresnl_sn = jnp.array([
      +1.647629463788700e-9,
      -1.522754752581096e-7,
      +8.424748808502400e-6,
      -3.120693124703272e-4,
      +7.244727626597022e-3,
      -9.228055941124598e-2,
      +5.235987735681432e-1,
    ], dtype=jnp.float32)

    fresnl_cn = jnp.array([
      +1.416802502367354e-8,
      -1.157231412229871e-6,
      +5.387223446683264e-5,
      -1.604381798862293e-3,
      +2.818489036795073e-2,
      -2.467398198317899e-1,
      +9.999999760004487e-1,
    ], dtype=jnp.float32)

    fresnl_fn = jnp.array([
      -1.903009855649792e12,
      +1.355942388050252e11,
      -4.158143148511033e9,
      +7.343848463587323e7,
      -8.732356681548485e5,
      +8.560515466275470e3,
      -1.032877601091159e2,
      +2.999401847870011e0,
    ], dtype=jnp.float32)

    fresnl_gn = jnp.array([
      -1.860843997624650e11,
      +1.278350673393208e10,
      -3.779387713202229e8,
      +6.492611570598858e6,
      -7.787789623358162e4,
      +8.602931494734327e2,
      -1.493439396592284e1,
      +9.999841934744914e-1,
    ], dtype=jnp.float32)
  elif xxa.dtype == jnp.float64:
    # Double-precision Cephes coefficients

    fresnl_sn = jnp.array([
      -2.99181919401019853726e3,
      +7.08840045257738576863e5,
      -6.29741486205862506537e7,
      +2.54890880573376359104e9,
      -4.42979518059697779103e10,
      +3.18016297876567817986e11,
    ], dtype=jnp.float64)

    fresnl_sd = jnp.array([
      +1.00000000000000000000e0,
      +2.81376268889994315696e2,
      +4.55847810806532581675e4,
      +5.17343888770096400730e6,
      +4.19320245898111231129e8,
      +2.24411795645340920940e10,
      +6.07366389490084639049e11,
    ], dtype=jnp.float64)

    fresnl_cn = jnp.array([
      -4.98843114573573548651e-8,
      +9.50428062829859605134e-6,
      -6.45191435683965050962e-4,
      +1.88843319396703850064e-2,
      -2.05525900955013891793e-1,
      +9.99999999999999998822e-1,
    ], dtype=jnp.float64)

    fresnl_cd = jnp.array([
      +3.99982968972495980367e-12,
      +9.15439215774657478799e-10,
      +1.25001862479598821474e-7,
      +1.22262789024179030997e-5,
      +8.68029542941784300606e-4,
      +4.12142090722199792936e-2,
      +1.00000000000000000118e0,
    ], dtype=jnp.float64)

    fresnl_fn = jnp.array([
      +4.21543555043677546506e-1,
      +1.43407919780758885261e-1,
      +1.15220955073585758835e-2,
      +3.45017939782574027900e-4,
      +4.63613749287867322088e-6,
      +3.05568983790257605827e-8,
      +1.02304514164907233465e-10,
      +1.72010743268161828879e-13,
      +1.34283276233062758925e-16,
      +3.76329711269987889006e-20,
    ], dtype=jnp.float64)

    fresnl_fd = jnp.array([
      +1.00000000000000000000e0,
      +7.51586398353378947175e-1,
      +1.16888925859191382142e-1,
      +6.44051526508858611005e-3,
      +1.55934409164153020873e-4,
      +1.84627567348930545870e-6,
      +1.12699224763999035261e-8,
      +3.60140029589371370404e-11,
      +5.88754533621578410010e-14,
      +4.52001434074129701496e-17,
      +1.25443237090011264384e-20,
    ], dtype=jnp.float64)

    fresnl_gn = jnp.array([
      +5.04442073643383265887e-1,
      +1.97102833525523411709e-1,
      +1.87648584092575249293e-2,
      +6.84079380915393090172e-4,
      +1.15138826111884280931e-5,
      +9.82852443688422223854e-8,
      +4.45344415861750144738e-10,
      +1.08268041139020870318e-12,
      +1.37555460633261799868e-15,
      +8.36354435630677421531e-19,
      +1.86958710162783235106e-22,
    ], dtype=jnp.float64)

    fresnl_gd = jnp.array([
      +1.00000000000000000000e0,
      +1.47495759925128324529e0,
      +3.37748989120019970451e-1,
      +2.53603741420338795122e-2,
      +8.14679107184306179049e-4,
      +1.27545075667729118702e-5,
      +1.04314589657571990585e-7,
      +4.60680728146520428211e-10,
      +1.10273215066240270757e-12,
      +1.38796531259578871258e-15,
      +8.39158816283118707363e-19,
      +1.86958710162783236342e-22,
    ], dtype=jnp.float64)
  else:
    raise NotImplementedError(
        f'Support for {xxa.dtype} dtype is not implemented yet.')

  assert xxa.dtype in (jnp.float32, jnp.float64)
  single_precision = (xxa.dtype == jnp.float32)

  x = jnp.abs(xxa)

  x2 = x * x

  # Infinite x values
  s_inf = c_inf = 0.5

  # Small x values
  t = x2 * x2

  if single_precision:
    s_small = x * x2 * jnp.polyval(fresnl_sn, t)
    c_small = x * jnp.polyval(fresnl_cn, t)
  else:
    s_small = x * x2 * jnp.polyval(fresnl_sn[:6], t) / jnp.polyval(fresnl_sd[:7], t)
    c_small = x * jnp.polyval(fresnl_cn[:6], t) / jnp.polyval(fresnl_cd[:7], t)

  # Large x values

  sinpi, cospi = sincospisquaredhalf(x)

  if single_precision:
    c_large = c_inf
    s_large = s_inf
  else:
    c_large = 0.5 + 1 / (jnp.pi * x) * sinpi
    s_large = 0.5 - 1 / (jnp.pi * x) * cospi

  # Other x values
  t = jnp.pi * x2
  u = 1.0 / (t * t)
  t = 1.0 / t

  if single_precision:
    f = 1.0 - u * jnp.polyval(fresnl_fn, u)
    g = t * jnp.polyval(fresnl_gn, u)
  else:
    f = 1.0 - u * jnp.polyval(fresnl_fn, u) / jnp.polyval(fresnl_fd, u)
    g = t * jnp.polyval(fresnl_gn, u) / jnp.polyval(fresnl_gd, u)

  t = jnp.pi * x
  c_other = 0.5 + (f * sinpi - g * cospi) / t
  s_other = 0.5 - (f * cospi + g * sinpi) / t

  isinf = jnp.isinf(xxa)
  small = x2 < 2.5625
  large = x > 36974.0
  s = jnp.where(
    isinf, s_inf, jnp.where(small, s_small, jnp.where(large, s_large, s_other))
  )
  c = jnp.where(
    isinf, c_inf, jnp.where(small, c_small, jnp.where(large, c_large, c_other))
  )

  neg = xxa < 0.0
  s = jnp.where(neg, -s, s)
  c = jnp.where(neg, -c, c)

  if original_dtype != xxa.dtype:
    s = s.astype(original_dtype)
    c = c.astype(original_dtype)

  return s, c

def _fresnel_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  result = fresnel(x)
  sinpi, cospi = sincospisquaredhalf(x)
  dSdx = sinpi * x_dot
  dCdx = cospi * x_dot
  return result, (dSdx, dCdx)
fresnel.defjvp(_fresnel_jvp)
