import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl


def cholesky(a):
    return jnp.linalg.cholesky(a)


def det(a):
    return jnp.linalg.det(a)


def inv(a):
    return jnp.linalg.inv(a)


def logdet(a):
    return jnp.linalg.logdet(a)


def solve(a, b):
    return jnp.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    return jsl.solve_triangular(a, b, lower=lower)
