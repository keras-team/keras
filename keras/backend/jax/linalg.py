import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def cholesky(a):
    return jnp.linalg.cholesky(a)


def det(a):
    return jnp.linalg.det(a)


def inv(a):
    return jnp.linalg.inv(a)


def solve(a, b):
    return jnp.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    return solve_triangular(a, b, lower=lower)
