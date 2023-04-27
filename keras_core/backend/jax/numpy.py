import jax.numpy as jnp


def add(x1, x2):
    return jnp.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    return jnp.einsum(subscripts, *operands, **kwargs)


def subtract(x1, x2):
    return jnp.subtract(x1, x2)


def matmul(x1, x2):
    return jnp.matmul(x1, x2)


def multiply(x1, x2):
    return jnp.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    return jnp.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    return jnp.max(x, axis=axis, keepdims=keepdims)


def ones(shape, dtype="float32"):
    return jnp.ones(shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    return jnp.zeros(shape, dtype=dtype)
