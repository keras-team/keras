import jax
from functools import partial
import jax.numpy as jnp
from jax import random
from jax import value_and_grad

def value_and_grad_wrapper(fn, **kwargs):
    @partial(value_and_grad, **kwargs)
    def inner(*args, **kwargs):
        return jnp.sum(fn(*args, **kwargs))
    return inner

def diff(t1, t2):
    return jnp.max(jnp.abs(t1 - t2))

def PRNGKeyGenerator(seed = 42):
    key = random.PRNGKey(seed)
    while True:
        sub_key, key = random.split(key)
        yield sub_key

def value_and_grad_difference(
    fn1,
    fn2,
    seed = 42,
    batch = 2,
    heads = 4,
    q_seq_len = 4096,
    k_seq_len = 8192,
    add_key_mask = True,
    dim = 512
):
    key_gen = PRNGKeyGenerator(seed)

    q = random.normal(next(key_gen), (batch, heads, q_seq_len, dim))
    k = random.normal(next(key_gen), (batch, heads, k_seq_len, dim))
    v = random.normal(next(key_gen), (batch, heads, k_seq_len, dim))

    key_mask = random.randint(next(key_gen), (batch, k_seq_len), 0, 2) == 1

    fn1_value_and_grad, fn2_value_and_grad = map(partial(value_and_grad_wrapper, argnums = (0, 1, 2)), (fn1, fn2))

    args = (q, k, v)
    if add_key_mask:
        args = (*args, key_mask)

    o1, grads1 = fn1_value_and_grad(*args)
    o2, grads2 = fn2_value_and_grad(*args)

    return diff(o1, o2), [diff(*args) for args in zip(grads1, grads2)]
