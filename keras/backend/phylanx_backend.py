"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from phylanx import Phylanx, PhylanxSession, execution_tree
from .common import floatx
from .common import epsilon
from .common import normalize_data_format

PhylanxSession.init(1)


def variable(value, dtype=None, name=None, constraint=None):
	if constraint is not None:
		raise TypeError("Constraint is the projection function to be "
						"applied to the variable after an optimizer update")
	return execution_tree.var(np.array(value, dtype))


def eval(x):
	return x.eval()


# not tested in the backend, should work on both variables and placeholders
@Phylanx
def ndim_eager(x):
	return ndim(x)

def ndim(x):
	return ndim_eager.lazy(x)




@Phylanx
def eye_eager(size, dtype, name):
	return np.eye(size)

def eye(size, dtype=None, name=None):
	return eye_eager.lazy(size)


# works up to 3d
@Phylanx
def ones_eager(shape, dtype, name):
	return np.ones(shape)

def ones(shape, dtype=floatx(), name=None):
	return ones_eager.lazy(shape)


@Phylanx
def zeros_eager(shape, dtype, name):
	return np.zeros(shape)

def zeros(shape, dtype=floatx(), name=None):
	return zeros_eager.lazy(shape)


@Phylanx
def ones_like_eager(x, dtype, name):
	return np.ones_like(x)

def ones_like(x, dtype=floatx(), name=None):
	return ones_like_eager.lazy(x)


@Phylanx
def zeros_like_eager(x, dtype, name):
	return np.zeros_like(x)

def zeros_like(x, dtype=floatx(), name=None):
	return zeros_like_eager.lazy(x)
###


@Phylanx
def dot_eager(x, y):
	return np.dot(x, y)

def dot(x, y):
	return dot_eager.lazy(x, y)





@Phylanx
def transpose_eager(x):
	return np.transpose(x)

def transpose(x):
	return transpose_eager.lazy(x)


@Phylanx
def reverse_eager(x, axes):
	return np.flip(x, axes)

def reverse(x, axes):
	return reverse_eager.lazy(x, axes)


@Phylanx
def phylanx_random_uniform_variable(shape, low, high):
	return random(shape, list("uniform", low, high))

def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
	return execution_tree.var(phylanx_random_uniform_variable(shape, low, high))


@Phylanx
def phylanx_random_normal_variable(shape, mean, scale):
	return random(shape, list("normal", mean, scale))

def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
	return execution_tree.var(phylanx_random_normal_variable(shape, mean, scale))


#@Phylanx
#def concatenate_eager(tensors, axis):
#	return np.concatenate(tensors, axis)

#def concatenate(tensors, axis=-1):
#	return concatenate_eager.lazy(tensors, axis)


@Phylanx
def reshape_eager(x, shape):
	return np.reshape(x, shape)

def reshape(x, shape):
	return reshape_eager.lazy(x, shape)


@Phylanx
def permute_dimensions_eager(x, pattern):
	return np.transpose(x, pattern)

def permute_dimensions(x, pattern):
	return permute_dimensions_eager.lazy(x, pattern)


@Phylanx
def repeat_eager(x, n):
	y = np.expand_dims(x, 1)
	return np.repeat(y, n, 1)

def repeat(x, n):
	return repeat_eager.lazy(x, n)


@Phylanx
def flatten_eager(x):
	return flatten(x)

def flatten(x):
	return flatten_eager.lazy(x)


@Phylanx
def batch_flatten_eager(x):
	return np.reshape(x, list(shape(x)[0], -1))

def batch_flatten(x):
	return batch_flatten_eager.lazy(x)

@Phylanx
def expand_dims_eager(x, axis):
	return np.expand_dims(x, axis)

def expand_dims(x, axis=-1):
	return expand_dims_eager.lazy(x, axis)


@Phylanx
def squeeze_eager(x, axis):
	return np.squeeze(x, axis)

def squeeze(x, axis):
	return squeeze_eager.lazy(x, axis)


#placeholders
@Phylanx
def repeat_elements_eager(x, rep, axis):
	return np.repeat(x, rep, axis)

def repeat_elements(x, rep, axis):
	return repeat_elements_eager.lazy(x, rep, axis)


#placeholders
@Phylanx
def tile_eager(x, n):
	return np.tile(x, n)

def tile(x, n):
	return tile_eager.lazy(x, n)

#///
# nil problem, axis problem
@Phylanx
def max_eager(x, axis, keepdims):
	return np.amax(x, axis, keepdims)

def max(x, axis=None, keepdims=False):
	return max_eager.lazy(x, axis, keepdims)


@Phylanx
def min_eager(x, axis, keepdims):
	return np.amin(x, axis, keepdims)

def min(x, axis=None, keepdims=False):
	return min_eager.lazy(x, axis, keepdims)


@Phylanx
def mean_eager(x, axis, keepdims):
	return np.mean(x, axis, keepdims)

def mean(x, axis=None, keepdims=False):
	return mean_eager.lazy(x, axis, keepdims)


@Phylanx
def var_eager(x, axis, keepdims):
	return np.var(x, axis, keepdims)

def var(x, axis=None, keepdims=False):
	return var_eager.lazy(x, axis, keepdims)


@Phylanx
def std_eager(x, axis, keepdims):
	return np.std(x, axis, keepdims)

def std(x, axis=None, keepdims=False):
	return std_eager.lazy(x, axis, keepdims)


@Phylanx
def logsumexp_eager(x, axis, keepdims):
	return logsumexp(x, axis, keepdims)

def logsumexp(x, axis=None, keepdims=False):
	return logsumexp_eager.lazy(x, axis, keepdims)


@Phylanx
def prod_eager(x, axis, keepdims):
	return np.prod(x, axis, keepdims)

def prod(x, axis=None, keepdims=False):
	return prod_eager.lazy(x, axis, keepdims)


@Phylanx
def any_eager(x, axis, keepdims):
	return np.any(x, axis, keepdims)

def any(x, axis=None, keepdims=False):
	return any_eager.lazy(x, axis, keepdims)


@Phylanx
def all_eager(x, axis, keepdims):
	return np.all(x, axis, keepdims)

def all(x, axis=None, keepdims=False):
	return all_eager.lazy(x, axis, keepdims)
#///


@Phylanx
def argmax_eager(x, axis):
	return np.argmax(x, axis)

def argmax(x, axis=-1):
	return argmax_eager.lazy(x, axis)


@Phylanx
def argmin_eager(x, axis):
	return np.argmin(x, axis)

def argmin(x, axis=-1):
	return argmin_eager.lazy(x, axis)


@Phylanx
def square_eager(x):
	return np.square(x)

def square(x):
	return square_eager.lazy(x)


@Phylanx
def abs_eager(x):
	return absolute(x)

def abs(x):
	return abs_eager.lazy(x)


#@Phylanx
#def sqrt_eager(x):
#	y = np.sqrt(x)
#	y[np.isnan(y)] = 0.
#	return y

#def sqrt(x):
#	return sqrt_eager.lazy(x)

@Phylanx
def exp_eager(x):
	return np.exp(x)

def exp(x):
	return exp_eager.lazy(x)


#passed although the data type should not be correct
@Phylanx
def round_eager(x):
	return rint(x)

def round(x):
	return round_eager.lazy(x)


@Phylanx
def sign_eager(x):
	return np.sign(x)

def sign(x):
	return sign_eager.lazy(x)


@Phylanx
def pow_eager(x, a):
	return np.power(x, a)

def pow(x, a=1.):
	return pow_eager.lazy(x, a)


@Phylanx
def clip_eager(x, min_value, max_value):
	return np.clip(x, min_value, max_value)

def clip(x, min_value, max_value):
	return clip_eager.lazy(x, min_value, max_value)


@Phylanx
def cos_eager(x):
	return np.cos(x)

def cos(x):
	return cos_eager.lazy(x)


@Phylanx
def sin_eager(x):
	return np.sin(x)

def sin(x):
	return sin_eager.lazy(x)


@Phylanx
def equal_eager(x, y):
	return x == y

def equal(x, y):
	return equal_eager.lazy(x, y)


@Phylanx
def not_equal_eager(x, y):
	return x != y

def not_equal(x, y):
	return not_equal_eager.lazy(x, y)


@Phylanx
def greater_eager(x, y):
	return x > y

def greater(x, y):
	return greater_eager.lazy(x, y)


@Phylanx
def greater_equal_eager(x, y):
	return x >= y

def greater_equal(x, y):
	return greater_equal_eager.lazy(x, y)


@Phylanx
def less_eager(x, y):
	return x < y

def less(x, y):
	return less_eager.lazy(x, y)


@Phylanx
def less_equal_eager(x, y):
	return x <= y

def less_equal(x, y):
	return less_equal_eager.lazy(x, y)


@Phylanx
def maximum_eager(x, y):
	return np.maximum(x,y)

def maximum(x, y):
	return maximum_eager.lazy(x, y)


@Phylanx
def minimum_eager(x, y):
	return np.minimum(x,y)

def minimum(x, y):
	return minimum_eager.lazy(x, y)


@Phylanx
def random_uniform_eager(shape, minval, maxval, dtype=None, seed=None):
	return random(shape, list("uniform", minval, maxval))

def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
	return random_uniform_eager.lazy(shape, minval, maxval)



