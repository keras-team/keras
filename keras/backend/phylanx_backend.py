"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from phylanx import Phylanx, PhylanxSession, execution_tree
from .common import floatx

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
def eye_eager(size, dtype=None, name=None):
	return np.eye(size)

def eye(size, dtype=None, name=None):
	return eye_eager.lazy(size)


# works up to 3d
@Phylanx
def ones_eager(shape, dtype=floatx(), name=None):
	return np.ones(shape)

def ones(shape, dtype=floatx(), name=None):
	return ones_eager.lazy(shape)


@Phylanx
def zeros_eager(shape, dtype=floatx(), name=None):
	return np.zeros(shape)

def zeros(shape, dtype=floatx(), name=None):
	return zeros_eager.lazy(shape)


@Phylanx
def ones_like_eager(x, dtype=floatx(), name=None):
	return np.ones_like(x)

def ones_like(x, dtype=floatx(), name=None):
	return ones_like_eager.lazy(x)


@Phylanx
def zeros_like_eager(x, dtype=floatx(), name=None):
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



@Phylanx
def random_uniform_eager(shape, minval, maxval, dtype=None, seed=None):
	return random(shape, list("uniform", minval, maxval))

def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
	return random_uniform_eager.lazy(shape, minval, maxval)



