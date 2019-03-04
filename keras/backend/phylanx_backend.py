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


def eval(func):
	return func.eval()


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
def random_uniform_eager(shape, minval, maxval, dtype=None, seed=None):
	return random(shape, list("uniform", minval, maxval))

def random_uniform(shape, minval, maxval, dtype=None, seed=None):
	return random_uniform_eager.lazy(shape, -1, 1)



