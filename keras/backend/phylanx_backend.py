"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .common import floatx

def variable(value, dtype=None, name=None, constraint=None):
    if constraint is not None:
        raise TypeError("Constraint must be None when "
                        "using the NumPy backend.")
    return np.array(value, dtype)

def eye(size, dtype=None, name=None):
    return np.eye(size, dtype=dtype)
