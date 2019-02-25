"""Utilities for backend functionality checks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import phylanx
from phylanx import Phylanx, PhylanxSession
from .common import floatx

PhylanxSession.init(1)

et = phylanx.execution_tree
cs = phylanx.compiler_state()

def variable(value, dtype=None, name=None, constraint=None):
    if constraint is not None:
        raise TypeError("Constraint must be None when "
                        "using the NumPy backend.")
    return phylanx.execution_tree.var(np.array(value, dtype))


class eye:
    def __init__(self, *args):
        @Phylanx
        def eye_impl(size, dtype=None, name=None):
            return np.eye(size, dtype='float')

        self.src = eye_impl.__src__
        self.args = args

    def eval(self):
        return et.eval(self.src, cs, *self.args)

def eval(func):
    return func.eval()


