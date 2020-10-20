"""Constraints: functions that impose constraints on weight values.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.constraints import Constraint

from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.constraints import MinMaxNorm

from tensorflow.keras.constraints import get
from tensorflow.keras.constraints import serialize
from tensorflow.keras.constraints import deserialize

# Aliases.

max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm


# Legacy aliases.
maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm

