from keras.src.backend.common import backend_utils
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.variables import AutocastScope
from keras.src.backend.common.variables import Variable as KerasVariable
from keras.src.backend.common.variables import (get_autocast_scope,
                                                is_float_dtype, is_int_dtype,
                                                standardize_dtype,
                                                standardize_shape)
from keras.src.random import random
