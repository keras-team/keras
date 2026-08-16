from keras.src.backend.common import backend_utils
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.variables import AutocastScope
from keras.src.backend.common.variables import Variable as KerasVariable
from keras.src.backend.common.variables import get_autocast_scope
from keras.src.backend.common.variables import is_float_dtype
from keras.src.backend.common.variables import is_int_dtype
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.common.variables import standardize_shape
from keras.src.random import random
