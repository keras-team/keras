from keras_core.initializers.constant_initializers import Ones
from keras_core.initializers.constant_initializers import Zeros
from keras_core.initializers.initializer import Initializer
from keras_core.initializers.random_initializers import GlorotNormal
from keras_core.initializers.random_initializers import GlorotUniform
from keras_core.initializers.random_initializers import HeNormal
from keras_core.initializers.random_initializers import HeUniform
from keras_core.initializers.random_initializers import LecunNormal
from keras_core.initializers.random_initializers import LecunUniform
from keras_core.initializers.random_initializers import RandomNormal
from keras_core.initializers.random_initializers import RandomUniform
from keras_core.initializers.random_initializers import VarianceScaling


def get(identifier):
    # Temporary shim
    if identifier == "zeros":
        return Zeros()
    if identifier == "ones":
        return Ones()
    if identifier == "glorot_uniform":
        return GlorotUniform()
    if identifier == "glorot_normal":
        return GlorotNormal()
    return identifier
