from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..engine.topology import Layer
from .. import backend as K

import numpy as np

from tensorflow.python.ops import variables as tf_variables


class Rman(Layer):
    def __init__(self, **kwargs):
        super(Rman, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, output_dim=1024):
        if not isinstance(inputs, list):
            raise ValueError('An rman layer should be called '
                             'on a list of inputs.')
        self.input_num = len(inputs)
        self.output_dim = output_dim
        self.random_matrix = [tf_variables(np.randome.randn(inputs[i], output_dim)) for i in range(self.input_num)]
        K.stop_gradient(self.random_matrix);
        output = [inputs[i] * self.random_matrix[i] for i in range(self.input_num)]
        output[0] = output[0] / float(self.output_dim)

        return output