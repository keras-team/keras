# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras recurrent layers."""
# pylint: disable=g-bad-import-order,g-direct-tensorflow-import,disable=g-import-not-at-top

import tensorflow.compat.v2 as tf

# Recurrent layers.
from keras.layers.rnn.base_rnn import RNN
from keras.layers.rnn.abstract_rnn_cell import AbstractRNNCell
from keras.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.layers.rnn.simple_rnn import SimpleRNNCell
from keras.layers.rnn.simple_rnn import SimpleRNN

if tf.__internal__.tf2.enabled():
    from keras.layers.rnn.gru import GRU
    from keras.layers.rnn.gru import GRUCell
    from keras.layers.rnn.lstm import LSTM
    from keras.layers.rnn.lstm import LSTMCell
    from keras.layers.rnn.gru_v1 import GRU as GRUV1
    from keras.layers.rnn.gru_v1 import GRUCell as GRUCellV1
    from keras.layers.rnn.lstm_v1 import LSTM as LSTMV1
    from keras.layers.rnn.lstm_v1 import LSTMCell as LSTMCellV1

    GRUV2 = GRU
    GRUCellV2 = GRUCell
    LSTMV2 = LSTM
    LSTMCellV2 = LSTMCell
else:
    from keras.layers.rnn.gru_v1 import GRU
    from keras.layers.rnn.gru_v1 import GRUCell
    from keras.layers.rnn.lstm_v1 import LSTM
    from keras.layers.rnn.lstm_v1 import LSTMCell
    from keras.layers.rnn.gru import GRU as GRUV2
    from keras.layers.rnn.gru import GRUCell as GRUCellV2
    from keras.layers.rnn.lstm import LSTM as LSTMV2
    from keras.layers.rnn.lstm import LSTMCell as LSTMCellV2

    GRUV1 = GRU
    GRUCellV1 = GRUCell
    LSTMV1 = LSTM
    LSTMCellV1 = LSTMCell

# Convolutional-recurrent layers.
from keras.layers.rnn.conv_lstm1d import ConvLSTM1D
from keras.layers.rnn.conv_lstm2d import ConvLSTM2D
from keras.layers.rnn.conv_lstm3d import ConvLSTM3D

# cuDNN recurrent layers.
from keras.layers.rnn.cudnn_lstm import CuDNNLSTM
from keras.layers.rnn.cudnn_gru import CuDNNGRU

# Wrapper functions.
from keras.layers.rnn.base_wrapper import Wrapper
from keras.layers.rnn.bidirectional import Bidirectional
from keras.layers.rnn.time_distributed import TimeDistributed

# RNN Cell wrappers.
from keras.layers.rnn.cell_wrappers import DeviceWrapper
from keras.layers.rnn.cell_wrappers import DropoutWrapper
from keras.layers.rnn.cell_wrappers import ResidualWrapper
