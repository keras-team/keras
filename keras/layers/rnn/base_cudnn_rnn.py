# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Base class for recurrent layers backed by cuDNN."""


import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine.input_spec import InputSpec
from keras.layers.rnn.base_rnn import RNN


class _CuDNNRNN(RNN):
    """Private base class for CuDNNGRU and CuDNNLSTM layers.

    Args:
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
          in addition to the output.
      go_backwards: Boolean (default False).
          If True, process the input sequence backwards and return the
          reversed sequence.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      time_major: Boolean (default False). If true, the inputs and outputs will
          be in shape `(timesteps, batch, ...)`, whereas in the False case, it
          will be `(batch, timesteps, ...)`.
    """

    def __init__(
        self,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        time_major=False,
        **kwargs
    ):
        # We invoke the base layer's initializer directly here because we do not
        # want to create RNN cell instance.
        super(RNN, self).__init__(**kwargs)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.time_major = time_major
        self.supports_masking = False
        self.input_spec = [InputSpec(ndim=3)]
        if hasattr(self.cell.state_size, "__len__"):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]
        self.state_spec = [InputSpec(shape=(None, dim)) for dim in state_size]
        self.constants_spec = None
        self._states = None
        self._num_constants = 0
        self._vector_shape = tf.constant([-1])

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if isinstance(mask, list):
            mask = mask[0]
        if mask is not None:
            raise ValueError("Masking is not supported for CuDNN RNNs.")

        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_state = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if len(initial_state) != len(self.states):
            raise ValueError(
                "Layer has "
                + str(len(self.states))
                + " states but was passed "
                + str(len(initial_state))
                + " initial states."
            )

        if self.go_backwards:
            # Reverse time axis.
            inputs = backend.reverse(inputs, 1)
        output, states = self._process_batch(inputs, initial_state)

        if self.stateful:
            updates = [
                tf.compat.v1.assign(self_state, state)
                for self_state, state in zip(self.states, states)
            ]
            self.add_update(updates)

        if self.return_state:
            return [output] + states
        else:
            return output

    def get_config(self):
        config = {
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards,
            "stateful": self.stateful,
            "time_major": self.time_major,
        }
        base_config = super(RNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def trainable_weights(self):
        if self.trainable and self.built:
            return [self.kernel, self.recurrent_kernel, self.bias]
        return []

    @property
    def non_trainable_weights(self):
        if not self.trainable and self.built:
            return [self.kernel, self.recurrent_kernel, self.bias]
        return []

    @property
    def losses(self):
        return super(RNN, self).losses

    def get_losses_for(self, inputs=None):
        return super(RNN, self).get_losses_for(inputs=inputs)
