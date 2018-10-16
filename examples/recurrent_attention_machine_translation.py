'''Machine Translation (word-level) with recurrent attention in Keras.

This script demonstrates how to implement a basic word-level machine
translation using the recurrent attention mechanism described in [1].

# Summary of the algorithm
- ...
- ...
- In inference mode, TODO when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.


# Data
We use the machine translation dataset described in [2]. Note that this
is newer dataset that than what was used in [1].

To download the data run:
    mkdir -p data/wmt16_mmt
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/wmt16_mmt && rm training.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/wmt16_mmt && rm validation.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C data/wmt16_mmt && rm mmt16_task1_test.tar.gz

# References
[1] Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/abs/1409.0473
[2] Multi30K: Multilingual English-German Image Descriptions
    https://arxiv.org/abs/1605.00459 (http://www.statmt.org/wmt16/multimodal-task.html)
'''

from __future__ import print_function

import os
from keras import backend as K
from keras.activations import tanh, softmax
from keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    RNN,
    GRUCell,
    TimeDistributed,
    Dense,
    concatenate, Wrapper, Lambda, add, RepeatVector, multiply, GRU, Masking)
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# DATA_DIR = 'data/wmt16_mmt'
DATA_DIR = '/Users/andershuss/Datasets/WMT16/mmt'
FROM_LANGUAGE = 'en'
TO_LANGUAGE = 'de'

# Meta parameters
MAX_UNIQUE_WORDS = 30000
MAX_WORDS_PER_SENTENCE = 40  # inf in [1]
EMBEDDING_SIZE = 620  # `m` in [1]
RECURRENT_UNITS = 1000  # `n` in [1]
DENSE_ATTENTION_UNITS = 1000  # fixed equal to `n` in [1]
READOUT_HIDDEN_UNITS = 500  # `l` in [1]
OPTIMIZER = Adadelta(rho=0.95, epsilon=1e-6)

# MAX_UNIQUE_WORDS = 1000
# MAX_WORDS_PER_SENTENCE = 20  # inf in [1]
# EMBEDDING_SIZE = 128  # `m` in [1]
# RECURRENT_UNITS = 128  # `n` in [1]
# DENSE_ATTENTION_UNITS = 256  # fixed equal to `n` in [1]
# READOUT_HIDDEN_UNITS = 128  # `l` in [1]
# OPTIMIZER = Adadelta(rho=0.95, epsilon=1e-6)


batch_size = 64  # Batch size for training.
epochs = 20  # Number of epochs to train for.


# Load and tokenize the data.
def get_sentences(partion, language):
    fpath = os.path.join(DATA_DIR, partion + '.' + language)
    with open(fpath, 'r') as f:
        sentences = f.readlines()
    return ["<start> {} <end>".format(sentence) for sentence in sentences]


input_texts_train = get_sentences("train", FROM_LANGUAGE)
input_texts_val = get_sentences("val", FROM_LANGUAGE)
target_texts_train = get_sentences("train", TO_LANGUAGE)
target_texts_val = get_sentences("val", TO_LANGUAGE)

input_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS)
target_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS)
input_tokenizer.fit_on_texts(input_texts_train + input_texts_val)
target_tokenizer.fit_on_texts(target_texts_train + target_texts_val)

input_seqs_train = input_tokenizer.texts_to_sequences(input_texts_train)
input_seqs_val = input_tokenizer.texts_to_sequences(input_texts_val)
target_seqs_train = target_tokenizer.texts_to_sequences(target_texts_train)
target_seqs_val = target_tokenizer.texts_to_sequences(target_texts_val)

input_seqs_train, input_seqs_val, target_seqs_train, target_seqs_val = (
    pad_sequences(seq, maxlen=MAX_WORDS_PER_SENTENCE, padding='post')
    for seq in [input_seqs_train, input_seqs_val, target_seqs_train, target_seqs_val]
)

# build the model
x = Input((None,), name="input_sequences")
y = Input((None,), name="target_sequences")
# x_mask = Masking(x)
# y_mask = Masking(y)
x_emb = Embedding(input_tokenizer.num_words, EMBEDDING_SIZE, mask_zero=True)(x)
y_emb = Embedding(target_tokenizer.num_words, EMBEDDING_SIZE, mask_zero=True)(y)


class CellModel(Wrapper):
    """Wrapper for allowing composition of RNN Cells using functional API.

    # Arguments:
        inputs: input tensor at a single time step
        outputs: output tensor at a single timestep
        input_states: state tensor(s) from previous time step
        output_states: state tensor(s) after cell transformation
        constants: tensor(s) or None, represents inputs that should be static
            (the same) for each time step.

    # Examples

    ```python
    # Use functional API to define RNN Cell transformation (in this case
    # simple vanilla RNN) for a single time step:

    units = 32
    input_size = 5
    x = Input((input_size,))
    h_tm1 = Input((units,))
    h_ = add([Dense(units)(x), Dense(units, use_bias=False)(h_tm1)])
    h = Activation('tanh')(h_)

    # Create the cell:

    cell = FunctionalRNNCell(
        inputs=x, outputs=h, input_states=h_tm1, output_states=h)

    x_sequence = Input((None, input_size))
    rnn = RNN(cell)
    y = rnn(x_sequence)

    # We can also define cells that make use of "external" constants, to
    # implement attention mechanisms:

    constant_shape = (10,)
    c = Input(constant_shape)
    density = Dense(constant_shape[0], activation='softmax')(
        concatenate([x, h_tm1]))
    attention = multiply([density, c])
    h2_ = add([h_, Dense(units)(attention)])
    h2 = Activation('tanh')(h2_)

    attention_cell = FunctionalRNNCell(
        inputs=x, outputs=h2, input_states=h_tm1, output_states=h2,
        constants=c)

    attention_rnn = RNN(attention_cell)
    y2 = attention_rnn(x_sequence, constants=c)

    # Remember to pass the constant to the RNN layer (which will pass it on to
    # the cell). Also note that shape of c is same as in cell (no time
    # dimension added)

    attention_model = Model([x_sequence, c], y2)
    ```
    """
    def __init__(
        self,
        inputs,
        outputs,
        input_states,
        output_states,
        constants=None,
        **kwargs
    ):
        input_states = self._to_list_or_none(input_states)
        output_states = self._to_list_or_none(output_states)
        constants = self._to_list_or_none(constants)
        model = Model(
            inputs=self._get_model_inputs(inputs, input_states, constants),
            outputs=[outputs] + output_states
        )
        super(CellModel, self).__init__(layer=model, **kwargs)

        in_states_shape = [K.int_shape(state) for state in input_states]
        out_states_shape = [K.int_shape(state) for state in output_states]
        if not in_states_shape == out_states_shape:
            raise ValueError(
                'shape of input_states: {} are not same as shape of '
                'output_states: {}'.format(in_states_shape, out_states_shape))
        self.state_size = [state_shape[-1] for state_shape in in_states_shape]
        self.output_size = K.int_shape(outputs)[-1]

    def call(self, inputs, states, constants=None):
        """Defines the cell transformation for a single time step.

        # Arguments
            inputs: Tensor representing input at current time step.
            states: Tensor or list/tuple of tensors representing states from
                previous time step.
            constants: Tensor or list of tensors or None representing inputs
                that should be the same at each time step.
        """
        outputs = self.layer(self._get_model_inputs(inputs, states, constants))
        output, states = outputs[0], outputs[1:]

        return output, states

    @staticmethod
    def _get_model_inputs(inputs, input_states, constants):
        inputs = [inputs] + list(input_states)
        if constants is not None:
            inputs += constants

        return inputs

    @staticmethod
    def _to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]


encoder = Bidirectional(GRU(RECURRENT_UNITS, return_sequences=True))
x_enc = encoder(x_emb)


def get_attentive_gru_cell(units):
    h_cell_tm1 = Input((units + K.int_shape(x_enc)[-1],), name="cell_h_tm1")
    x_cell_t = Input((EMBEDDING_SIZE,), name="cell_x_t")  # input to cell at time t, i.e. y_emb_t
    attended = Input(K.int_shape(x_enc)[1:], name="cell_attended")

    # compute attention weights
    e_h = Dense(DENSE_ATTENTION_UNITS)(h_cell_tm1)
    e_h_repeated = Lambda(lambda (h, seq): RepeatVector(K.shape(seq)[1])(h))([e_h, attended])
    e_att = TimeDistributed(Dense(DENSE_ATTENTION_UNITS))(attended)
    e = TimeDistributed(Dense(1))(Lambda(tanh)(add([e_h_repeated, e_att])))
    a = Lambda(lambda _x: softmax(_x, axis=1))(e)

    # weighted average of attended
    attended_weighted = multiply([a, attended])
    c = Lambda(lambda _x: K.sum(_x, axis=1, keepdims=False))(attended_weighted)

    # concatenate attention encoding to input to GRUCell
    x_gru_t = concatenate([x_cell_t, c])

    # apply cell transformation
    # TODO why doesn't it work to just call the cell?
    gru_cell = GRUCell(units)
    gru_cell.build(K.int_shape(x_gru_t))

    def call_cell((x_tm1, h_tm1)):
        y_t, [h_t] = gru_cell.call(x_tm1, [h_tm1[:, :units]])
        return [y_t, h_t]

    y_gru_t, h_gru_t = Lambda(call_cell)([x_gru_t, h_cell_tm1])

    y_cell_t = concatenate([y_gru_t, c])

    return CellModel(inputs=x_cell_t,
                     outputs=y_cell_t,
                     input_states=h_cell_tm1,
                     output_states=y_cell_t,
                     constants=attended)


# cell = DenseAnnotationAttention(cell=GRUCell(RECURRENT_UNITS),
#                                 units=DENSE_ATTENTION_UNITS,
#                                 output_attention_encoding=True)  # concatenates attention_h to cell output
# decoder = RNN(cell, return_sequences=True)
cell = get_attentive_gru_cell(RECURRENT_UNITS)
decoder = RNN(cell, return_sequences=True)

h1 = decoder(y_emb, constants=x_enc)
h2 = TimeDistributed(Dense(READOUT_HIDDEN_UNITS, activation='relu'))(concatenate([h1, y_emb]))  # TODO maxout
y_pred = TimeDistributed(Dense(target_tokenizer.num_words))(h2)


model = Model([y, x], y_pred)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['acc'])

model.fit([target_seqs_train[:, :-1], input_seqs_train],
          target_seqs_train[:, 1:, None],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(
              [target_seqs_train[:, :-1], input_seqs_train],
              target_seqs_train[:, 1:, None]))
