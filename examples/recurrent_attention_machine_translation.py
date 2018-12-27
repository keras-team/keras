'''Machine Translation (word-level) with recurrent attention in Keras.

This script demonstrates how to implement basic word-level machine
translation using the recurrent attention mechanism described in [1].

# Summary of the algorithm
The task is to generate a sequence of words in the target language
given the sequence of words in the input language. The overall approach
can be summarized as:
    teacher forcing (i.e. next word is predicted given previous words
    in target language) conditioned on the input language sentence
    using attention.

Main steps:
- Input and target sentences are tokenized (word level).
- Sentences are mapped to word embedding sequences.
- An Encoder encodes the input sequence using a Bidirectional GRU
    into a _sequence_ of encodings - with same length as the input
    sequence.
- A Decoder is constructed using a (unidirectional) GRU with attention:
    At each time step:
    - It is fed the previous target word embedding and previous GRU state.
    - Based on the previous state and the input encoding sequence, it
        computes a weight for each time step (i.e. word index) of the
        input encoding sequence, using a single hidden layer MLP. The
        attention encoding is taken as the weighted sum over the input
        encoding sequence using these weights. (NOTE: this way the
        Decoder has access to all time steps of the input for every
        generated output word.)
    - The attention encoding is concatenated to the previous target word
        embedding and fed as input to the regular GRUCell transform.
    - The output (updated state) of the GRUCell is fed to a single
        hidden (maxout) layer MLP which outputs the probabilities of the
        next target word.
- In inference mode (greedy approach),
    - Encode the input sequence
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the encoded input and 1-char target sequence
        to the decoder and predict probabilities for the next word
    - Select the next word using argmax of probabilities
    - Append the predicted word to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the word limit.

# Data
We use the machine translation dataset described in [2].
To download the data run:
    mkdir -p data/wmt16_mmt
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz
    tar -xf training.tar.gz -C data/wmt16_mmt && rm training.tar.gz
    tar -xf validation.tar.gz -C data/wmt16_mmt && rm validation.tar.gz
    tar -xf mmt16_task1_test.tar.gz -C data/wmt16_mmt && rm mmt16_task1_test.tar.gz

# References
[1] Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/abs/1409.0473
[2] Multi30K: Multilingual English-German Image Descriptions
    https://arxiv.org/abs/1605.00459
    (http://www.statmt.org/wmt16/multimodal-task.html)

# Differences between this implementation and [1]
- NOTE that a different dataset (wmt14) is used in [1], which is _orders of
    magnitude_ larger than the dataset used here (348M vs 0.35M words).
- In [1] a custom scheme is used to batch sequences of similar lengths to minimize
    the computational waste from padding of short sequences in the same batch as
    longer sequences.
- The tokenization here is similar but not identical to [1].
- Initialisation of weights are not identical here and [1].
- In the detailed description of the architecture in [1] it is stated:
    "From here on, we omit all bias terms in order to increase
    readability". It is thus not fully clear which linear transformations
    also has bias terms, here all have.
(- There is no mention of Dropout or other regularisation methods in [1],
    this could improve performance.)
'''
from __future__ import print_function

import os
import heapq

import numpy as np

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine.base_layer import _collect_previous_mask
from keras.layers import Layer, Input, InputSpec
from keras.layers import Embedding, Dense, GRU, GRUCell
from keras.layers import RNN, Bidirectional, TimeDistributed, Lambda, concatenate
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import has_arg


class AttentionCellWrapper(Layer):
    """Base class for recurrent attention mechanisms.

    This base class implements the RNN cell interface and defines a standard
    way for attention mechanisms to interact with a wrapped RNNCell
    (such as the `SimpleRNNCell`, `GRUCell` or `LSTMCell`).

    The main idea is that the attention mechanism, implemented by
    `attention_call` in extensions of this class, computes an "attention
    encoding", based on the attended input as well as the input and the wrapped
    cell state(s) at the current time step, which will be used as modified
    input for the wrapped cell.

    # Arguments
        cell: A RNN cell instance. The cell to wrap by the attention mechanism.
            See docs of `cell` argument in the `RNN` Layer for further details.
        attend_after: Boolean (default False). If True, the attention
            transformation defined by `attention_call` will be applied after
            the wrapped cell transformation (and the attention encoding will be
            used as input for wrapped cell transformation next time step).
        input_mode: String, one of `"replace"` (default) or `"concatenate"`.
            `"replace"`: only the attention encoding will be used as input for
                the wrapped cell.
            `"concatenate"` the concatenation of the original input and the
                attention encoding will be used as input to the wrapped cell.
            TODO set "concatenate" to default?
        output_mode: String, one of `"cell_output"` (default) or `"concatenate"`.
            `"cell_output"`: the output from the wrapped cell will be used.
            `"concatenate"`: the attention encoding will be concatenated to the
                output of the wrapped cell.
            TODO set "concatenate" to default?

    # Abstract Methods and Properties
        Extension of this class must implement:
            - `attention_build` (method): Builds the attention transformation
              based on input shapes.
            - `attention_call` (method): Defines the attention transformation
              returning the attention encoding.
            - `attention_size` (property): After `attention_build` has been
              called, this property should return the size (int) of the
              attention encoding. Do this by setting `_attention_size` in scope
              of `attention_build` or by implementing `attention_size`
              property.
        Extension of this class can optionally implement:
            - `attention_state_size` (property): Default [`attention_size`].
              If the attention mechanism has it own internal states (besides
              the attention encoding which is by default the only part of
              `attention_states`) override this property accordingly.
        See docs of the respective method/property for further details.

    # Details of interaction between attention and cell transformations
        Let "cell" denote wrapped RNN cell and "att(cell)" the complete
        attentive RNN cell defined by this class. We write the wrapped cell
        transformation as:

            y{t}, s_cell{t+1} = cell.call(x{t}, s_cell{t})

        where y{t} denotes the output, x{t} the input at and s_cell{t} the wrapped
        cell state(s) at time t and s_cell{t+1} the updated state(s).

        We can then write the complete "attentive" cell transformation as:

            y{t}, s_att(cell){t+1} = att(cell).call(x{t}, s_att(cell){t},
                                                    constants=attended)

        where s_att(cell) denotes the complete states of the attentive cell,
        which consists of the wrapped cell state(s) followed but the attention
        state(s), and attended denotes the tensor attended to (note: no time
        indexing as this is the same constant input at each time step).

        Internally, this is how the attention transformation, implemented by
        `attention_call`, interacts with the wrapped cell transformation
        `cell.call`:

        - with `attend_after=False` (default):
            a{t}, s_att{t+1} = att(cell).attention_call(x_t, s_cell{t},
                                                        attended, s_att{t})
            with `input_mode="replace"` (default):
                x'{t} = a{t}
            with `input_mode="concatenate"`:
                x'{t} = [x{t}, a{t}]

            y{t}, s_cell{t+1} = cell.call(x'{t}, s_cell{t})

        - with `attend_after=True`:
            with `input_mode="replace"` (default):
                x'{t} = a{t}
            with `input_mode="concatenate"`:
                x'{t} = [x{t}, a{t}]

            y{t}, s_cell{t+1} = cell.call(x'{t}, s_cell{t})
            a{t}, s_att{t+1} = att(cell).attention_call(x_t, s_cell{t+1},
                                                        attended, s_att{t})

        where a{t} denotes the attention encoding, s_att{t} the attention
        state(s), x'{t} the modified wrapped cell input and [x{.}, a{.}] the
        (tensor) concatenation of the input and attention encoding.
    """
    # in/output modes
    _REPLACE = "replace"
    _CELL_OUTPUT = "cell_output"
    _CONCATENATE = "concatenate"
    _input_modes = [_REPLACE, _CONCATENATE]
    _output_modes = [_CELL_OUTPUT, _CONCATENATE]

    def __init__(self, cell,
                 attend_after=False,
                 input_mode="replace",
                 output_mode="cell_output",
                 **kwargs):
        self.cell = cell  # must be set before calling super
        super(AttentionCellWrapper, self).__init__(**kwargs)
        self.attend_after = attend_after
        if input_mode not in self._input_modes:
            raise ValueError(
                "input_mode must be one of {}".format(self._input_modes))
        self.input_mode = input_mode
        if output_mode not in self._output_modes:
            raise ValueError(
                "output_mode must be one of {}".format(self._output_modes))
        self.output_mode = output_mode
        self.attended_spec = None
        self._attention_size = None

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       attended_mask,
                       training=None):
        """The main logic for computing the attention encoding.

        # Arguments
            inputs: The input at current time step.
            cell_states: States for the wrapped RNN cell.
            attended: The constant tensor(s) to attend at each time step.
            attention_states: States dedicated for the attention mechanism.
            attended_mask: Collected masks for the attended.
            training: Whether run in training mode or not.

        # Returns
            attention_h: The computed attention encoding at current time step.
            attention_states: States to be passed to next `attention_call`. By
                default this should be [`attention_h`].
                NOTE: if additional states are used, these should be appended
                after `attention_h`, i.e. `attention_states[0]` should always
                be `attention_h`.
        """
        raise NotImplementedError(
            '`attention_call` must be implemented by extensions of `{}`'.format(
                self.__class__.__name__))

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        """Build the attention mechanism.

        NOTE: `self._attention_size` should be set in this method to the size
        of the attention encoding (i.e. size of first `attention_states`)
        unless `attention_size` property is implemented in another way.

        # Arguments
            input_shape: Tuple of integers. Shape of the input at a single time
                step.
            cell_state_size: List of tuple of integers.
            attended_shape: List of tuple of integers.

            NOTE: both `cell_state_size` and `attended_shape` will always be
            lists - for simplicity. For example: even if (wrapped)
            `cell.state_size` is an integer, `cell_state_size` will be a list
            of this one element.
        """
        raise NotImplementedError(
            '`attention_build` must be implemented by extensions of `{}`'.format(
                self.__class__.__name__))

    @property
    def attention_size(self):
        """Size off attention encoding, an integer.
        """
        if self._attention_size is None and self.built:
            raise NotImplementedError(
                'extensions of `{}` must either set property `_attention_size`'
                ' in `attention_build` or implement the or implement'
                ' `attention_size` in some other way'.format(
                    self.__class__.__name__))

        return self._attention_size

    @property
    def attention_state_size(self):
        """Size of attention states, defaults to `attention_size`, an integer.

        Modify this property to return list of integers if the attention
        mechanism has several internal states. Note that the first size should
        always be the size of the attention encoding, i.e.:
            `attention_state_size[0]` = `attention_size`
        """
        return self.attention_size

    @property
    def state_size(self):
        """Size of states of the complete attentive cell, a tuple of integers.

        The attentive cell's states consists of the wrapped RNN cell state size(s)
        followed by attention state size(s). NOTE it is important that the wrapped
        cell states are first as the first state of any RNN cell should be same
        as the cell's output.
        """
        state_size_s = []
        for state_size in [self.cell.state_size, self.attention_state_size]:
            if hasattr(state_size, '__len__'):
                state_size_s += list(state_size)
            else:
                state_size_s.append(state_size)

        return tuple(state_size_s)

    @property
    def output_size(self):
        if self.output_mode == self._CELL_OUTPUT:
            return self._wrapped_cell_output_size
        if self.output_mode == self._CONCATENATE:
            return self._wrapped_cell_output_size + self.attention_size
        raise RuntimeError(  # already validated in __init__
            "got unexpected output_mode: {}".format(self.output_mode))

    def call(self, inputs, states, constants, training=None):
        """Complete attentive cell transformation.
        """
        attended = constants
        attended_mask = _collect_previous_mask(attended)
        # attended and mask are always lists for uniformity:
        if not isinstance(attended_mask, list):
            attended_mask = [attended_mask]
        cell_states = states[:self._num_wrapped_states]
        attention_states = states[self._num_wrapped_states:]

        if self.attend_after:
            call = self._call_attend_after
        else:
            call = self._call_attend_before

        return call(inputs=inputs,
                    cell_states=cell_states,
                    attended=attended,
                    attention_states=attention_states,
                    attended_mask=attended_mask,
                    training=training)

    def _call_attend_before(self,
                            inputs,
                            cell_states,
                            attended,
                            attention_states,
                            attended_mask,
                            training=None):
        """Complete attentive cell transformation, if `attend_after=False`.
        """
        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=cell_states,
            attended=attended,
            attention_states=attention_states,
            attended_mask=attended_mask,
            training=training)

        cell_input = self._get_cell_input(inputs, attention_h)

        if has_arg(self.cell.call, 'training'):
            cell_output, new_cell_states = self.cell.call(
                cell_input, cell_states, training=training)
        else:
            cell_output, new_cell_states = self.cell.call(cell_input, cell_states)

        output = self._get_output(cell_output, attention_h)

        return output, new_cell_states + new_attention_states

    def _call_attend_after(self,
                           inputs,
                           cell_states,
                           attended,
                           attention_states,
                           attended_mask,
                           training=None):
        """Complete attentive cell transformation, if `attend_after=True`.
        """
        attention_h_previous = attention_states[0]

        cell_input = self._get_cell_input(inputs, attention_h_previous)

        if has_arg(self.cell.call, 'training'):
            cell_output, new_cell_states = self.cell.call(
                cell_input, cell_states, training=training)
        else:
            cell_output, new_cell_states = self.cell.call(cell_input, cell_states)

        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=new_cell_states,
            attended=attended,
            attention_states=attention_states,
            attended_mask=attended_mask,
            training=training)

        output = self._get_output(cell_output, attention_h)

        return output, new_cell_states, new_attention_states

    def _get_cell_input(self, inputs, attention_h):
        if self.input_mode == self._REPLACE:
            return attention_h
        if self.input_mode == self._CONCATENATE:
            return concatenate([inputs, attention_h])
        raise RuntimeError(  # already validated in __init__
            "got unexpected input_mode: {}".format(self.input_mode))

    def _get_output(self, cell_output, attention_h):
        if self.output_mode == self._CELL_OUTPUT:
            return cell_output
        if self.output_mode == self._CONCATENATE:
            return concatenate([cell_output, attention_h])
        raise RuntimeError(  # already validated in __init__
            "got unexpected output_mode: {}".format(self.output_mode))

    @staticmethod
    def _num_elements(x):
        if hasattr(x, '__len__'):
            return len(x)
        else:
            return 1

    @property
    def _num_wrapped_states(self):
        return self._num_elements(self.cell.state_size)

    @property
    def _num_attention_states(self):
        return self._num_elements(self.attention_state_size)

    @property
    def _wrapped_cell_output_size(self):
        if hasattr(self.cell, "output_size"):
            return self.cell.output_size
        if hasattr(self.cell.state_size, '__len__'):
            return self.cell.state_size[0]
        return self.cell.state_size

    def build(self, input_shape):
        """Builds attention mechanism and wrapped cell (if keras layer).

        Arguments:
            input_shape: list of tuples of integers, the input feature shape
                (inputs sequence shape without time dimension) followed by
                constants (i.e. attended) shapes.
        """
        if not isinstance(input_shape, list):
            raise ValueError('input shape should contain shape of both cell '
                             'inputs and constants (attended)')

        attended_shape = input_shape[1:]
        input_shape = input_shape[0]
        self.attended_spec = [InputSpec(shape=shape) for shape in attended_shape]
        if isinstance(self.cell.state_size, int):
            cell_state_size = [self.cell.state_size]
        else:
            cell_state_size = list(self.cell.state_size)
        self.attention_build(
            input_shape=input_shape,
            cell_state_size=cell_state_size,
            attended_shape=attended_shape,
        )

        if isinstance(self.cell, Layer):
            if self.input_mode == self._REPLACE:
                cell_input_size = self._attention_size
            elif self.input_mode == self._CONCATENATE:
                cell_input_size = self.attention_size + input_shape[-1]
            else:
                raise RuntimeError(  # already validated in __init__
                    "got unexpected input_mode: {}".format(self.input_mode))

            cell_input_shape = (input_shape[0], cell_input_size)
            self.cell.build(cell_input_shape)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size

    @property
    def trainable_weights(self):
        return (super(AttentionCellWrapper, self).trainable_weights +
                self.cell.trainable_weights)

    @property
    def non_trainable_weights(self):
        return (super(AttentionCellWrapper, self).non_trainable_weights +
                self.cell.non_trainable_weights)

    def get_config(self):
        config = {'attend_after': self.attend_after,
                  'input_mode': self.input_mode,
                  'output_mode': self.output_mode}

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(AttentionCellWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseAnnotationAttention(AttentionCellWrapper):
    """Recurrent attention mechanism for attending sequences.

    This class implements the attention mechanism used in [1] for machine
    translation. It is, however, a generic sequence attention mechanism that can be
    used for other sequence-to-sequence problems.

    As any recurrent attention mechanism extending `_RNNAttentionCell`, this class
    should be used in conjunction with a wrapped (non attentive) RNN Cell, such as
    the `SimpleRNNCell`, `LSTMCell` or `GRUCell`. It modifies the input of the
    wrapped cell by attending to a constant sequence (i.e. independent of the time
    step of the recurrent application of the attention mechanism). The attention
    encoding is obtained by computing a scalar weight for each time step of the
    attended by applying two stacked Dense layers to the concatenation of the
    attended feature vector at the respective time step with the previous state of
    the RNN Cell. The attention encoding is the weighted sum of the attended feature
    vectors using these weights.

    Half of the first Dense transformation is independent of the RNN Cell state and
    can be computed once for the attended sequence. Therefore this transformation
    should be computed externally of the attentive RNN Cell (for efficiency) and this
    layer expects both the attended sequence and the output of a Dense transformation
    of the attended sequence (see Example below). The number of hidden units of the
    attention mechanism is subsequently defined by the number of units of this
    (external) dense transformation.

    # Arguments
        cell: A RNN cell instance. The wrapped RNN cell wrapped by this attention
            mechanism. See docs of `cell` argument in the `RNN` Layer for further
            details.
        kernel_initializer: Initializer for all weights matrices
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for all bias vectors
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            all weights matrices. (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to all biases
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            all weights matrices. (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to all bias vectors
            (see [constraints](../constraints.md)).

     # Example

    ```python
        # machine translation (similar to the architecture used in [1])
        x = Input((None,), name="input_sequences")
        y = Input((None,), name="target_sequences")
        x_emb = Embedding(INPUT_NUM_WORDS, 256, mask_zero=True)(x)
        y_emb = Embedding(TARGET_NUM_WORDS, 256, mask_zero=True)(y)
        encoder = Bidirectional(GRU(512, return_sequences=True))
        x_enc = encoder(x_emb)

        # first part of the dense annotation, independent of the decoder time step
        u = TimeDistributed(Dense(128, use_bias=False))(x_enc)
        decoder = RNN(cell=DenseAnnotationAttention(cell=GRUCell(512)),
                      return_sequences=True)
        h = decoder(y_emb, constants=[x_enc, u])
        y_pred = TimeDistributed(Dense(TARGET_NUM_WORDS, activation='softmax'))(h)
        model = Model([y, x], y_pred)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER)
    ```

    # References
    [1] Neural Machine Translation by Jointly Learning to Align and Translate
        https://arxiv.org/abs/1409.0473
    """
    def __init__(self, cell,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DenseAnnotationAttention, self).__init__(cell, **kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       attended_mask,
                       training=None):
        # there must be two attended sequences (verified in build)
        [attended, u] = attended
        attended_mask = attended_mask[0]
        h_cell_tm1 = cell_states[0]

        # compute attention weights
        w = K.repeat(K.dot(h_cell_tm1, self.W_a) + self.b_UW, K.shape(attended)[1])
        e = K.exp(K.dot(K.tanh(w + u), self.v_a) + self.b_v)

        if attended_mask is not None:
            e = e * K.cast(K.expand_dims(attended_mask, -1), K.dtype(e))

        # weighted average of attended
        a = e / K.sum(e, axis=1, keepdims=True)
        c = K.sum(a * attended, axis=1, keepdims=False)

        return c, [c]

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        if not len(attended_shape) == 2:
            raise ValueError('There must be two attended tensors')
        for a in attended_shape:
            if not len(a) == 3:
                raise ValueError('only support attending tensors with dim=3')
        [attended_shape, u_shape] = attended_shape

        # NOTE _attention_size must always be set in `attention_build`
        self._attention_size = attended_shape[-1]
        units = u_shape[-1]

        kernel_kwargs = dict(initializer=self.kernel_initializer,
                             regularizer=self.kernel_regularizer,
                             constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(cell_state_size[0], units),
                                   name='W_a', **kernel_kwargs)
        self.v_a = self.add_weight(shape=(units, 1),
                                   name='v_a', **kernel_kwargs)

        bias_kwargs = dict(initializer=self.bias_initializer,
                           regularizer=self.bias_regularizer,
                           constraint=self.bias_constraint)
        self.b_UW = self.add_weight(shape=(units,),
                                    name="b_UW", **bias_kwargs)
        self.b_v = self.add_weight(shape=(1,),
                                   name="b_v", **bias_kwargs)

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseAnnotationAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    DATA_DIR = 'data/wmt16_mmt'
    FROM_LANGUAGE = 'en'
    TO_LANGUAGE = 'de'

    # Set meta parameters
    MAX_UNIQUE_WORDS = 30000
    MAX_WORDS_PER_SENTENCE = 40  # inf in [1]
    EMBEDDING_SIZE = 620  # `m` in [1]
    RECURRENT_UNITS = 1000  # `n` in [1]
    DENSE_ATTENTION_UNITS = 1000  # fixed equal to `n` in [1]
    READOUT_HIDDEN_UNITS = 500  # `l` in [1]
    OPTIMIZER = Adadelta(rho=0.95, epsilon=1e-6, clipnorm=1.)
    BATCH_SIZE = 80
    EPOCHS = 20  # only 5 epochs in [1] but a orders of magnitude larger dataset

    # Load and tokenize the data
    start_token = "'start'"
    end_token = "'end'"
    # NOTE: using single quotes (which are not dropped by Tokenizer by default)
    # for the tokens to be distinguished from other use of "start" and "end"

    def get_sentences(partion, language):
        fpath = os.path.join(DATA_DIR, partion + '.' + language)
        with open(fpath, 'r') as f:
            sentences = f.readlines()
        return ["{} {} {}".format(start_token, sentence.replace('\n', ''), end_token)
                for sentence in sentences]

    input_texts_train = get_sentences("train", FROM_LANGUAGE)
    input_texts_val = get_sentences("val", FROM_LANGUAGE)
    target_texts_train = get_sentences("train", TO_LANGUAGE)
    target_texts_val = get_sentences("val", TO_LANGUAGE)

    input_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS, oov_token='?')
    target_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS, oov_token='?')
    input_tokenizer.fit_on_texts(input_texts_train)
    target_tokenizer.fit_on_texts(target_texts_train)

    input_seqs_train = input_tokenizer.texts_to_sequences(input_texts_train)
    input_seqs_val = input_tokenizer.texts_to_sequences(input_texts_val)
    target_seqs_train = target_tokenizer.texts_to_sequences(target_texts_train)
    target_seqs_val = target_tokenizer.texts_to_sequences(target_texts_val)

    # we need to know the largest index to set size of embeddings and output layer
    input_num_words = max([max(s) for s in input_seqs_train]) + 1
    target_num_words = max([max(s) for s in target_seqs_train]) + 1
    # Note that the Tokenizer does't have a property for this, but below also works:
    # input_num_words = min(
    #     max(input_tokenizer.word_index.values()) + 1,
    #     input_tokenizer.num_words
    # )
    input_seqs_train, input_seqs_val, target_seqs_train, target_seqs_val = (
        pad_sequences(seq, maxlen=MAX_WORDS_PER_SENTENCE,
                      padding='post', truncating='post')
        for seq in [input_seqs_train,
                    input_seqs_val,
                    target_seqs_train,
                    target_seqs_val])

    # Build model
    x = Input((None,), name="input_sequences")
    y = Input((None,), name="target_sequences")
    x_emb = Embedding(input_num_words, EMBEDDING_SIZE, mask_zero=True)(x)
    y_emb = Embedding(target_num_words, EMBEDDING_SIZE, mask_zero=True)(y)

    encoder_rnn = Bidirectional(GRU(RECURRENT_UNITS,
                                    return_sequences=True,
                                    return_state=True))
    x_enc, h_enc_fwd_final, h_enc_bkw_final = encoder_rnn(x_emb)

    # half of the dense annotation can be computed one per input sequence sine it is
    # independent of the RNN state
    u = TimeDistributed(Dense(DENSE_ATTENTION_UNITS, use_bias=False))(x_enc)

    # the final state of the backward-GRU (closest to the start of the input
    # sentence) is used to initialize the state of the decoder
    initial_state_gru = Dense(RECURRENT_UNITS, activation='tanh')(h_enc_bkw_final)
    initial_attention_h = Lambda(lambda _x: K.zeros_like(_x)[:, 0, :])(x_enc)
    initial_state = [initial_state_gru, initial_attention_h]

    cell = DenseAnnotationAttention(cell=GRUCell(RECURRENT_UNITS),
                                    input_mode="concatenate",
                                    output_mode="concatenate")
    decoder_rnn = RNN(cell=cell, return_sequences=True)
    h1 = decoder_rnn(y_emb, initial_state=initial_state, constants=[x_enc, u])

    def maxout(x_):
        """Maxout "activation" where max is taken over _two_ units"""
        x_1 = x_[:, :READOUT_HIDDEN_UNITS]
        x_2 = x_[:, READOUT_HIDDEN_UNITS:]
        return K.max(K.stack([x_1, x_2], axis=-1), axis=-1, keepdims=False)

    _h2 = TimeDistributed(Dense(READOUT_HIDDEN_UNITS * 2))(concatenate([h1, y_emb]))
    h2 = TimeDistributed(Lambda(maxout))(_h2)
    y_pred = TimeDistributed(Dense(target_num_words, activation='softmax'))(h2)

    model = Model([y, x], y_pred)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER)

    # Run training
    model.fit([target_seqs_train[:, :-1], input_seqs_train],
              target_seqs_train[:, 1:, None],
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(
                  [target_seqs_val[:, :-1], input_seqs_val],
                  target_seqs_val[:, 1:, None]))
    # Save model
    model.save('rec_att_mt.h5')

    # Inference: use to model to translate new sentences!
    # Greedy and Beam Search inference is implemented below. Note that in both
    # approaches the input and entire history of outputs are processed for every
    # token that is generated. A more computationally efficient way is to decompose
    # the model into one encoding model (that is ran once per input text) and one
    # decoding model that returns its recurrent state, so that it can be applied
    # iteratively on only the latest generated token. This is left out here for
    # brevity.

    def translate_greedy(input_text, t_max=None):
        """Select the next token highest predicted likelihood iteratively.
        """
        t = 0
        y_ = np.array(target_tokenizer.texts_to_sequences([start_token]))
        x_ = np.array(input_tokenizer.texts_to_sequences([input_text]))

        if t_max is None:
            t_max = x_.shape[-1] * 2
        end_idx = target_tokenizer.word_index[end_token]
        score = 0  # track the cumulative log likelihood
        while y_[0, -1] != end_idx and t < t_max:
            y_pred_ = model.predict([y_, x_])
            t += 1
            y_t = np.argmax(y_pred_[:, -1:], axis=-1)
            score += np.log(y_pred_[0, 0, y_t[0, 0]])
            y_ = np.hstack([y_, y_t])
        output_text = target_tokenizer.sequences_to_texts(y_)[0]
        # length normalised score, skipping start token
        score = score / (y_.shape[1] - 1)

        return output_text, score


    def translate_beam_search(input_text,
                              search_width=20,
                              branch_factor=None,
                              t_max=None):
        """Perform beam search to (approximately) find the translated sentence that
        maximises the conditional probability given the input sequence.

        Returns the completed sentences (reached end-token) in order of decreasing
        score (the first is most probable) followed by incomplete sentences in order
        of decreasing score - as well as the score for the respective sentence.

        References:
            [1] "Sequence to sequence learning with neural networks"
            (https://arxiv.org/pdf/1409.3215.pdf)
        """

        if branch_factor is None:
            branch_factor = search_width
        elif branch_factor > search_width:
            raise ValueError("branch_factor must be smaller than search_width")
        elif branch_factor < 2:
            raise ValueError("branch_factor must be >= 2")

        def k_largest_val_idx(a, k):
            """Returns top k largest values of a and their indices, ordered by
            decreasing value"""
            top_k = np.argpartition(a, -k)[-k:]
            return sorted(zip(a[top_k], top_k))[::-1]

        class Beam(object):
            """A Beam holds the tokens seen so far and its accumulated score
            (log likelihood).
            """
            def __init__(self, sequence, scores=(0,)):
                self._sequence = list(sequence)
                self.scores = list(scores)

            @property
            def sequence(self):
                return np.array(self._sequence)

            @property
            def score(self):
                return sum(self.scores)

            def get_child(self, element, score):
                return Beam(self._sequence + [element], self.scores + [score])

            def __len__(self):
                return len(self._sequence)

            def __lt__(self, other):
                return self.score < other.score

            def __gt__(self, other):
                return other.score > other.score

        x_ = np.array(input_tokenizer.texts_to_sequences([input_text]))
        x_ = np.repeat(x_, search_width, axis=0)

        if t_max is None:
            t_max = x_.shape[-1] * 2

        start_idx = target_tokenizer.word_index[start_token]
        end_idx = target_tokenizer.word_index[end_token]

        # A list of the <search_width> number of beams with highest score is
        # maintained through out the search. Initially there is only one beam.
        incomplete_beams = [Beam(sequence=[start_idx], scores=[0])]
        # All beams that reached the end-token are kept separately.
        complete_beams = []

        t = 0
        while len(complete_beams) < search_width and t < t_max:
            t += 1
            # create a batch of inputs representing the incomplete_beams
            y_ = np.vstack([beam.sequence for beam in incomplete_beams])
            # predict next tokes for every incomplete beam
            batch_size = len(incomplete_beams)
            y_pred_ = model.predict([y_, x_[:batch_size]])
            # from each previous beam create new candidate beams and save the once
            # with highest score for next iteration.
            beams_updated = []
            for i, beam in enumerate(incomplete_beams):
                for proba, idx in k_largest_val_idx(y_pred_[i, -1], branch_factor):
                    new_beam = beam.get_child(element=idx, score=np.log(proba))
                    if idx == end_idx:
                        # beam completed
                        complete_beams.append(new_beam)
                    elif len(beams_updated) < search_width:
                        # not full search width
                        heapq.heappush(beams_updated, new_beam)
                    elif new_beam.score > beams_updated[0].score:
                        # better than candidate with lowest score
                        heapq.heapreplace(beams_updated, new_beam)
                    else:
                        # if score is worse that existing candidates we abort search
                        # for children of this beam (since next token processed
                        # in order of decreasing likelihood)
                        break

            # faster to process beams in order of decreasing score next iteration,
            # due to break above
            incomplete_beams = sorted(beams_updated, reverse=True)

        # want to return in order of decreasing score
        complete_beams = sorted(complete_beams, reverse=True)

        output_texts = []
        output_scores = []
        for beam in complete_beams + incomplete_beams:
            text = target_tokenizer.sequences_to_texts(beam.sequence[None])[0]
            output_texts.append(text)
            # average score, skipping start token
            output_scores.append(beam.score / (len(beam) - 1))

        return output_texts, output_scores

    # Translate some sentences from validation data
    for input_text, target_text in zip(input_texts_val[:5], target_texts_val[:5]):
        print("Translating: ", input_text)
        print("Expected: ", target_text)
        output_greedy, score_greedy = translate_greedy(input_text)
        print("Greedy output: ", output_greedy)
        outputs_beam, scores_beam = translate_beam_search(input_text)
        print("Beam search outputs (top 5):")
        for beam in outputs_beam[:5]:
            print("\t" + beam)
