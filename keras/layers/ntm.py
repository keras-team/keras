import numpy as np

from .. import backend as K
from keras.layers.recurrent import Recurrent, GRU, LSTM

tol = 1e-4


def _update_controller(self, inp, h_tm1, M):
    """ Update inner RNN controler
    We have to update the inner RNN inside the Neural Turing Machine, this
    is an almost literal copy of keras.layers.recurrent.GRU and
    keras.layers.recurrent.LSTM see these clases for further details.
    """
    x = K.concatenate([inp, M], axis=-1)
    # update state
    _, h = self.rnn.step(x, h_tm1)

    return h


def _circulant(leng, n_shifts):
    """ Generate circulant copies of a vector.
    This will generate a tensor with `n_shifts` of rotated versions the
    identity matrix. When this tensor is multiplied by a vector
    the result are `n_shifts` shifted versions of that vector. Since
    everything is done with inner products, this operation is differentiable.
    Paramters:
    ----------
    leng: int > 0, number of memory locations
    n_shifts: int > 0, number of allowed shifts (if 1, no shift)
    Returns:
    --------
    shift operation, a tensor with dimensions (n_shifts, leng, leng)
    """
    eye = np.eye(leng)
    shifts = range(n_shifts // 2, -n_shifts // 2, -1)
    C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
    return K.variable(C)


def _renorm(x):
    return x / (K.sum(x, axis=1, keepdims=True))


def _softmax(x):
    wt = K.flatten(x)
    w = K.softmax(wt)
    return w.reshape(x.shape)


def _cosine_distance(M, k):
    dot = K.sum(M * K.expand_dims(k, 1), axis=-1)
    nM = K.sum(K.sqrt(M ** 2), axis=-1)
    nk = K.sum(K.sqrt(k ** 2), axis=-1, keepdims=True)
    return dot / (nM * nk)


class NeuralTuringMachine(Recurrent):
    """ Neural Turing Machines
    Parameters:
    -----------
    shift_range: int, number of available shifts, ex. if 3, avilable shifts are
                 (-1, 0, 1)
    n_slots: number of memory locations
    m_length: memory length at each location
    inner_rnn: str, supported values are 'gru' and 'lstm'
    output_dim: hidden state size (RNN controller output_dim)
    Known issues and TODO:
    ----------------------
    Theano may complain when n_slots == 1.
    Add multiple reading and writing heads.
    """
    def __init__(self, output_dim, n_slots=128, m_length=20, shift_range=3,
                 inner_rnn='gru', truncate_gradient=-1, return_sequences=False,
                 init='glorot_uniform', inner_init='orthogonal',
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.n_slots = n_slots
        self.m_length = m_length
        self.shift_range = shift_range
        self.init = init
        self.inner_init = inner_init
        self.inner_rnn = inner_rnn
        self.return_sequences = return_sequences
        self.truncate_gradient = truncate_gradient

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(NeuralTuringMachine, self).__init__(**kwargs)

    def build(self):
        input_leng, input_dim = self.input_shape[1:]
        self.input = K.placeholder(shape=self.input_shape)

        if self.inner_rnn == 'gru':
            self.rnn = GRU(
                input_dim=input_dim + self.m_length,
                input_length=input_leng,
                output_dim=self.output_dim, init=self.init,
                inner_init=self.inner_init)
        elif self.inner_rnn == 'lstm':
            self.rnn = LSTM(
                input_dim=input_dim + self.m_length,
                input_length=input_leng,
                output_dim=self.output_dim, init=self.init,
                inner_init=self.inner_init)
        else:
            raise ValueError('this inner_rnn is not implemented yet.')

        self.rnn.build()

        # initial memory, state, read and write vecotrs
        self.M = K.variable(.001 * np.ones((1,)))
        self.init_h = K.zeros((self.output_dim))
        self.init_wr = self.rnn.init((self.n_slots,))
        self.init_ww = self.rnn.init((self.n_slots,))

        # write
        self.W_e = self.rnn.init((self.output_dim, self.m_length))  # erase
        self.b_e = K.zeros((self.m_length))
        self.W_a = self.rnn.init((self.output_dim, self.m_length))  # add
        self.b_a = K.zeros((self.m_length))

        # get_w  parameters for reading operation
        self.W_k_read = self.rnn.init((self.output_dim, self.m_length))
        self.b_k_read = self.rnn.init((self.m_length, ))
        self.W_c_read = self.rnn.init((self.output_dim, 3))  # 3 = beta, g, gamma see eq. 5, 7, 9 in Graves et. al 2014
        self.b_c_read = K.zeros((3))
        self.W_s_read = self.rnn.init((self.output_dim, self.shift_range))
        self.b_s_read = K.zeros((self.shift_range))

        # get_w  parameters for writing operation
        self.W_k_write = self.rnn.init((self.output_dim, self.m_length))
        self.b_k_write = self.rnn.init((self.m_length, ))
        self.W_c_write = self.rnn.init((self.output_dim, 3))  # 3 = beta, g, gamma see eq. 5, 7, 9
        self.b_c_write = K.zeros((3))
        self.W_s_write = self.rnn.init((self.output_dim, self.shift_range))
        self.b_s_write = K.zeros((self.shift_range))

        self.C = _circulant(self.n_slots, self.shift_range)

        self.params = self.rnn.params + [
            self.W_e, self.b_e,
            self.W_a, self.b_a,
            self.W_k_read, self.b_k_read,
            self.W_c_read, self.b_c_read,
            self.W_s_read, self.b_s_read,
            self.W_k_write, self.b_k_write,
            self.W_s_write, self.b_s_write,
            self.W_c_write, self.b_c_write,
            self.M,
            self.init_h, self.init_wr, self.init_ww]

        if self.inner_rnn == 'lstm':
            self.init_c = K.zeros((self.output_dim))
            self.params = self.params + [self.init_c, ]

    def _read(self, w, M):
        return (K.expand_dims(w, 2) * M).sum(axis=1)

    def _write(self, w, e, a, M):
        w_exp = K.expand_dims(w, 2)
        Mtilda = M * (1 - w_exp * K.expand_dims(e, 1))
        Mout = Mtilda + w_exp * K.expand_dims(a, 1)
        return Mout

    def _get_content_w(self, beta, k, M):
        num = K.expand_dims(beta, 1) * _cosine_distance(M, k)
        return _softmax(num)

    def _get_location_w(self, g, s, C, gamma, wc, w_tm1):
        g = K.expand_dims(g, 1)
        gamma = K.expand_dims(gamma, 1)
        s = K.expand_dims(s, 2)
        C = K.expand_dims(C, 0)

        wg = g * wc + (1 - g) * w_tm1
        wg = K.expand_dims(K.expand_dims(wg, 1), 2)
        Cs = K.sum(C * wg, axis=3)
        wtilda = K.sum(Cs * s, axis=1)
        wout = _renorm(wtilda ** gamma)
        return wout

    def _get_controller_output(self, h, W_k, b_k, W_c, b_c, W_s, b_s):
        k = K.tanh(K.dot(h, W_k) + b_k)
        c = K.dot(h, W_c) + b_c
        beta = K.relu(c[:, 0]) + 1e-6
        g = K.sigmoid(c[:, 1])
        gamma = K.relu(c[:, 2]) + 1
        s = K.softmax(K.dot(h, W_s) + b_s)
        return k, beta, g, gamma, s

    def _get_initial_states(self, X):
        batch_size = K.sum(K.sum(K.zeros_like(X), axis=1), axis=1)  # (samples, )
        batch_size_dim = K.expand_dims(batch_size, axis=1)
        init_M = self.M + batch_size
        init_M = K.repeat(K.repeat(self.M,
                                   self.n_slots, axis=1),
                          self.m_length, axis=2)

        init_h = batch_size_dim * K.expand_dims(self.init_h, axis=0)
        init_wr = batch_size_dim * K.expand_dims(self.init_wr, axis=0)
        init_ww = batch_size_dim * K.expand_dims(self.init_ww, axis=0)
        if self.inner_rnn == 'lstm':
            init_c = batch_size_dim * K.expand_dims(self.init_c, axis=0)
            return [init_M, K.softmax(init_wr), K.softmax(init_ww), init_h,
                    init_c]
        else:
            return [init_M, K.softmax(init_wr), K.softmax(init_ww), init_h]

    def step(self, x, states):
        # read
        if self.inner_rnn == 'lstm':
            assert len(states) == 5
            M_tm1, wr_tm1, ww_tm1, h_tm1, cell_tm1 = states
        else:
            assert len(states) == 4
            M_tm1, wr_tm1, ww_tm1, h_tm1 = states
        k_read, beta_read, g_read, gamma_read, s_read = self._get_controller_output(
            h_tm1, self.W_k_read, self.b_k_read, self.W_c_read, self.b_c_read,
            self.W_s_read, self.b_s_read)
        wc_read = self._get_content_w(beta_read, k_read, M_tm1)
        wr_t = self._get_location_w(g_read, s_read, self.C, gamma_read,
                                    wc_read, wr_tm1)
        M_read = self._read(wr_t, M_tm1)

        # update controller
        if self.inner_rnn == 'lstm':
            h_t, cell_t = _update_controller(self, x, [h_tm1, cell_tm1], M_read)
            states = [h_t, cell_t]
        else:
            h_t = _update_controller(self, x, [h_tm1, ], M_read)
            states = [h_t, ]

        # write
        k_write, beta_write, g_write, gamma_write, s_write = self._get_controller_output(
            h_t, self.W_k_write, self.b_k_write, self.W_c_write,
            self.b_c_write, self.W_s_write, self.b_s_write)
        wc_write = self._get_content_w(beta_write, k_write, M_tm1)
        ww_t = self._get_location_w(g_write, s_write, self.C, gamma_write,
                                    wc_write, ww_tm1)
        e = K.sigmoid(K.dot(h_t, self.W_e) + self.b_e)
        a = K.tanh(K.dot(h_t, self.W_a) + self.b_a)
        M_t = self._write(ww_t, e, a, M_tm1)

        return h_t[1], [M_t, wr_t, ww_t] + states

    def get_output(self, train=False):
        outputs = self.get_full_output(train)

        if self.return_sequences:
            return outputs[-1]
        else:
            return outputs[-1][:, -1]

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim

    def get_full_output(self, train=False):
        """
        This method is for research and visualization purposes. Use it as:
        X = model.get_input()  # full model
        Y = ntm.get_output()    # this layer
        F = theano.function([X], Y, allow_input_downcast=True)
        [memory, read_address, write_address, rnn_state] = F(x)
        if inner_rnn == "lstm" use it as:
        [memory, read_address, write_address, rnn_cell, rnn_state] = F(x)
        """
        X = self.get_input(train)
        mask = self.get_output_mask(train)
        if mask:
            # apply mask
            X *= K.expand_dims(mask)
            masking = True
        else:
            masking = False

        init_states = self._get_initial_states(X)
        last_output, outputs, states = K.rnn(
            self.step,
            X, init_states,
            go_backwards=self.go_backwards, masking=masking)

        out = [K.permute_dimensions(outputs[0], (1, 0, 2, 3)),
               K.permute_dimensions(outputs[1], (1, 0, 2)),
               K.permute_dimensions(outputs[2], (1, 0, 2)),
               K.permute_dimensions(outputs[3], (1, 0, 2))]
        if self.inner_rnn == 'lstm':
            out + [K.permute_dimensions(outputs[4], (1, 0, 2))]
        return out
