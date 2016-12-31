import keras
from keras.layers import Layer, InputSpec
from keras import initializations, activations, regularizers
import keras.backend as K
from keras.objectives import categorical_crossentropy, sparse_categorical_crossentropy

if K._BACKEND == 'tensorflow':
    import tensorflow as tf
else:
    import theano
    import theano.tensor as T


class CRF(Layer):
    '''An implementation of linear chain conditional random field (CRF).

    An linear chain CRF is defined to maximize the following likelihood function:

    $$ L(W, U, b; y_1, ..., y_n) := \frac{1}{Z} \sum_{y_1, ..., y_n} \exp(-a_1' y_1 - a_n' y_n
        - \sum_{k=1^n}(((x_k' W + b) y_k) + y_1' U y_2)), $$

    where:
        $Z$: normalization constant
        $x_k, y_k$:  inputs and outputs

    This implementation has two modes for optimization:
    1. (`join mode`) optimized by maximizing join likelihood, which is optimal in theory of statistics.
       Note that in this case, CRF mast be the output/last layer.
    2. (`marginal mode`) return marginal probabilities on each time step and optimized via composition
       likelihood (product of marginal likelihood), i.e., using `categorical_crossentropy` loss.
       Note that in this case, CRF can be either the last layer or an intermediate layer.

    For prediction (test phrase), one can choose either Viterbi best path (class indices) or marginal
    probabilities if probabilities are needed. However, if one chooses *join mode* for training,
    Viterbi output is typically better than marginal output, but the marginal output will still perform
    reasonably close, while if *marginal mode* is used for training, marginal output usually performs
    much better. The default behavior is set according to this observation.

    In addition, this implementation supports masking and accepts either one-hot or sparse target.


    # Examples

    ```python
        X = Input((sent_len,), dtype='int32', name='input_x')
        Embed = Embedding(3001, embed_dim, mask_zero=True)(X)

        # use learn_mode = 'join', test_mode = 'viterbi'
        crf = CRF(10)

        Y = crf(Embed)
        model = Model(X, Y)

        # crf.accuracy is default to Viterbi acc if using join-mode (default).
        # One can add crf.marginal_acc if interested, but may slow down learning
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

        # y_label can be either one-hot representation or label indices (with shape 1 at dim 3)
        model.fit(x, y_label, batch_size=100, nb_epoch=5)

        # prediction give one-hot representation of Viterbi best path
        y_hat = model.predict(x_test)
    ```


    # Arguments
        output_dim: dimension of layer output or number of classes.
        learn_mode: either 'join' or 'marginal'.
            The former train the model by maximizing join likelihood while the latter
            maximize the product of marginal likelihood over all time steps.
        test_mode: either 'viterbi' or 'marginal'.
            The former is recommended and as default when `learn_mode = 'join'` and
            gives one-hot representation of the best path at test (prediction) time,
            while the latter is recommended and chosen as default when `learn_mode = 'marginal'`,
            which produces marginal probabilities for each time step.
        sparse_target: boolen (default False) indicating if provided labels are one-hot or
            indices (with shape 1 at dim 3).
        in_init, chain_init:
            Initialization for input weight matrix and chain connecting matrix (W, U above).
        in_activation, chain_activation:
            Transforms for input and chain energy (f, g above). Both default to linear
            Indeed, these functions are used as range regulations.
            E.g., a `tanh` forces the input or chain energy to be bounded within [-1, 1].
        W_regularizer, U_regularizer, b_regularizer:
            Instances of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrix,
            chain connecting matrix and input bias.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified when it is constant.
        unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used.
            Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        - 3D tensor with shape `(nb_samples, timesteps, output_dim)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    '''

    def __init__(self, output_dim,
                 learn_mode='join', test_mode=None,
                 boundary_energy=False, sparse_target=False,
                 in_init='orthogonal', chain_init='orthogonal',
                 in_activation='linear', chain_activation='linear',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 weights=None, input_dim=None, input_length=None, unroll=False, **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        assert learn_mode in ['join', 'marginal']
        self.learn_mode = learn_mode
        self.test_mode = test_mode
        if test_mode is None:
            self.test_mode = 'viterbi' if learn_mode == 'join' else 'marginal'
        else:
            assert test_mode in ['viterbi', 'marginal']
        self.boundary_energy = boundary_energy
        self.sparse_target = sparse_target
        self.in_init = initializations.get(in_init)
        self.chain_init = initializations.get(chain_init)
        self.in_activation = activations.get(in_activation)
        self.chain_activation = activations.get(chain_activation)
        self.initial_weights = weights
        self.unroll = unroll

        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.input_length is None:
            self.input_length = self.input_spec[0].shape[1]
        input_dim = input_shape[-1]
        self.W = self.in_init((input_dim, self.output_dim), name='{}_W'.format(self.name))
        self.U = self.chain_init((self.output_dim, self.output_dim), name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
        self.trainable_weights = [self.W, self.U, self.b]
        if self.boundary_energy:
            self.a1 = K.zeros((self.output_dim,), name='{}_a1'.format(self.name))
            self.an = K.zeros((self.output_dim,), name='{}_an'.format(self.name))
            self.trainable_weights += [self.a1, self.an]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, X, mask=None):
        if mask is not None:
            assert K.ndim(mask) == 2, 'Input mask to CRF must have dim 2 if not None'

        if self.test_mode == 'viterbi':
            test_output = self.viterbi_decoding(X, mask)
        else:
            test_output = self.get_marginal_prob(X, mask)

        self.uses_learning_phase = True
        if self.learn_mode == 'join':
            train_output = K.zeros_like(K.dot(X, self.W))
            out = K.in_train_phase(train_output, test_output)
        else:
            if self.test_mode == 'viterbi':
                train_output = self.get_marginal_prob(X, mask)
                out = K.in_train_phase(train_output, test_output)
            else:
                out = test_output
        return out

    def get_output_shape_for(self, input_shape):
        return input_shape[:2] + (self.output_dim,)

    def compute_mask(self, input, mask=None):
        if mask is not None and self.learn_mode == 'join':
            return K.any(mask, axis=1)
        return mask

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'learn_mode': self.learn_mode,
                  'test_mode': self.test_mode,
                  'sparse_target': self.sparse_target,
                  'in_init': self.in_init.__name__,
                  'chain_init': self.chain_init.__name__,
                  'in_activation': self.in_activation.__name__,
                  'chain_activation': self.chain_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'input_length': self.input_length,
                  'input_dim': self.input_dim,
                  'unroll': self.unroll}
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def loss_function(self):
        if self.learn_mode == 'join':
            def loss(y_true, y_pred):
                assert self.inbound_nodes, 'CRF has not connected to any layer.'
                assert not self.outbound_nodes, 'When learn_model="join", CRF must be the last layer.'
                if self.sparse_target:
                    y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), self.output_dim)
                X = self.inbound_nodes[0].input_tensors[0]
                mask = self.inbound_nodes[0].input_masks[0]
                nloglik = self.get_nloglik(y_true, X, mask)
                return nloglik
            return loss
        else:
            if self.sparse_target:
                return sparse_categorical_crossentropy
            else:
                return categorical_crossentropy

    @property
    def accuracy(self):
        if self.test_mode == 'viterbi':
            return self.viterbi_acc
        else:
            return self.marginal_acc

    @staticmethod
    def _get_accuracy(y_true, y_pred, mask, sparse_target=False):
        y_pred = K.argmax(y_pred, -1)
        if sparse_target:
            y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
        else:
            y_true = K.argmax(y_true, -1)
        judge = K.cast(K.equal(y_pred, y_true), K.floatx())
        if mask is None:
            return K.mean(judge)
        else:
            mask = K.cast(mask, K.floatx())
            return K.sum(judge * mask) / K.sum(mask)

    @property
    def viterbi_acc(self):
        def acc(y_true, y_pred):
            X = self.inbound_nodes[0].input_tensors[0]
            mask = self.inbound_nodes[0].input_masks[0]
            y_pred = self.viterbi_decoding(X, mask)
            return self._get_accuracy(y_true, y_pred, mask, self.sparse_target)
        acc.func_name = 'viterbi_acc'
        return acc

    @property
    def marginal_acc(self):
        def acc(y_true, y_pred):
            X = self.inbound_nodes[0].input_tensors[0]
            mask = self.inbound_nodes[0].input_masks[0]
            y_pred = self.get_marginal_prob(X, mask)
            return self._get_accuracy(y_true, y_pred, mask, self.sparse_target)
        acc.func_name = 'marginal_acc'
        return acc

    @staticmethod
    def log_sum_exp(x, axis=-1):
        '''log(sum(exp(x)) = m + log(sum(exp(x-m))), where m = max(x, axis).'''
        m = K.max(x, axis=axis, keepdims=True)
        m_ = K.max(x, axis=axis)
        return m_ + K.log(K.sum(K.exp(x - m), axis=axis))

    @staticmethod
    def softmaxNd(x, axis=-1):
        m = K.max(x, axis=axis, keepdims=True)
        exp_x = K.exp(x - m)
        prob_x = exp_x / K.sum(exp_x, axis=axis, keepdims=True)
        return prob_x

    @staticmethod
    def shift_left(x, offset=1):
        assert offset > 0
        return K.concatenate([x[:, offset:], K.zeros_like(x[:, :offset])], axis=1)

    @staticmethod
    def shift_right(x, offset=1):
        assert offset > 0
        return K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)

    def add_boundary_energy(self, energy, mask, start, end):
        start = K.expand_dims(K.expand_dims(self.chain_activation(start), 0), 0)
        end = K.expand_dims(K.expand_dims(self.chain_activation(end), 0), 0)
        if mask is None:
            energy = K.concatenate([energy[:, :1, :] + start, energy[:, 1:, :]], axis=1)
            energy = K.concatenate([energy[:, :-1, :], energy[:, -1:, :] + end], axis=1)
        else:
            mask = K.expand_dims(K.cast(mask, K.floatx()))
            start_mask = K.cast(K.greater(mask, self.shift_right(mask)), K.floatx())
            end_mask = K.cast(K.greater(self.shift_left(mask), mask), K.floatx())
            energy = energy + start_mask * start
            energy = energy + end_mask * end
        return energy

    def get_logZ(self, in_energy, mask):
        '''Compute logarithm of the normalization constance Z, where
        Z = sum exp(-E) -> logZ = log sum exp(-E) =: -nlogZ
        '''
        # should have logZ[:, i] == logZ[:, j] for any i, j
        logZ = self.recursion(in_energy, mask, return_sequences=False)
        return logZ[:, 0]

    def get_energy(self, y_true, in_energy, mask):
        '''Energy = a1' y1 + u1' y1 + y1' U y2 + u2' y2 + y2' U y3 + u3' y3 + an' y3
        '''
        in_energy = K.sum(in_energy * y_true, 2) # (B, T)
        chain_energy = K.sum(K.dot(y_true[:, :-1, :], self.U) * y_true[:, 1:, :], 2)  # (B, T-1)
        chain_energy = self.chain_activation(chain_energy)

        if mask is not None:
            mask = K.cast(mask, K.floatx())
            chain_mask = mask[:, :-1] * mask[:, 1:] # (B, T-1), mask[:,:-1]*mask[:,1:] makes it work with any padding
            in_energy = in_energy * mask
            chain_energy = chain_energy * chain_mask
        total_energy = K.sum(in_energy, -1) + K.sum(chain_energy, -1) # (B, )

        return total_energy

    def get_nloglik(self, y_true, X, mask):
        '''Compute the loss, i.e., negative log likelihood (normalize by number of time steps)
           likelihood = 1/Z * exp(-E) ->  neg_log_like = - log(1/Z * exp(-E)) = logZ + E
        '''
        in_energy = self.in_activation(K.dot(X, self.W) + self.b)
        if self.boundary_energy:
            in_energy = self.add_boundary_energy(in_energy, mask, self.a1, self.an)
        energy = self.get_energy(y_true, in_energy, mask)
        logZ = self.get_logZ(in_energy, mask)
        nloglik = logZ + energy
        if mask is not None:
            nloglik = nloglik / K.sum(K.cast(mask, K.floatx()), 1)
        else:
            nloglik = nloglik / K.cast(K.shape(X)[1], K.floatx())
        return nloglik

    def step(self, in_energy_t, states, return_logZ=True):
        # not in the following  `prev_target_val` has shape = (B, F)
        # where B = batch_size, F = output feature dim
        # Note: `i` is of float32, due to the behavior of `K.rnn`
        prev_target_val, i, chain_energy = states[:3]
        t = K.cast(i[0, 0], dtype='int32')
        if len(states) > 3:
            if K._BACKEND == 'theano':
                m = states[3][:, t:(t+2)]
            else:
                m = tf.slice(states[3], [0, t], [-1, 2])
            in_energy_t = in_energy_t * K.expand_dims(m[:, 0])
            chain_energy = chain_energy * K.expand_dims(K.expand_dims(m[:, 0] * m[:, 1]))  # (1, F, F)*(B, 1, 1) -> (B, F, F)
        if return_logZ:
            energy = chain_energy + K.expand_dims(in_energy_t - prev_target_val, 2) # shapes: (1, B, F) + (B, F, 1) -> (B, F, F)
            new_target_val = self.log_sum_exp(-energy, 1) # shapes: (B, F)
            return new_target_val, [new_target_val, i + 1]
        else:
            energy = chain_energy + K.expand_dims(in_energy_t + prev_target_val, 2)
            min_energy = K.min(energy, 1)
            argmin_table = K.cast(K.argmin(energy, 1), K.floatx()) # cast for tf-version `K.rnn`
            return argmin_table, [min_energy, i + 1]

    def recursion(self, in_energy, mask=None, go_backwards=False, return_sequences=True, return_logZ=True):
        '''Forward (alpha) or backward (beta) recursion

        If `return_logZ = True`, compute the logZ, the normalization constance:

        \[ Z = \sum_{y1, y2, y3} exp(-E) # energy
          = \sum_{y1, y2, y3} exp(-(u1' y1 + y1' W y2 + u2' y2 + y2' W y3 + u3' y3))
          = sum_{y2, y3} (exp(-(u2' y2 + y2' W y3 + u3' y3)) sum_{y1} exp(-(u1' y1' + y1' W y2))) \]

        Denote:
            \[ S(y2) := sum_{y1} exp(-(u1' y1 + y1' W y2)), \]
            \[ Z = sum_{y2, y3} exp(log S(y2) - (u2' y2 + y2' W y3 + u3' y3)) \]
            \[ logS(y2) = log S(y2) = log_sum_exp(-(u1' y1' + y1' W y2)) \]
        Note that:
              yi's are one-hot vectors
              u1, u3: boundary energies have been merged

        If `return_logZ = False`, compute the Viterbi's best path lookup table.
        '''
        chain_energy = self.chain_activation(self.U)
        chain_energy = K.expand_dims(chain_energy, 0) # shape=(1, F, F): F=num of output features. 1st F is for t-1, 2nd F for t
        prev_target_val = K.zeros_like(in_energy[:, 0, :]) # shape=(B, F), dtype=float32

        if go_backwards:
            in_energy = K.reverse(in_energy, 1)
            if mask is not None:
                mask = K.reverse(mask, 1)

        initial_states = [prev_target_val, K.zeros_like(prev_target_val[:, :1])]
        constants = [chain_energy]

        if mask is not None:
            mask2 = K.cast(K.concatenate([mask, K.zeros_like(mask[:, :1])], axis=1), K.floatx())
            constants.append(mask2)

        def _step(in_energy_i, states):
            return self.step(in_energy_i, states, return_logZ)

        target_val_last, target_val_seq, _ = K.rnn(_step, in_energy, initial_states, constants=constants,
                                                   input_length=self.input_length, unroll=self.unroll)

        if return_sequences:
            if go_backwards:
                target_val_seq = K.reverse(target_val_seq, 1)
            return target_val_seq
        else:
            return target_val_last

    def forward_recursion(self, in_energy, mask=None):
        return self.recursion(in_energy, mask)

    def backward_recursion(self, in_energy, mask=None):
        return self.recursion(in_energy, mask, go_backwards=True)

    def get_marginal_prob(self, X, mask=None):
        in_energy = self.in_activation(K.dot(X, self.W) + self.b)
        if self.boundary_energy:
            in_energy = self.add_boundary_energy(in_energy, mask, self.a1, self.an)
        alpha = self.forward_recursion(in_energy, mask)
        beta = self.backward_recursion(in_energy, mask)
        if mask is not None:
            in_energy = in_energy * K.expand_dims(K.cast(mask, K.floatx()))
        margin = -(self.shift_right(alpha) + in_energy + self.shift_left(beta))
        return self.softmaxNd(margin)

    def viterbi_decoding(self, X, mask=None):
        in_energy = self.in_activation(K.dot(X, self.W) + self.b)
        if self.boundary_energy:
            in_energy = self.add_boundary_energy(in_energy, mask, self.a1, self.an)

        argmin_tables = self.recursion(in_energy, mask, return_logZ=False)
        argmin_tables = K.cast(argmin_tables, 'int32')

        # backward to find best path, `initial_best_idx` can be any, as all elements in the last argmin_table are the same
        argmin_tables = K.reverse(argmin_tables, 1)
        initial_best_idx = [K.expand_dims(argmin_tables[:, 0, 0])] # matrix instead of vector is required by tf `K.rnn`

        def gather_each_row(params, indices):
            n = K.shape(indices)[0]
            if K._BACKEND == 'theano':
                return params[T.arange(n), indices]
            else:
                indices = K.transpose(tf.pack([tf.range(n), indices]))
                return tf.gather_nd(params, indices)

        def find_path(argmin_table, best_idx):
            next_best_idx = gather_each_row(argmin_table, best_idx[0][:, 0])
            next_best_idx = K.expand_dims(next_best_idx)
            return next_best_idx, [next_best_idx]
        _, best_paths, _ = K.rnn(find_path, argmin_tables, initial_best_idx, input_length=self.input_length, unroll=self.unroll)
        best_paths = K.reverse(best_paths, 1)

        if K.ndim(best_paths) == 3:
            # due to inconsistent of theano (drop after scan) and tensorflow on broadcast dim
            best_paths = K.squeeze(best_paths, 2)
        return K.one_hot(best_paths, self.output_dim)
