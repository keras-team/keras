import keras
from .. import backend as K
from .. import activations
from .. import initializers
from .. import regularizers
from .. import constraints
from ..engine import Layer
from ..engine import InputSpec
from ..objectives import categorical_crossentropy, sparse_categorical_crossentropy


class CRF(Layer):
    """An implementation of linear chain conditional random field (CRF).

    An linear chain CRF is defined to maximize the following likelihood function:

    $$ L(W, U, b; y_1, ..., y_n) := \frac{1}{Z} \sum_{y_1, ..., y_n} \exp(-a_1' y_1 - a_n' y_n
        - \sum_{k=1^n}((f(x_k' W + b) y_k) + y_1' U y_2)), $$

    where:
        $Z$: normalization constant
        $x_k, y_k$:  inputs and outputs

    This implementation has two modes for optimization:
    1. (`join mode`) optimized by maximizing join likelihood, which is optimal in theory of statistics.
       Note that in this case, CRF mast be the output/last layer.
    2. (`marginal mode`) return marginal probabilities on each time step and optimized via composition
       likelihood (product of marginal likelihood), i.e., using `categorical_crossentropy` loss.
       Note that in this case, CRF can be either the last layer or an intermediate layer (though not explored).

    For prediction (test phrase), one can choose either Viterbi best path (class indices) or marginal
    probabilities if probabilities are needed. However, if one chooses *join mode* for training,
    Viterbi output is typically better than marginal output, but the marginal output will still perform
    reasonably close, while if *marginal mode* is used for training, marginal output usually performs
    much better. The default behavior is set according to this observation.

    In addition, this implementation supports masking and accepts either onehot or sparse target.


    # Examples

    ```python
        model = Sequential()
        model.add(Embedding(3001, 300, mask_zero=True)(X)

        # use learn_mode = 'join', test_mode = 'viterbi', sparse_target = True (label indice output)
        crf = CRF(10, sparse_target=True)
        model.add(crf)

        # crf.accuracy is default to Viterbi acc if using join-mode (default).
        # One can add crf.marginal_acc if interested, but may slow down learning
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

        # y must be label indices (with shape 1 at dim 3) here, since `sparse_target=True`
        model.fit(x, y)

        # prediction give onehot representation of Viterbi best path
        y_hat = model.predict(x_test)
    ```


    # Arguments
        units: Positive integer, dimensionality of the output space.
        learn_mode: Either 'join' or 'marginal'.
            The former train the model by maximizing join likelihood while the latter
            maximize the product of marginal likelihood over all time steps.
        test_mode: Either 'viterbi' or 'marginal'.
            The former is recommended and as default when `learn_mode = 'join'` and
            gives one-hot representation of the best path at test (prediction) time,
            while the latter is recommended and chosen as default when `learn_mode = 'marginal'`,
            which produces marginal probabilities for each time step.
        sparse_target: Boolen (default False) indicating if provided labels are one-hot or
            indices (with shape 1 at dim 3).
        use_boundary: Boolen (default True) inidicating if trainable start-end chain energies
            should be added to model.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        chain_initializer: Initializer for the `chain_kernel` weights matrix,
            used for the CRF chain energy.
            (see [initializers](../initializers.md)).
        boundary_initializer: Initializer for the `left_boundary`, 'right_boundary' weights vectors,
            used for the start/left and end/right boundary energy.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        chain_regularizer: Regularizer function applied to
            the `chain_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        boundary_regularizer: Regularizer function applied to
            the 'left_boundary', 'right_boundary' weight vectors
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        chain_constraint: Constraint function applied to
            the `chain_kernel` weights matrix
            (see [constraints](../constraints.md)).
        boundary_constraint: Constraint function applied to
            the `left_boundary`, `right_boundary` weights vectors
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used.
            Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        3D tensor with shape `(nb_samples, timesteps, units)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    """

    def __init__(self, units,
                 learn_mode='join',
                 test_mode=None,
                 sparse_target=False,
                 use_boundary=True,
                 use_bias=True,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 chain_initializer='orthogonal',
                 bias_initializer='zeros',
                 boundary_initializer='zeros',
                 kernel_regularizer=None,
                 chain_regularizer=None,
                 boundary_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 chain_constraint=None,
                 boundary_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 unroll=False,
                 **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.learn_mode = learn_mode
        assert self.learn_mode in ['join', 'marginal']
        self.test_mode = test_mode
        if self.test_mode is None:
            self.test_mode = 'viterbi' if self.learn_mode == 'join' else 'marginal'
        else:
            assert self.test_mode in ['viterbi', 'marginal']
        self.sparse_target = sparse_target
        self.use_boundary = use_boundary
        self.use_bias = use_bias

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.chain_initializer = initializers.get(chain_initializer)
        self.boundary_initializer = initializers.get(boundary_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.chain_regularizer = regularizers.get(chain_regularizer)
        self.boundary_regularizer = regularizers.get(boundary_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.chain_constraint = constraints.get(chain_constraint)
        self.boundary_constraint = constraints.get(boundary_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.unroll = unroll

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight((self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.chain_kernel = self.add_weight((self.units, self.units),
                                            name='chain_kernel',
                                            initializer=self.chain_initializer,
                                            regularizer=self.chain_regularizer,
                                            constraint=self.chain_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_boundary:
            self.left_boundary = self.add_weight((self.units,),
                                                 name='left_boundary',
                                                 initializer=self.boundary_initializer,
                                                 regularizer=self.boundary_regularizer,
                                                 constraint=self.boundary_constraint)
            self.right_boundary = self.add_weight((self.units,),
                                                  name='right_boundary',
                                                  initializer=self.boundary_initializer,
                                                  regularizer=self.boundary_regularizer,
                                                  constraint=self.boundary_constraint)
        self.built = True

    def call(self, X, mask=None):
        input_length = K.int_shape(X)[1]
        if mask is not None:
            assert K.ndim(mask) == 2, 'Input mask to CRF must have dim 2 if not None'

        if self.test_mode == 'viterbi':
            test_output = self.viterbi_decoding(X, mask)
        else:
            test_output = self.get_marginal_prob(X, mask)

        self.uses_learning_phase = True
        if self.learn_mode == 'join':
            train_output = K.zeros_like(K.dot(X, self.kernel))
            out = K.in_train_phase(train_output, test_output)
        else:
            if self.test_mode == 'viterbi':
                train_output = self.get_marginal_prob(X, mask)
                out = K.in_train_phase(train_output, test_output)
            else:
                out = test_output
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.units,)

    def compute_mask(self, input, mask=None):
        if mask is not None and self.learn_mode == 'join':
            return K.any(mask, axis=1)
        return mask

    def get_config(self):
        config = {'units': self.units,
                  'learn_mode': self.learn_mode,
                  'test_mode': self.test_mode,
                  'use_boundary': self.use_boundary,
                  'use_bias': self.use_bias,
                  'sparse_target': self.sparse_target,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'chain_initializer': initializers.serialize(self.chain_initializer),
                  'boundary_initializer': initializers.serialize(self.boundary_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'activation': activations.serialize(self.activation),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'chain_regularizer': regularizers.serialize(self.chain_regularizer),
                  'boundary_regularizer': regularizers.serialize(self.boundary_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'chain_constraint': constraints.serialize(self.chain_constraint),
                  'boundary_constraint': constraints.serialize(self.boundary_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
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
                    y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), self.units)
                X = self.inbound_nodes[0].input_tensors[0]
                mask = self.inbound_nodes[0].input_masks[0]
                nloglik = self.get_negative_log_likelihood(y_true, X, mask)
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
        """log(sum(exp(x)) = m + log(sum(exp(x-m))), where m = max(x, axis)."""
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
        start = K.expand_dims(K.expand_dims(start, 0), 0)
        end = K.expand_dims(K.expand_dims(end, 0), 0)
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

    def get_log_normalization_constant(self, input_energy, mask, **kwargs):
        """Compute logarithm of the normalization constance Z, where
        Z = sum exp(-E) -> logZ = log sum exp(-E) =: -nlogZ
        """
        # should have logZ[:, i] == logZ[:, j] for any i, j
        logZ = self.recursion(input_energy, mask, return_sequences=False, **kwargs)
        return logZ[:, 0]

    def get_energy(self, y_true, input_energy, mask):
        """Energy = a1' y1 + u1' y1 + y1' U y2 + u2' y2 + y2' U y3 + u3' y3 + an' y3
        """
        input_energy = K.sum(input_energy * y_true, 2)  # (B, T)
        chain_energy = K.sum(K.dot(y_true[:, :-1, :], self.chain_kernel) * y_true[:, 1:, :], 2)  # (B, T-1)

        if mask is not None:
            mask = K.cast(mask, K.floatx())
            chain_mask = mask[:, :-1] * mask[:, 1:]  # (B, T-1), mask[:,:-1]*mask[:,1:] makes it work with any padding
            input_energy = input_energy * mask
            chain_energy = chain_energy * chain_mask
        total_energy = K.sum(input_energy, -1) + K.sum(chain_energy, -1)  # (B, )

        return total_energy

    def get_negative_log_likelihood(self, y_true, X, mask):
        """Compute the loss, i.e., negative log likelihood (normalize by number of time steps)
           likelihood = 1/Z * exp(-E) ->  neg_log_like = - log(1/Z * exp(-E)) = logZ + E
        """
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)
        energy = self.get_energy(y_true, input_energy, mask)
        logZ = self.get_log_normalization_constant(input_energy, mask, input_length=K.int_shape(X)[1])
        nloglik = logZ + energy
        if mask is not None:
            nloglik = nloglik / K.sum(K.cast(mask, K.floatx()), 1)
        else:
            nloglik = nloglik / K.cast(K.shape(X)[1], K.floatx())
        return nloglik

    def step(self, input_energy_t, states, return_logZ=True):
        # not in the following  `prev_target_val` has shape = (B, F)
        # where B = batch_size, F = output feature dim
        # Note: `i` is of float32, due to the behavior of `K.rnn`
        prev_target_val, i, chain_energy = states[:3]
        t = K.cast(i[0, 0], dtype='int32')
        if len(states) > 3:
            if K.backend() == 'theano':
                m = states[3][:, t:(t + 2)]
            else:
                m = K.tf.slice(states[3], [0, t], [-1, 2])
            input_energy_t = input_energy_t * K.expand_dims(m[:, 0])
            chain_energy = chain_energy * K.expand_dims(K.expand_dims(m[:, 0] * m[:, 1]))  # (1, F, F)*(B, 1, 1) -> (B, F, F)
        if return_logZ:
            energy = chain_energy + K.expand_dims(input_energy_t - prev_target_val, 2)  # shapes: (1, B, F) + (B, F, 1) -> (B, F, F)
            new_target_val = self.log_sum_exp(-energy, 1)  # shapes: (B, F)
            return new_target_val, [new_target_val, i + 1]
        else:
            energy = chain_energy + K.expand_dims(input_energy_t + prev_target_val, 2)
            min_energy = K.min(energy, 1)
            argmin_table = K.cast(K.argmin(energy, 1), K.floatx())  # cast for tf-version `K.rnn`
            return argmin_table, [min_energy, i + 1]

    def recursion(self, input_energy, mask=None, go_backwards=False, return_sequences=True, return_logZ=True, input_length=None):
        """Forward (alpha) or backward (beta) recursion

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
        """
        chain_energy = self.chain_kernel
        chain_energy = K.expand_dims(chain_energy, 0)  # shape=(1, F, F): F=num of output features. 1st F is for t-1, 2nd F for t
        prev_target_val = K.zeros_like(input_energy[:, 0, :])  # shape=(B, F), dtype=float32

        if go_backwards:
            input_energy = K.reverse(input_energy, 1)
            if mask is not None:
                mask = K.reverse(mask, 1)

        initial_states = [prev_target_val, K.zeros_like(prev_target_val[:, :1])]
        constants = [chain_energy]

        if mask is not None:
            mask2 = K.cast(K.concatenate([mask, K.zeros_like(mask[:, :1])], axis=1), K.floatx())
            constants.append(mask2)

        def _step(input_energy_i, states):
            return self.step(input_energy_i, states, return_logZ)

        target_val_last, target_val_seq, _ = K.rnn(_step, input_energy, initial_states, constants=constants,
                                                   input_length=input_length, unroll=self.unroll)

        if return_sequences:
            if go_backwards:
                target_val_seq = K.reverse(target_val_seq, 1)
            return target_val_seq
        else:
            return target_val_last

    def forward_recursion(self, input_energy, **kwargs):
        return self.recursion(input_energy, **kwargs)

    def backward_recursion(self, input_energy, **kwargs):
        return self.recursion(input_energy, go_backwards=True, **kwargs)

    def get_marginal_prob(self, X, mask=None):
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)
        input_length = K.int_shape(X)[1]
        alpha = self.forward_recursion(input_energy, mask=mask, input_length=input_length)
        beta = self.backward_recursion(input_energy, mask=mask, input_length=input_length)
        if mask is not None:
            input_energy = input_energy * K.expand_dims(K.cast(mask, K.floatx()))
        margin = -(self.shift_right(alpha) + input_energy + self.shift_left(beta))
        return self.softmaxNd(margin)

    def viterbi_decoding(self, X, mask=None):
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(input_energy, mask, self.left_boundary, self.right_boundary)

        argmin_tables = self.recursion(input_energy, mask, return_logZ=False)
        argmin_tables = K.cast(argmin_tables, 'int32')

        # backward to find best path, `initial_best_idx` can be any, as all elements in the last argmin_table are the same
        argmin_tables = K.reverse(argmin_tables, 1)
        initial_best_idx = [K.expand_dims(argmin_tables[:, 0, 0])]  # matrix instead of vector is required by tf `K.rnn`
        if K.backend() == 'theano':
            initial_best_idx = [K.T.unbroadcast(initial_best_idx[0], 1)]

        def gather_each_row(params, indices):
            n = K.shape(indices)[0]
            if K.backend() == 'theano':
                return params[K.T.arange(n), indices]
            else:
                indices = K.transpose(K.stack([K.tf.range(n), indices]))
                return K.tf.gather_nd(params, indices)

        def find_path(argmin_table, best_idx):
            next_best_idx = gather_each_row(argmin_table, best_idx[0][:, 0])
            next_best_idx = K.expand_dims(next_best_idx)
            if K.backend() == 'theano':
                next_best_idx = K.T.unbroadcast(next_best_idx, 1)
            return next_best_idx, [next_best_idx]

        _, best_paths, _ = K.rnn(find_path, argmin_tables, initial_best_idx, input_length=K.int_shape(X)[1], unroll=self.unroll)
        best_paths = K.reverse(best_paths, 1)
        best_paths = K.squeeze(best_paths, 2)

        return K.one_hot(best_paths, self.units)
