from docs import autogen
import pytest

test_doc1 = {
    'doc': """Base class for recurrent layers.

    # Arguments
        cell: A RNN cell instance. A RNN cell is a class that has:
            - a `call(input_at_t, states_at_t)` method, returning
                `(output_at_t, states_at_t_plus_1)`. The call method of the
                cell can also take the optional argument `constants`, see
                section "Note on passing external constants" below.
            - a `state_size` attribute. This can be a single integer
                (single state) in which case it is
                the size of the recurrent state
                (which should be the same as the size of the cell output).
                This can also be a list/tuple of integers
                (one size per state). In this case, the first entry
                (`state_size[0]`) should be the same as
                the size of the cell output.
            It is also possible for `cell` to be a list of RNN cell instances,
            in which cases the cells get stacked on after the other in the RNN,
            implementing an efficient stacked RNN.
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
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively,
            the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)

    # Input shape
        3D tensor with shape `(batch_size, timesteps, input_dim)`.

    # Output shape
        - if `return_state`: a list of tensors. The first tensor is
            the output. The remaining tensors are the last states,
            each with shape `(batch_size, units)`.
        - if `return_sequences`: 3D tensor with shape
            `(batch_size, timesteps, units)`.
        - else, 2D tensor with shape `(batch_size, units)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch. This assumes a one-to-one mapping
        between samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                if sequential model:
                  `batch_input_shape=(...)` to the first layer in your model.
                else for functional model with 1 or more Input layers:
                  `batch_shape=(...)` to all the first layers in your model.
                This is the expected shape of your inputs
                *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.
            - specify `shuffle=False` when calling fit().

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.

    # Note on specifying the initial state of RNNs
    Note: that
        One: You can specify the initial state of RNN layers symbolically by
            calling them with the keyword argument `initial_state`.
        Two: The value of `initial_state` should be a tensor or list of
            tensors representing
            the initial state of the RNN layer.
        You can specify the initial state of RNN layers numerically by:
        One: calling `reset_states`
            - With the keyword argument `states`.
                - The value of
            `states` should be a numpy array or
            list of numpy arrays representing
        the initial state of the RNN layer.

    # Note on passing external constants to RNNs
        You can pass "external" constants to the cell using the `constants`
        keyword: argument of `RNN.__call__` (as well as `RNN.call`) method.
        This: requires that the `cell.call` method accepts the same keyword argument
        `constants`. Such constants can be used to condition the cell
        transformation on additional static inputs (not changing over time),
        a.k.a. an attention mechanism.

    # Examples

    ```python
        # First, let's define a RNN Cell, as a layer subclass.

        class MinimalRNNCell(keras.layers.Layer):

            def __init__(self, units, **kwargs):
                self.units = units
                self.state_size = units
                super(MinimalRNNCell, self).__init__(**kwargs)

            def build(self, input_shape):
                self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                              initializer='uniform',
                                              name='kernel')
                self.recurrent_kernel = self.add_weight(
                    shape=(self.units, self.units),
                    initializer='uniform',
                    name='recurrent_kernel')
                self.built = True

            def call(self, inputs, states):
                prev_output = states[0]
                h = K.dot(inputs, self.kernel)
                output = h + K.dot(prev_output, self.recurrent_kernel)
                return output, [output]

        # Let's use this cell in a RNN layer:

        cell = MinimalRNNCell(32)
        x = keras.Input((None, 5))
        layer = RNN(cell)
        y = layer(x)

        # Here's how to use the cell to build a stacked RNN:

        cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
        x = keras.Input((None, 5))
        layer = RNN(cells)
        y = layer(x)
    ```
    """,
    'result': '''Base class for recurrent layers.

__Arguments__

- __cell__: A RNN cell instance. A RNN cell is a class that has:
    - a `call(input_at_t, states_at_t)` method, returning
        `(output_at_t, states_at_t_plus_1)`. The call method of the
        cell can also take the optional argument `constants`, see
        section "Note on passing external constants" below.
    - a `state_size` attribute. This can be a single integer
        (single state) in which case it is
        the size of the recurrent state
        (which should be the same as the size of the cell output).
        This can also be a list/tuple of integers
        (one size per state). In this case, the first entry
        (`state_size[0]`) should be the same as
        the size of the cell output.

    It is also possible for `cell` to be a list of RNN cell instances,
    in which cases the cells get stacked on after the other in the RNN,
    implementing an efficient stacked RNN.

- __return_sequences__: Boolean. Whether to return the last output
    in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
    in addition to the output.
- __go_backwards__: Boolean (default False).
    If True, process the input sequence backwards and return the
    reversed sequence.
- __stateful__: Boolean (default False). If True, the last state
    for each sample at index i in a batch will be used as initial
    state for the sample of index i in the following batch.
- __unroll__: Boolean (default False).
    If True, the network will be unrolled,
    else a symbolic loop will be used.
    Unrolling can speed-up a RNN,
    although it tends to be more memory-intensive.
    Unrolling is only suitable for short sequences.
- __input_dim__: dimensionality of the input (integer).
    This argument (or alternatively,
    the keyword argument `input_shape`)
    is required when using this layer as the first layer in a model.
- __input_length__: Length of input sequences, to be specified
    when it is constant.
    This argument is required if you are going to connect
    `Flatten` then `Dense` layers upstream
    (without it, the shape of the dense outputs cannot be computed).
    Note that if the recurrent layer is not the first layer
    in your model, you would need to specify the input length
    at the level of the first layer
    (e.g. via the `input_shape` argument)

__Input shape__

3D tensor with shape `(batch_size, timesteps, input_dim)`.

__Output shape__

- if `return_state`: a list of tensors. The first tensor is
    the output. The remaining tensors are the last states,
    each with shape `(batch_size, units)`.
- if `return_sequences`: 3D tensor with shape
    `(batch_size, timesteps, units)`.
- else, 2D tensor with shape `(batch_size, units)`.

__Masking__

This layer supports masking for input data with a variable number
of timesteps. To introduce masks to your data,
use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
set to `True`.

__Note on using statefulness in RNNs__

You can set RNN layers to be 'stateful', which means that the states
computed for the samples in one batch will be reused as initial states
for the samples in the next batch. This assumes a one-to-one mapping
between samples in different successive batches.

To enable statefulness:
- specify `stateful=True` in the layer constructor.
- specify a fixed batch size for your model, by passing
if sequential model:
`batch_input_shape=(...)` to the first layer in your model.
else for functional model with 1 or more Input layers:
`batch_shape=(...)` to all the first layers in your model.
This is the expected shape of your inputs
*including the batch size*.
It should be a tuple of integers, e.g. `(32, 10, 100)`.
- specify `shuffle=False` when calling fit().

To reset the states of your model, call `.reset_states()` on either
a specific layer, or on your entire model.

__Note on specifying the initial state of RNNs__

Note: that
- __One__: You can specify the initial state of RNN layers symbolically by
    calling them with the keyword argument `initial_state`.
- __Two__: The value of `initial_state` should be a tensor or list of
    tensors representing
    the initial state of the RNN layer.

You can specify the initial state of RNN layers numerically by:

- __One__: calling `reset_states`
    - With the keyword argument `states`.
        - The value of

    `states` should be a numpy array or
    list of numpy arrays representing

the initial state of the RNN layer.

__Note on passing external constants to RNNs__

You can pass "external" constants to the cell using the `constants`
- __keyword__: argument of `RNN.__call__` (as well as `RNN.call`) method.
- __This__: requires that the `cell.call` method accepts the same keyword argument

`constants`. Such constants can be used to condition the cell
transformation on additional static inputs (not changing over time),
a.k.a. an attention mechanism.

__Examples__


```python
# First, let's define a RNN Cell, as a layer subclass.

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# Let's use this cell in a RNN layer:

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```
'''}


def test_doc_lists():
    docstring = autogen.process_docstring(test_doc1['doc'])
    assert docstring == test_doc1['result']


if __name__ == '__main__':
    pytest.main([__file__])
