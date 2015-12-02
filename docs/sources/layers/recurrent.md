
## SimpleRNN

```python
keras.layers.recurrent.SimpleRNN(output_dim,
        init='glorot_uniform', inner_init='orthogonal',
        activation='sigmoid',
        weights=None,
        return_sequences=False,
        go_backwards=False,
        stateful=False,
        input_dim=None, input_length=None)
```
Fully connected RNN where the output is to fed back to the input. 

- __Input shape__: 3D tensor with shape: `(nb_samples, timesteps, input_dim)`.

- __Output shape__: 
    - if `return_sequences`: 3D tensor with shape: `(nb_samples, timesteps, output_dim)`.
    - else: 2D tensor with shape: `(nb_samples, output_dim)`.

- __Masking__: This layer supports masking for input data with a variable number of timesteps To introduce masks to your data, use an [Embedding](embeddings.md) layer with the `mask_zero` parameter set to `True`. **Note:** for the time being, masking in only supported with Theano.

- __Notes__: When using the TensorFlow backend, the number of timesteps used must be fixed. Make sure to pass an `input_length` int argument or a complete `input_shape` tuple argument.


- __Arguments__:
    - __output_dim__: dimension of the internal projections and the final output.
    - __init__: weight initialization function. Can be the name of an existing function (str), or a Theano function (see: [initializations](../initializations.md)).
    - __activation__: activation function. Can be the name of an existing function (str), or a Theano function (see: [activations](../activations.md)).
    - __weights__: list of numpy arrays to set as initial weights. The list should have 3 elements, of shapes: `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
    - __return_sequences__: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    - __go_backwards__: Boolean (default False). If True, rocess the input sequence backwards.
    - __stateful__: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
    - __input_dim__: dimensionality of the input (integer). This argument (or alternatively, the keyword argument `input_shape`) is required when using this layer as the first layer in a model.
    - __input_length__: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten` then `Dense` layers upstream (without it, the shape of the dense outputs cannot be computed).


---

## GRU

```python
keras.layers.recurrent.GRU(output_dim,
        init='glorot_uniform', inner_init='orthogonal',
        activation='sigmoid', inner_activation='hard_sigmoid',
        return_sequences=False,
        go_backwards=False,
        stateful=False,
        input_dim=None, input_length=None)
```

Gated Recurrent Unit - Cho et al. 2014.

- __Input shape__: 3D tensor with shape: `(nb_samples, timesteps, input_dim)`.

- __Output shape__:
    - if `return_sequences`: 3D tensor with shape: `(nb_samples, timesteps, output_dim)`.
    - else: 2D tensor with shape: `(nb_samples, output_dim)`.

- __Masking__: This layer supports masking for input data with a variable number of timesteps To introduce masks to your data, use an [Embedding](embeddings.md) layer with the `mask_zero` parameter set to true. **Note:** for the time being, masking in only supported with Theano.

- __Notes__: When using the TensorFlow backend, the number of timesteps used must be fixed. Make sure to pass an `input_length` int argument or a complete `input_shape` tuple argument.

- __Arguments__:
    - __output_dim__: dimension of the internal projections and the final output.
    - __init__: weight initialization function for the output cell. Can be the name of an existing function (str), or a Theano function (see: [initializations](../initializations.md)).
    - __inner_init__: weight initialization function for the inner cells.
    - __activation__: activation function for the output. Can be the name of an existing function (str), or a Theano function (see: [activations](../activations.md)).
    - __inner_activation__: activation function for the inner cells.
    - __weights__: list of numpy arrays to set as initial weights. The list should have 9 elements.
    - __return_sequences__: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    - __go_backwards__: Boolean (default False). If True, rocess the input sequence backwards.
    - __stateful__: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
    - __input_dim__: dimensionality of the input (integer). This argument (or alternatively, the keyword argument `input_shape`) is required when using this layer as the first layer in a model.
    - __input_length__: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten` then `Dense` layers upstream (without it, the shape of the dense outputs cannot be computed).


- __References__: 
    - [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
    - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)

---

## LSTM

```python
keras.layers.recurrent.LSTM(output_dim,
        init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None,
        return_sequences=False,
        go_backwards=False,
        stateful=False,
        input_dim=None, input_length=None)
```

Long-Short Term Memory unit - Hochreiter 1997.

- __Input shape__: 3D tensor with shape: `(nb_samples, timesteps, input_dim)`.

- __Output shape__:
    - if `return_sequences`: 3D tensor with shape: `(nb_samples, timesteps, output_dim)`.
    - else: 2D tensor with shape: `(nb_samples, output_dim)`.

- __Masking__: This layer supports masking for input data with a variable number of timesteps To introduce masks to your data, use an [Embedding](embeddings.md) layer with the `mask_zero` parameter set to true. **Note:** for the time being, masking in only supported with Theano.

- __Notes__: When using the TensorFlow backend, the number of timesteps used must be fixed. Make sure to pass an `input_length` int argument or a complete `input_shape` tuple argument.

- __Arguments__:
    - __output_dim__: dimension of the internal projections and the final output.
    - __init__: weight initialization function for the output cell. Can be the name of an existing function (str), or a Theano function (see: [initializations](../initializations.md)).
    - __inner_init__: weight initialization function for the inner cells.
    - __forget_bias_init__: initialization function for the bias of the forget gate. [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) recommend initializing with ones.
    - __activation__: activation function for the output. Can be the name of an existing function (str), or a Theano function (see: [activations](../activations.md)).
    - __inner_activation__: activation function for the inner cells.
    - __weights__: list of numpy arrays to set as initial weights. The list should have 12 elements.
    - __return_sequences__: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    - __go_backwards__: Boolean (default False). If True, rocess the input sequence backwards.
    - __stateful__: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
    - __input_dim__: dimensionality of the input (integer). This argument (or alternatively, the keyword argument `input_shape`) is required when using this layer as the first layer in a model.
    - __input_length__: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten` then `Dense` layers upstream (without it, the shape of the dense outputs cannot be computed).


- __References__: 
    - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
    - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
    - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)

---
