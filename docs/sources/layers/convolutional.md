
## Convolution1D

```python
keras.layers.convolutional.Convolution1D(nb_filter, stack_size, filter_length, 
        init='uniform', activation='linear', weights=None, 
        border_mode='valid', subsample_length=1, 
        W_regularizer=None, b_regularizer=None, W_constraint=None, 
        b_constraint=None)
```

Convolution operator for filtering neighborhoods of one-dimensional inputs.

- __Arguments__:

    - __nb_filter__: Number of convolution kernels to use.
    - __stack_size__: Number of channels in the input.
    - __filter_length__: The extension (spatial or temporal) of each filter.
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights.
    - __border_mode__: 'valid' or 'full'. see scipy.signal.convolve2d.
    - __subsample_length__: factor by which to subsample output.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.

---

## Convolution2D

```python
keras.layers.convolutional.Convolution2D(nb_filter, stack_size, nb_row, nb_col, 
        init='glorot_uniform', activation='linear', weights=None, 
        border_mode='valid', subsample=(1, 1),
        W_regularizer=None, b_regularizer=None, W_constraint=None)
```

Convolution operator for filtering windows of two-dimensional inputs. 

- __Arguments__:

    - __nb_filter__: Number of convolution kernels to use.
    - __stack_size__: Number of channels in the input.
    - __nb_row__: Number of rows in the convolution kernels
    - __nb_col__: Number of columns in the convolution kernels
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights.
    - __border_mode__: 'valid' or 'full'. see scipy.signal.convolve2d.
    - __subsample__: tuple of length 2. Factor by which to subsample output. Also called strides elsewhere.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.


---

## MaxPooling1D

```python
keras.layers.convolutional.MaxPooling1D(pool_length=2, ignore_border=True)
```

- __Arguments__:

    - __pool_length__: factor by which to downscale. 2 will halve the input.
    - __ignore_border__: boolean. When True, (5,) input with pool_length=2 will generate a (2,) output, (3,) otherwise.

---

## MaxPooling2D

```python
keras.layers.convolutional.MaxPooling2D(poolsize=(2, 2), ignore_border=True)
```

- __Arguments__:

    - __pool_size__: factor by which to downscale (vertical ds, horizontal ds). (2, 2) will halve the image in each dimension.
    - __ignore_border__: boolean. When True, (5, 5) input with pool_size=(2, 2) will generate a (2, 2) output, (3, 3) otherwise.

