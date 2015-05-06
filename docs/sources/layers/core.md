## Base class

```python
keras.layers.core.Layer()
```

__Methods__:

```python
connect(previous_layer)
```

Connect the input of the current layer to the output of the argument layer.

- __Return__: None.

- __Arguments__: 
    - __previous_layer__: Layer object.



```python
output(train)
```

Get the output of the layer.

- __Return__: Theano tensor.

- __Arguments__: 
    - __train__: Boolean. Specifies whether output is computed in training mode or in testing mode, which can change the logic, for instance in there are any `Dropout` layers in the network. 



```python
get_input(train)
```

Get the input of the layer.

- __Return__: Theano tensor.

- __Arguments__: 
    - __train__: Boolean. Specifies whether output is computed in training mode or in testing mode, which can change the logic, for instance in there are any `Dropout` layers in the network. 



```python
get_weights()
```

Get the weights of the parameters of the layer.

- __Return__: List of numpy arrays (one per layer parameter). 



```python
set_weights(weights)
```

Set the weights of the parameters of the layer.

- __Arguments__: 
    - __weights__: List of numpy arrays (one per layer parameter). Should be in the same order as what `get_weights(self)` returns.



---

## Dense
```python
keras.layers.core.Dense(input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None \
W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None)
```

Standard 1D fully-connect layer. 

- __Input shape__: 2D tensor with shape: `(nb_samples, input_dim)`.

- __Output shape__: 2D tensor with shape: `(nb_samples, output_dim)`.

- __Arguments__:

    - __input_dim__: int >= 0. 
    - __output_dim__: int >= 0. 
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.
    - __W_regularizer__: instance of the [regularizers](../regularizers.md) module (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of the [regularizers](../regularizers.md) module, applied to the bias.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.

---

## TimeDistributedDense
```python
keras.layers.core.TimeDistributedDense(input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None \
W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None)
```

Fully-connected layer distributed over the time dimension. Useful after a recurrent network set to `return_sequences=True`.

- __Input shape__: 3D tensor with shape: `(nb_samples, nb_timesteps, input_dim)`.

- __Arguments__:
    - __input_dim__: int >= 0. 
    - __output_dim__: int >= 0. 
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.
    - __W_regularizer__: instance of the [regularizers](../regularizers.md) module (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of the [regularizers](../regularizers.md) module, applied to the bias.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.

- __Example__:
```python
# input shape: (nb_samples, nb_timesteps, 10)
model.add(LSTM(10, 5, return_sequences=True)) # output shape: (nb_samples, nb_timesteps, 5)
model.add(TimeDistributedDense(5, 10)) # output shape: (nb_samples, nb_timesteps, 10)
```


---

## Activation
```python
keras.layers.core.Activation(activation)
```
Apply an activation function to the input. 

- __Input shape__: This layer does not assume a specific input shape. As a result, it cannot be used as the first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:

    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function.


---

## Dropout
```python
keras.layers.core.Dropout(p)
```
Apply dropout to the input. Dropout consists in randomly setting a fraction `p` of input units to 0 at each update during training time, which helps prevent overfitting. Reference: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

- __Input shape__: This layer does not assume a specific input shape. 

- __Output shape__: Same as input.

- __Arguments__:

    - __p__: float (0 <= p < 1). Fraction of the input that gets dropped out at training time. 


---

## Reshape
```python
keras.layers.core.Reshape(*dims)
```

Reshape the input to a new shape containing the same number of units. 

- __Input shape__: This layer does not assume a specific input shape. 

- __Output shape__: `(nb_samples, *dims)`.

- __Arguments__:

    - *dims: integers. Dimensions of the new shape.

- __Example__:
```python
# input shape: (nb_samples, 10)
model.add(Dense(10, 100)) # output shape: (nb_samples, 100)
model.add(Reshape(10, 10))  # output shape: (nb_samples, 10, 10)
```

---

## Flatten
```python
keras.layers.core.Flatten()
```

Convert a nD input to 1D. 

- __Input shape__: (nb_samples, *). This layer cannot be used as the first layer in a model.

- __Output shape__: `(nb_samples, nb_input_units)`.

---

## RepeatVector
```python
keras.layers.core.RepeatVector(n)
```

Repeat the 1D input n times. Dimensions of input are assumed to be (nb_samples, dim). Output will have the shape (nb_samples, n, dim).

- __Input shape__: This layer does not assume a specific input shape. This layer cannot be used as the first layer in a model.

- __Output shape__: `(nb_samples, n, input_dims)`.

- __Arguments__:
    - __n__: int. 

- __Example__:

```python
# input shape: (nb_samples, 10)
model.add(Dense(10, 100)) # output shape: (nb_samples, 100)
model.add(RepeatVector(2))  # output shape: (nb_samples, 2, 10)
```

