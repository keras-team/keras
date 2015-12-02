## Base class

```python
keras.layers.core.Layer()
```

__Methods__:

```python
set_previous(previous_layer)
```

Connect the input of the current layer to the output of the argument layer.

- __Return__: None.

- __Arguments__: 
    - __previous_layer__: Layer object.



```python
get_output(train)
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


```python
get_config()
```

- __Return__: Configuration dictionary describing the layer.


---

## Dense
```python
keras.layers.core.Dense(output_dim,
                        init='glorot_uniform',
                        activation='linear',
                        weights=None,
                        W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                        W_constraint=None, b_constraint=None,
                        input_dim=None)
```

Standard 1D fully-connect layer. 

- __Input shape__: 2D tensor with shape: `(nb_samples, input_dim)`.

- __Output shape__: 2D tensor with shape: `(nb_samples, output_dim)`.

- __Arguments__:

    - __output_dim__: int >= 0. 
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.
    - __input_dim__: dimensionality of the input (integer). This argument (or alternatively, the keyword argument `input_shape`) is required when using this layer as the first layer in a model. 

---

## TimeDistributedDense
```python
keras.layers.core.TimeDistributedDense(output_dim,
                                       init='glorot_uniform',
                                       activation='linear',
                                       weights=None
                                       W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                                       W_constraint=None, b_constraint=None,
                                       input_dim=None, input_length=None)
```

Fully-connected layer distributed over the time dimension. Useful after a recurrent network set to `return_sequences=True`.

- __Input shape__: 3D tensor with shape: `(nb_samples, timesteps, input_dim)`.

- __Arguments__:
    - __output_dim__: int >= 0. 
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.
    - __input_dim__: dimensionality of the input (integer). This argument (or alternatively, the keyword argument `input_shape`) is required when using this layer as the first layer in a model.
    - __input_length__: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten` then `Dense` layers upstream (without it, the shape of the dense outputs cannot be computed).

- __Example__:
```python
# input shape: (nb_samples, timesteps, 10)
model.add(LSTM(5, return_sequences=True, input_dim=10)) # output shape: (nb_samples, timesteps, 5)
model.add(TimeDistributedDense(15)) # output shape: (nb_samples, timesteps, 15)
```


---

## AutoEncoder
```python
keras.layers.core.AutoEncoder(encoder, decoder, output_reconstruction=True, weights=None):
```

A customizable autoencoder model. If `output_reconstruction = True` then dim(input) = dim(output) else dim(output) = dim(hidden)


- __Input shape__: The layer shape is defined by the encoder definitions

- __Output shape__: The layer shape is defined by the decoder definitions

- __Arguments__:

    - __encoder__: A [layer](./) or [layer container](./containers.md).

    - __decoder__: A [layer](./) or [layer container](./containers.md).
    
    - __output_reconstruction__: If this is False, then when .predict() is called, the output is the deepest hidden layer's activation. Otherwise, the output of the final decoder layer is presented. Be sure your validation data conforms to this logic if you decide to use any.
    
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.

- __Example__:
```python
from keras.layers import containers

# input shape: (nb_samples, 32)
encoder = containers.Sequential([Dense(16, input_dim=32), Dense(8)])
decoder = containers.Sequential([Dense(16, input_dim=8), Dense(32)])

autoencoder = Sequential()
autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
```


---

## Activation
```python
keras.layers.core.Activation(activation)
```
Apply an activation function to the input. 


- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model. To specify the number of samples per batch, you can use the keyword argument `batch_input_shape` (tuple of integers, including the samples axis).

- __Output shape__: Same as input.

- __Arguments__:

    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function.


---

## Dropout
```python
keras.layers.core.Dropout(p)
```
Apply dropout to the input. Dropout consists in randomly setting a fraction `p` of input units to 0 at each update during training time, which helps prevent overfitting. Reference: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)


- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model. To specify the number of samples per batch, you can use the keyword argument `batch_input_shape` (tuple of integers, including the samples axis).

- __Output shape__: Same as input.

- __Arguments__:

    - __p__: float (0 <= p < 1). Fraction of the input that gets dropped out at training time. 

---


## Reshape
```python
keras.layers.core.Reshape(dims)
```

Reshape the input to a new shape containing the same number of units. 


- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model. To specify the number of samples per batch, you can use the keyword argument `batch_input_shape` (tuple of integers, including the samples axis).

- __Output shape__: `(nb_samples, dims)`.

- __Arguments__:

    - dims: tuple of integers. Dimensions of the new shape.

- __Example__:
```python
# input shape: (nb_samples, 10)
model.add(Dense(100, input_dim=10)) # output shape: (nb_samples, 100)
model.add(Reshape(dims=(10, 10)))  # output shape: (nb_samples, 10, 10)
```

---

## Flatten
```python
keras.layers.core.Flatten()
```

Convert a nD input to 1D. 

- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model. To specify the number of samples per batch, you can use the keyword argument `batch_input_shape` (tuple of integers, including the samples axis).

- __Output shape__: `(nb_samples, nb_input_units)`.

---

## RepeatVector
```python
keras.layers.core.RepeatVector(n)
```

Repeat the 1D input n times. Dimensions of input are assumed to be `(nb_samples, dim)`. Output will have the shape `(nb_samples, n, dim)`.

Note that the output is still a single tensor; `RepeatVector` does not split the data flow.

- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model. To specify the number of samples per batch, you can use the keyword argument `batch_input_shape` (tuple of integers, including the samples axis).

- __Output shape__: `(nb_samples, n, input_dims)`.

- __Arguments__:
    - __n__: int. 

---

## Permute
```python
keras.layers.core.Permute(dims)
```
Permute the dimensions of the input data according to the given tuple. Sometimes useful for connecting RNNs and convnets together.

- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model. To specify the number of samples per batch, you can use the keyword argument `batch_input_shape` (tuple of integers, including the samples axis).

- __Output shape__: Same as the input shape, but with the dimensions re-ordered according to the ordering specified by the tuple.

- __Argument__: tuple specifying the permutation scheme (e.g. `(2, 1)` permutes the first and second dimension of the input).

- __Example__:
```python
# input shape: (nb_samples, 10)
model.add(Dense(50, input_dim=10)) # output shape: (nb_samples, 50)
model.add(Reshape(dims=(10, 5))) # output shape: (nb_samples, 10, 5)
model.add(Permute(dims=(2, 1))) #output shape: (nb_samples, 5, 10)
```

---

## ActivityRegularization
```python
keras.layers.core.ActivityRegularization(l1=0., l2=0.)
```

Leaves the input unchanged, but adds a term to the loss function based on the input activity. L1 and L2 regularization supported.

This layer can be used, for instance, to induce activation sparsity in the previous layer.

---

## MaxoutDense
```python
keras.layers.core.MaxoutDense(output_dim, nb_feature=4,
        init='glorot_uniform',
        weights=None,
        W_regularizer=None, b_regularizer=None, activity_regularizer=None,
        W_constraint=None, b_constraint=None,
        input_dim=None)
```

A dense maxout layer. A `MaxoutDense` layer takes the element-wise maximum of `nb_feature` `Dense(input_dim, output_dim)` linear layers. This allows the layer to learn a convex, piecewise linear activation function over the inputs. See [this paper](http://arxiv.org/pdf/1302.4389.pdf) for more details. Note that this is a *linear* layer -- if you wish to apply activation function (you shouldn't need to -- they are universal function approximators), an `Activation` layer must be added after.

- __Input shape__: 2D tensor with shape: `(nb_samples, input_dim)`.

- __Output shape__: 2D tensor with shape: `(nb_samples, output_dim)`.

- __Arguments__:

    - __output_dim__: int >= 0. 
    - __nb_feature__: int >= 0. the number of features to create for the maxout. This is equivalent to the number of piecewise elements to be allowed for the activation function. 
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.
    - __input_dim__: dimensionality of the input (integer). This argument (or alternatively, the keyword argument `input_shape`) is required when using this layer as the first layer in a model.

```python
# input shape: (nb_samples, 10)
model.add(Dense(100, input_dim=10)) # output shape: (nb_samples, 100)
model.add(MaxoutDense(50, nb_feature=10)) # output shape: (nb_samples, 50)
```

## Merge
```python
keras.layers.core.Merge(layers, mode='sum', concat_axis=-1, dot_axes=-1)
```

Merge the output of a list of layers (or containers) into a single tensor.

- __Arguments__:
    - __layers__: List of layers or [containers](/layers/containers/).
    - __mode__: String, one of `{'sum', 'mul', 'concat', 'ave', 'dot'}`. `sum`, `mul` and `ave` will simply sum/multiply/average the outputs of the layers (therefore all layers should have an output with the same shape). `concat` will concatenate the outputs along the dimension specified by `concate_axis` (therefore all layers should have an output that only differ along this dimension). `dot` will dot tensor contraction on the axes specified by `dot_axes` (see [the Numpy documentation](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.tensordot.html) for more details).
    - __concat_axis__: axis to use in `concat` mode.
    - __dot_axes__: axis or axes to use in `dot` mode (see [the Numpy documentation](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.tensordot.html) for more details).


- __Notes__:
    - `dot` mode only works with Theano for the time being.

- __Example__:

```python
left = Sequential()
left.add(Dense(50, input_shape=(784,)))
left.add(Activation('relu'))

right = Sequential()
right.add(Dense(50, input_shape=(784,)))
right.add(Activation('relu'))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit([X_train, X_train], Y_train, batch_size=128, nb_epoch=20, validation_data=([X_test, X_test], Y_test))
```

## Masking
```python
keras.layers.core.Masking(mask_value=0.)
```

Create a mask for the input data by using `mask_value` as the sentinel value which should be masked out.
Given an input of dimensions `(nb_samples, timesteps, input_dim)`, return the input untouched as output, and supply a mask of shape `(nb_samples, timesteps)` where all timesteps which had *all* their values equal to `mask_value` are masked out.

- __Input shape__: 3D tensor with shape: `(nb_samples, timesteps, features)`.

- __Output shape__: 3D tensor with shape: `(nb_samples, timesteps, features)`.

- __Notes__: Masking only works in Theano for the time being.

## Lambda
```python
keras.layers.core.Lambda(function, output_shape=None)
```

Used for evaluating an arbitrary Theano expression on the output of the previous layer.

- __Input shape__: Arbitrary. Use the keyword argument input_shape (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

- __Output shape__: Specified by the `output_shape` argument.

- __Arguments__:

    - __function__: The expression to be evaluated. Takes one argument: the output of the previous layer.
    - __output_shape__: Shape of the tensor returned by `function`. Should be a shape tuple (not including the samples dimension) or a function of the full input shape tuple (including samples dimension).

- __Example__:

```python
# custom softmax function
def sharp_softmax(X, beta=1.5):
    return theano.tensor.nnet.softmax(X * beta)

def output_shape(input_shape):
    # here input_shape includes the samples dimension
    return input_shape  # shape is unchanged

model = Sequential()
model.add(Dense(input_dim=10, output_dim=10))
model.add(Lambda(sharp_softmax, output_shape))
model.add(Dense(1))
model.add(Activation('sigmoid'))
```


## LambdaMerge
```python
keras.layers.core.LambdaMerge(layers, function, output_shape=None)
```

Merge the output of a list of layers (or containers) into a single tensor, using an arbitrary Theano expression.

- __Arguments__:
    - __layers__: List of layers or [containers](/layers/containers/).
    - __function__: The expression to be evaluated. Takes one argument: the list of input tensors.
    - __output_shape__: Shape of the tensor returned by `function`. Should be a shape tuple (not including samples dimension) or a function of the list of input shape tuples (including samples dimension).

- __Example__:

```python
# root mean square function
def rms(inputs):
    # inputs is a list of tensors
    s = inputs[0] ** 2
    for i in range(1, len(inputs)):
        s += inputs[i] ** 2
    s /= len(inputs)
    s = theano.tensor.sqrt(s)
    # return a single tensor
    return s

def output_shape(input_shapes):
    # return the shape of the first tensor
    return input_shapes[0]

left = Sequential()
left.add(Dense(input_dim=10, output_dim=10))
left.add(Activation('sigmoid'))

right = Sequential()
right.add(Dense(input_dim=10, output_dim=10))
right.add(Activation('sigmoid'))

model = Sequential()
model.add(LambdaMerge([left, right], rms, output_shape))

model.add(Dense(1))
model.add(Activation('sigmoid'))
```

---
