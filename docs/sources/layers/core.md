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
W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None)
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
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.

---

## TimeDistributedDense
```python
keras.layers.core.TimeDistributedDense(input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None \
W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None)
```

Fully-connected layer distributed over the time dimension. Useful after a recurrent network set to `return_sequences=True`.

- __Input shape__: 3D tensor with shape: `(nb_samples, nb_timesteps, input_dim)`.

- __Arguments__:
    - __input_dim__: int >= 0. 
    - __output_dim__: int >= 0. 
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __activation__: name of activation function to use (see: [activations](../activations.md)), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.

- __Example__:
```python
# input shape: (nb_samples, nb_timesteps, 10)
model.add(LSTM(10, 5, return_sequences=True)) # output shape: (nb_samples, nb_timesteps, 5)
model.add(TimeDistributedDense(5, 10)) # output shape: (nb_samples, nb_timesteps, 10)
```


---

## AutoEncoder
```python
keras.layers.core.AutoEncoder(encoder, decoder, output_reconstruction=True, tie_weights=False, weights=None):
```

A customizable autoencoder model. If `output_reconstruction = True` then dim(input) = dim(output) else dim(output) = dim(hidden)


- __Input shape__: The layer shape is defined by the encoder definitions

- __Output shape__: The layer shape is defined by the decoder definitions

- __Arguments__:

    - __encoder__: A [layer](./) or [layer container](./containers.md).

    - __decoder__: A [layer](./) or [layer container](./containers.md).
    
    - __output_reconstruction__: If this is False the when .predict() is called the output is the deepest hidden layer's activation. Otherwise the output of the final decoder layer is presented. Be sure your validation data confirms to this logic if you decide to use any.
    
    - __tie_weights__: If True then the encoder bias is tied to the decoder bias. **Note**: This required the encoder layer corresponding to this decoder layer to be of the same time, eg: Dense:Dense
    
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.

- __Example__:
```python
from keras.layers import containers

# input shape: (nb_samples, 32)
encoder = containers.Sequential([Dense(32, 16), Dense(16, 8)])
decoder = containers.Sequential([Dense(8, 16), Dense(16, 32)])
autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False, tie_weights=True))
```

---

## DenoisingAutoEncoder
```python
keras.layers.core.AutoEncoder(encoder, decoder, output_reconstruction=True, tie_weights=False, weights=None, corruption_level=0.3):
```

A denoising autoencoder model that inherits the base features from autoencoder.
Since this layer uses similar logic to Dropout it cannot be the first layer in a pipeline.

- __Input shape__: The layer shape is defined by the encoder definitions

- __Output shape__: The layer shape is defined by the decoder definitions

- __Arguments__:

    - __encoder__: A [layer](./) or [layer container](./containers.md).

    - __decoder__: A [layer](./) or [layer container](./containers.md).
    
    - __output_reconstruction__: If this is False the when .predict() is called the output is the deepest hidden layer's activation. Otherwise the output of the final decoder layer is presented. Be sure your validation data confirms to this logic if you decide to use any.
    
    - __tie_weights__: If True then the encoder bias is tied to the decoder bias. **Note**: This required the encoder layer corresponding to this decoder layer to be of the same time, eg: Dense:Dense
    
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.
    
    - __corruption_level__: the amount of binomial noise added to the input layer of the model.

- __Example__:
```python
# input shape: (nb_samples, 32)
autoencoder.add(Dense(32, 32))
autoencoder.add(DenoisingAutoEncoder(encoder=Dense(32, 16),
                                     decoder=Dense(16, 32),
                                     output_reconstruction=False, tie_weights=True,
                                     corruption_level=0.3))
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

Repeat the 1D input n times. Dimensions of input are assumed to be `(nb_samples, dim)`. Output will have the shape `(nb_samples, n, dim)`.

Note that the output is still a single tensor; `RepeatVector` does not split the data flow.

- __Input shape__: This layer does not assume a specific input shape. This layer cannot be used as the first layer in a model.

- __Output shape__: `(nb_samples, n, input_dims)`.

- __Arguments__:
    - __n__: int. 

- __Example__:

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
keras.layers.core.MaxoutDense(input_dim, output_dim, nb_feature=4, init='glorot_uniform', weights=None, \
        W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None)
```

A dense maxout layer. A `MaxoutDense` layer takes the element-wise maximum of `nb_feature` `Dense(input_dim, output_dim)` linear layers. This allows the layer to learn a convex, piecewise linear activation function over the inputs. See [this paper](http://arxiv.org/pdf/1302.4389.pdf) for more details. Note that this is a *linear* layer -- if you wish to apply activation function (you shouldn't need to -- they are universal function approximators), an `Activation` layer must be added after.

- __Input shape__: 2D tensor with shape: `(nb_samples, input_dim)`.

- __Output shape__: 2D tensor with shape: `(nb_samples, output_dim)`.

- __Arguments__:

    - __input_dim__: int >= 0. 
    - __output_dim__: int >= 0. 
    - __nb_feature__: int >= 0. the number of features to create for the maxout. This is equivalent to the number of piecewise elements to be allowed for the activation function. 
    - __init__: name of initialization function for the weights of the layer (see: [initializations](../initializations.md)), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
    - __weights__: list of numpy arrays to set as initial weights. The list should have 1 element, of shape `(input_dim, output_dim)`.
    - __W_regularizer__: instance of [WeightRegularizer](../regularizers.md) (eg. L1 or L2 regularization), applied to the main weights matrix.
    - __b_regularizer__: instance of [WeightRegularizer](../regularizers.md), applied to the bias.
    - __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md), applied to the network output.
    - __W_constraint__: instance of the [constraints](../constraints.md) module (eg. maxnorm, nonneg), applied to the main weights matrix.
    - __b_constraint__: instance of the [constraints](../constraints.md) module, applied to the bias.

```python
# input shape: (nb_samples, 10)
model.add(Dense(10, 100)) # output shape: (nb_samples, 100)
model.add(MaxoutDense(100, 100, nb_feature=10)) # output shape: (nb_samples, 100)
model.add(RepeatVector(2))  # output shape: (nb_samples, 2, 10)
```

## Merge
```python
keras.layers.core.Merge(models, mode='sum')
```

Merge the output of a list of models into a single tensor, following one of two modes: `sum` or `concat`. 

- __Arguments__:
    - __models__: List of `Sequential` models.
    - __mode__: String, one of `{'sum', 'concat'}`. `sum` will simply sum the outputs of the models (therefore all models should have an output with the same shape). `concat` will concatenate the outputs along the last dimension (therefore all models should have an output that only differ along the last dimension). 

- __Example__:

```python
left = Sequential()
left.add(Dense(784, 50))
left.add(Activation('relu'))

right = Sequential()
right.add(Dense(784, 50))
right.add(Activation('relu'))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

model.add(Dense(50, 10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit([X_train, X_train], Y_train, batch_size=128, nb_epoch=20, validation_data=([X_test, X_test], Y_test))
```

