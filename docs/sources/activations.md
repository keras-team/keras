
## Usage of activations

Activations can either be used through an `Activation` layer, or through the `activation` argument supported by all forward layers:

```python
from keras.layers.core import Activation, Dense

model.add(Dense(64, 64, init='uniform'))
model.add(Activation('tanh'))
```
is equivalent to:
```python
model.add(Dense(20, 64, init='uniform', activation='tanh'))
```

You can also pass an element-wise Theano function as an activation:

```python
def tanh(x):
    return theano.tensor.tanh(x)

model.add(Dense(20, 64, init='uniform', activation=tanh))
model.add(Activation(tanh))
```

## Available activations

- __softmax__: Softmax applied across inputs last dimension. Expects shape either `(nb_samples, nb_timesteps, nb_dims)` or `(nb_samples, nb_dims)`.
- __softplus__
- __relu__
- __tanh__
- __sigmoid__
- __hard_sigmoid__
- __linear__

## On Advanced Activations

Activations that are more complex than a simple Theano function (eg. learnable activations, configurable activations, etc.) are available as [Advanced Activation layers](layers/advanced_activations.md), and can be found in the module `keras.layers.advanced_activations`. These include PReLU and LeakyReLU.
