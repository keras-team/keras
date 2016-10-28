
## Usage of initializations

Initializations define the way to set the initial random weights of Keras layers.

The keyword arguments used for passing initializations to layers will depend on the layer. Usually it is simply `init`:

```python
model.add(Dense(64, init='uniform'))
```

## Available initializations

- __uniform__
- __lecun_uniform__: Uniform initialization scaled by the square root of the number of inputs (LeCun 98).
- __normal__
- __identity__: Use with square 2D layers (`shape[0] == shape[1]`).
- __orthogonal__: Use with square 2D layers (`shape[0] == shape[1]`).
- __zero__
- __glorot_normal__: Gaussian initialization scaled by fan_in + fan_out (Glorot 2010)
- __glorot_uniform__
- __he_normal__: Gaussian initialization scaled by fan_in (He et al., 2014)
- __he_uniform__


An initialization may be passed as a string (must match one of the available initializations above), or as a callable.
If a callable, then it must take two arguments: `shape` (shape of the variable to initialize) and `name` (name of the variable),
and it must return a variable (e.g. output of `K.variable()`):

```python
from keras import backend as K
import numpy as np

def my_init(shape, name=None):
    value = np.random.random(shape)
    return K.variable(value, name=name)

model.add(Dense(64, init=my_init))
```

You could also use functions from `keras.initializations` in this way:

```python
from keras import initializations

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

model.add(Dense(64, init=my_init))
```