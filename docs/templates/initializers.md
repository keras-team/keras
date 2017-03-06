## Usage of initializers

Initializations define the way to set the initial random weights of Keras layers.

The keyword arguments used for passing initializers to layers will depend on the layer. Usually it is simply `kernel_initializer` and `bias_initializer`:

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## Available initializers

- __random_uniform__
- __lecun_uniform__ (LeCun 98).
- __random_normal__
- __truncated_normal__
- __identity__: Use with square 2D weights (`shape[0] == shape[1]`).
- __orthogonal__: Use with 2D weights.
- __zeros__
- __ones__
- __glorot_normal__ (Glorot 2010)
- __glorot_uniform__
- __he_normal__ (He et al., 2014)
- __he_uniform__


An initializer may be passed as a string (must match one of the available initializers above), or as a callable:

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works
model.add(Dense(64, kernel_initializer='random_normal'))
```

If passing a custom callable, then it must take the argument `shape` (shape of the variable to initialize) and `dtype` (dtype of generated values):

```python
from keras import backend as K
import numpy as np

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, init=my_init))
```
