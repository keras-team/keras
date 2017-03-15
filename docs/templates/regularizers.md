## Usage of regularizers

Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.

The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers `Dense`, `Conv1D`, `Conv2D` and `Conv3D` have a unified API.

These layers expose 3 keyword arguments:

- `kernel_regularizer`: instance of `keras.regularizers.Regularizer`
- `bias_regularizer`: instance of `keras.regularizers.Regularizer`
- `activity_regularizer`: instance of `keras.regularizers.Regularizer`

## Available penalties

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(0., 0.)
```

## Example

```python
from keras.regularizers import l2, l1l2
# l1l2(0, 0.01) is equal to l2(0.01), which sets l1 to 0.
model.add(Dense(64, input_dim=64,
                W_regularizer=l1l2(0, 0.01),
                kernel_regularizer=l2(0.01),
                activity_regularizer=l2(0.01)))
```

## Developing new regularizers

Any function that takes in a weight matrix and returns a loss contribution tensor can be used as a regularizer, e.g.:

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg)
```

Alternatively, you can write your regularizers in an object-oriented way;
see the [keras/regularizers.py](https://github.com/fchollet/keras/blob/master/keras/regularizers.py) module for examples.
