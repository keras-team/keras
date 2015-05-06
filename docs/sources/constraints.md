## Usage of constraints

Functions from the `constraints` module allow setting constraints (eg. non-negativity) on network parameters during optimization.

The keyword arguments used for passing constraints to parameters in a layer will depend on the layer. 

In the `Dense` layer it is simply `W_constraint` for the main weights matrix, and `b_constraint` for the bias.


```python
from keras.constraints import maxnorm
model.add(Dense(64, 64, W_constraint = maxnorm(2)))
```

## Available constraints

- __maxnorm__(m=2): maximum-norm constraint
- __nonneg__(): non-negativity constraint