## Usage of constraints

Functions from the `constraints` module allow setting constraints (eg. non-negativity) on network parameters during optimization.

The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers `Dense`, `TimeDistributedDense`, `MaxoutDense`, `Convolution1D` and `Convolution2D` have a unified API.

These layers expose 2 keyword arguments:

- `W_constraint` for the main weights matrix
- `b_constraint` for the bias.


```python
from keras.constraints import maxnorm
model.add(Dense(64, 64, W_constraint = maxnorm(2)))
```

## Available constraints

- __maxnorm__(m=2): maximum-norm constraint
- __nonneg__(): non-negativity constraint
- __unitnorm__(): unit-norm constraint, enforces the matrix to have unit norm along the last axis