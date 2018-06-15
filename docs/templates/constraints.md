## Usage of constraints

Functions from the `constraints` module allow setting constraints (eg. non-negativity) on network parameters during optimization.

The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers `Dense`, `Conv1D`, `Conv2D` and `Conv3D` have a unified API.

These layers expose 2 keyword arguments:

- `kernel_constraint` for the main weights matrix
- `bias_constraint` for the bias.


```python
from keras.constraints import max_norm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## Available constraints

- __max_norm(max_value=2, axis=0)__: maximum-norm constraint
- __non_neg()__: non-negativity constraint
- __unit_norm(axis=0)__: unit-norm constraint
- __min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)__:  minimum/maximum-norm constraint
