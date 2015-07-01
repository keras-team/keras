
## Usage of initializations

Initializations define the probability distribution used to set the initial random weights of Keras layers.

The keyword arguments used for passing initializations to layers will depend on the layer. Usually it is simply `init`:

```python
model.add(Dense(64, 64, init='uniform'))
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
