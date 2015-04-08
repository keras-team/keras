# Initializations

## Usage of initializations

Initializations define the probability distribution used to set the initial random weights of Keras layers.

The keyword arguments used for passing initializations to layers will depend on the layer. Usually it is simply `init`:

```python
model.add(Dense(64, 64, init='uniform'))
```

## Available initializations

- __uniform__
- __normal__
- __orthogonal__: use with square 2D layers (`shape[0] == shape[1]`).
- __zero__