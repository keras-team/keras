Containers are ensembles of layers that can be interacted with through the same API as `Layer` objects.

## Sequential

```python
keras.layers.containers.Sequential(layers=[])
```

The Sequential container is a linear stack of layers. Apart from the `add` methods and the `layers` constructor argument, the API is identical to that of the `Layer` class.

This class is also the basis for the `keras.models.Sequential` architecture.

The `layers` constructor argument is a list of Layer instances.

__Methods__:

```python
add(layer)
```

Add a new layer to the stack.