
## BatchNormalization

```python
keras.layers.normalization.BatchNormalization(epsilon=1e-6, weights=None)
```

Normalize the activations of the previous layer at each batch.

- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

- __Output shape__: Same as input.

- __Arguments__: 
    - __epsilon__: small float > 0. Fuzz parameter.
    - __weights__: Initialization weights. List of 2 numpy arrays, with shapes: `[(input_shape,), (input_shape,)]`

- __References__:
    - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf)