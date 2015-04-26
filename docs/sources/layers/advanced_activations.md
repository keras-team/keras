
## LeakyReLU

```python
keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
```

Special version of a Rectified Linear Unit that allows a small gradient when the unit is not active (`f(x) = alpha*x for x < 0`).

- __Input shape__: This layer does not assume a specific input shape. As a result, it cannot be used as the first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:
    - __alpha__: float >= 0. Negative slope coefficient.

---

## PReLU

```python
keras.layers.advanced_activations.PReLU(input_shape)
```

Parametrized linear unit. Similar to a LeakyReLU, where each input unit has its alpha coefficient, and where these coefficients are learned during training.

- __Input shape__: Same as `input_shape`. This layer cannot be used as first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:
    - __input_shape__: tuple.

- __References__:
    - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)