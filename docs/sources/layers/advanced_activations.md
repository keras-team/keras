
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

---

## ParametricSoftplus

```python
keras.layers.advanced_activations.ParametricSoftplus(input_shape)
```

Parametric Softplus of the form: (`f(x) = alpha * (1 + exp(beta * x))`). This is essentially a smooth version of ReLU where the parameters control the sharpness of the rectification. The parameters are initialized to more closely approximate a ReLU than the standard `softplus`: `alpha` initialized to `0.2` and `beta`  initialized to `5.0`. The parameters are fit separately for each hidden unit.

- __Input shape__: Same as `input_shape`. This layer cannot be used as first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:
    - __input_shape__: tuple.

- __References__:
    - [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)