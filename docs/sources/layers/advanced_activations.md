
## LeakyReLU

```python
keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
```

Special version of a Rectified Linear Unit that allows a small gradient when the unit is not active (`f(x) = alpha*x for x < 0`).


- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:
    - __alpha__: float >= 0. Negative slope coefficient.

---

## PReLU

```python
keras.layers.advanced_activations.PReLU()
```

Parametrized linear unit. Similar to a LeakyReLU, where each input unit has its alpha coefficient, and where these coefficients are learned during training.


- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

- __Output shape__: Same as input.

- __References__:
    - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)

---

## ELU

```python
keras.layers.advanced_activations.ELU()
```

Exponential linear unit. Negative values pushes mean unit activations closer to zero, with the advantage of having a noise-robust deactivation state.


- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

- __Output shape__: Same as input.

- __References__:
    - [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/pdf/1511.07289v1.pdf)

---

## ParametricSoftplus

```python
keras.layers.advanced_activations.ParametricSoftplus()
```

Parametric Softplus of the form: (`f(x) = alpha * (1 + exp(beta * x))`). This is essentially a smooth version of ReLU where the parameters control the sharpness of the rectification. The parameters are initialized to more closely approximate a ReLU than the standard `softplus`: `alpha` initialized to `0.2` and `beta`  initialized to `5.0`. The parameters are fit separately for each hidden unit.

- __Input shape__: Arbitrary. Use the keyword argument `input_shape=...` when using this layer as the first layer in a model.

- __Output shape__: Same as input.

- __References__:
    - [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)

## Thresholded Linear

```python
keras.layers.advanced_activations.ThresholdedLinear(theta)
```

Parametrized linear unit. provides a threshold near zero where values are zeroed.


- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:
    - __theta__: float >= 0. Threshold location of activation

- __References__:
    - [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)

## Thresholded ReLu

```python
keras.layers.advanced_activations.ThresholdedReLu(theta)
```

Parametrized rectified linear unit. provides a threshold near zero where values are zeroed.

- __Input shape__: Arbitrary. Use the keyword argument `input_shape=...` when using this layer as the first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:
    - __theta__: float >= 0. Threshold location of activation

- __References__:
    - [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)
