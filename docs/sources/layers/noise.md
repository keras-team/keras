## Dropout
```python
keras.layers.core.Dropout(p)
```
Apply dropout to the input. Dropout consists in randomly setting a fraction `p` of input units to 0 at each update during training time, which helps prevent overfitting. Reference: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

- __Input shape__: This layer does not assume a specific input shape. As a result, it cannot be used as the first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:

    - __p__: float (0 <= p < 1). Fraction of the input that gets dropped out at training time. 


---

## GaussianNoise
```python
keras.layers.core.GaussianNoise(p)
```
Apply Gaussian dropout (multiplicative Gaussain noise) to the input. Gaussian dropout consists in multiplying the input units by samples from Gaussian with mean 0 and variance `p/(1-p)` each update during training time, which helps prevent overfitting. Reference: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

- __Input shape__: This layer does not assume a specific input shape. As a result, it cannot be used as the first layer in a model.

- __Output shape__: Same as input.

- __Arguments__:

    - __p__: float (0 <= p < 1). Controls variance of multiplicative noise. 