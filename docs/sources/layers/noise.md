

## GaussianNoise
```python
keras.layers.core.GaussianNoise(sigma)
```
Apply to the input an additive zero-centred gaussian noise with standard deviation `sigma`. Gaussian Noise (GS) is a natural choise as corruption process for real valued inputs.

- __Input shape__: This layer does not assume a specific input shape. 

- __Output shape__: Same as input.

- __Arguments__:

    - __sigma__: float, standard deviation of the noise distribution.

---