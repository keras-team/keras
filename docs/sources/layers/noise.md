

## GaussianNoise
```python
keras.layers.core.GaussianNoise(sigma)
```
Apply to the input an additive zero-centred gaussian noise with standard deviation `sigma`. This is useful to mitigate overfitting (you could see it as a kind of random data augmentation). Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.

The gaussian noise is only added at training time.

- __Input shape__: This layer does not assume a specific input shape. 

- __Output shape__: Same as input.

- __Arguments__:

    - __sigma__: float, standard deviation of the noise distribution.

---