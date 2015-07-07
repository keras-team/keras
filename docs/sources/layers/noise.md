

## GaussianNoise
```python
keras.layers.noise.GaussianNoise(sigma)
```
Apply to the input an additive zero-centred gaussian noise with standard deviation `sigma`. This is useful to mitigate overfitting (you could see it as a kind of random data augmentation). Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.

The Gaussian noise is only added at training time.

- __Input shape__: This layer does not assume a specific input shape. 

- __Output shape__: Same as input.

- __Arguments__:

    - __sigma__: float, standard deviation of the noise distribution.

---

## GaussianDropout
```python
keras.layers.noise.GaussianDropout(p)
```
Apply to the input an multiplicative one-centred gaussian noise with standard deviation `sqrt(p/(1-p))`. p refers to drop probability to match Dropout layer syntax. 

http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

The Gaussian noise is only used at training time.

- __Input shape__: This layer does not assume a specific input shape. 

- __Output shape__: Same as input.

- __Arguments__:

    - __p__: float, drop probability as with Dropout.

