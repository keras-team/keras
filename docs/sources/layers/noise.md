

## GaussianNoise
```python
keras.layers.noise.GaussianNoise(sigma)
```
Apply to the input an additive zero-centred gaussian noise with standard deviation `sigma`. This is useful to mitigate overfitting (you could see it as a kind of random data augmentation). Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.

Only active at training time.

- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model. To specify the number of samples per batch, you can use the keyword argument `batch_input_shape` (tuple of integers, including the samples axis).

- __Output shape__: Same as input.

- __Arguments__:

    - __sigma__: float, standard deviation of the noise distribution.

---

## GaussianDropout
```python
keras.layers.noise.GaussianDropout(p)
```
Apply to the input an multiplicative one-centred gaussian noise with standard deviation `sqrt(p/(1-p))`. p refers to drop probability to match Dropout layer syntax. 

Only active at training time.

- __Input shape__: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model. To specify the number of samples per batch, you can use the keyword argument `batch_input_shape` (tuple of integers, including the samples axis).

- __Output shape__: Same as input.

- __Arguments__:

    - __p__: float, drop probability as with Dropout.


