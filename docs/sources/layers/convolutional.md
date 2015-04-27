
## Convolution2D

```python
keras.layers.convolutional.Convolution2D(nb_filter, stack_size, nb_row, nb_col, 
        init='glorot_uniform', activation='linear', weights=None, 
        image_shape=None, border_mode='valid', subsample=(1,1))
```

This is a wrapper for Theano's [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d). 

---

## MaxPooling2D

```python
keras.layers.convolutional.MaxPooling2D(poolsize=(2, 2), ignore_border=True)
```

This is a wrapper for Theano's [max_pool_2d](http://deeplearning.net/software/theano/library/tensor/signal/downsample.html).
