
## Convolution1D

```python
keras.layers.convolutional.Convolution1D(nb_filter, stack_size, filter_length, 
        init='uniform', activation='linear', weights=None, 
        image_shape=None, border_mode='valid', subsample_length=1, 
        W_regularizer=None, b_regularizer=None, W_constraint=None, 
        b_constraint=None)
```

Convolution operator for filtering neighborhoods of one-dimensional inputs.

---

## Convolution2D

```python
keras.layers.convolutional.Convolution2D(nb_filter, stack_size, nb_row, nb_col, 
        init='glorot_uniform', activation='linear', weights=None, 
        image_shape=None, border_mode='valid', subsample=(1,1))
```

Convolution operator for filtering windows of two-dimensional inputs. This is a wrapper for Theano's [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d). 

---

## MaxPooling1D

```python
keras.layers.convolutional.MaxPooling1D(pool_length=2, ignore_border=True)
```
---

## MaxPooling2D

```python
keras.layers.convolutional.MaxPooling2D(poolsize=(2, 2), ignore_border=True)
```

This is a wrapper for Theano's [max_pool_2d](http://deeplearning.net/software/theano/library/tensor/signal/downsample.html).
