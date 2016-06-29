
## Usage of objectives

An objective function (or loss function, or optimization score function) is one of the two parameters required to compile a model:

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

You can either pass the name of an existing objective, or pass a Theano/TensorFlow symbolic function that returns a scalar for each data-point and takes the following two arguments:

- __y_true__: True labels. Theano/TensorFlow tensor.
- __y_pred__: Predictions. Theano/TensorFlow tensor of the same shape as y_true.

The actual optimized objective is the mean of the output array across all datapoints.

For a few examples of such functions, check out the [objectives source](https://github.com/fchollet/keras/blob/master/keras/objectives.py).

## Available objectives

- __mean_squared_error__ / __mse__
- __mean_absolute_error__ / __mae__
- __mean_absolute_percentage_error__ / __mape__
- __mean_squared_logarithmic_error__ / __msle__
- __squared_hinge__
- __hinge__
- __binary_crossentropy__: Also known as logloss. 
- __categorical_crossentropy__: Also known as multiclass logloss. __Note__: using this objective requires that your labels are binary arrays of shape `(nb_samples, nb_classes)`.
- __sparse_categorical_crossentropy__: As above but accepts sparse labels. __Note__: this objective still requires that your labels have the same number of dimensions as your outputs; you may need to add a length-1 dimension to the shape of your labels, e.g with `np.expand_dims(y, -1)`.
- __kullback_leibler_divergence__ / __kld__: Information gain from a predicted probability distribution Q to a true probability distribution P. Gives a measure of difference between both distributions.
- __poisson__: Mean of `(predictions - targets * log(predictions))`
- __cosine_proximity__: The opposite (negative) of the mean cosine proximity between predictions and targets.
