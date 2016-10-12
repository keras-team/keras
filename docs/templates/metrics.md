
## Usage of metrics

A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the `metrics` parameter when a model is compiled.

A metric function is similar to an [objective function](/objectives), except that the results from evaluating a metric are not used when training the model.

You can either pass the name of an existing metric, or pass a Theano/TensorFlow symbolic function that returns a scalar for each data-point and takes the following two arguments:

- __y_true__: True labels. Theano/TensorFlow tensor.
- __y_pred__: Predictions. Theano/TensorFlow tensor of the same shape as y_true.

The actual metric is the mean of the output array across all datapoints.

For a few examples of such functions, check out the [metrics source](https://github.com/fchollet/keras/blob/master/keras/metrics.py).

## Available metrics

- __accuracy__ : This will use the relevant function for your classification problem (choosing between __binary_accuracy__ , __categorical_accuracy__ and __sparse_categorical_accuracy__)
- __top_k_categorical_accuracy__
- __mean_squared_error__ / __mse__
- __mean_absolute_error__ / __mae__
- __mean_absolute_percentage_error__ / __mape__
- __mean_squared_logarithmic_error__ / __msle__
- __squared_hinge__
- __hinge__
- __binary_crossentropy__: Also known as logloss.
- __categorical_crossentropy__: Also known as multiclass logloss. __Note__: using this metric requires that your labels are binary arrays of shape `(nb_samples, nb_classes)`.
- __sparse_categorical_crossentropy__: As above but accepts sparse labels. __Note__: this metric still requires that your labels have the same number of dimensions as your outputs; you may need to add a length-1 dimension to the shape of your labels, e.g with `np.expand_dims(y, -1)`.
- __kullback_leibler_divergence__ / __kld__: Information gain from a predicted probability distribution Q to a true probability distribution P. Gives a measure of difference between both distributions.
- __poisson__: Mean of `(predictions - targets * log(predictions))`
- __cosine_proximity__: The opposite (negative) of the mean cosine proximity between predictions and targets.
- __matthews_correlation__
- __fbeta_score__

**Note**: when using the `categorical_crossentropy` metric, your targets should be in categorical format (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros expect for a 1 at the index corresponding to the class of the sample). In order to convert *integer targets* into *categorical targets*, you can use the Keras utility `to_categorical`:

```python
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, nb_classes=None)
```

