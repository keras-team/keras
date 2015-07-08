
## Usage of objectives

An objective function (or loss function, or optimization score function) is one of the two parameters required to compile a model:

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

You can either pass the name of an existing objective, or pass a Theano symbolic function that returns a scalar for each data-point and takes the following two arguments:

- __y_true__: True labels. Theano tensor.
- __y_pred__: Predictions. Theano tensor of the same shape as y_true.

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
