# Wrappers for the Scikit-Learn API

You can use `Sequential` (or single-input/output `Model`) Keras models as part of your Scikit-Learn workflow via the wrappers found at `keras.wrappers.scikit_learn.py`.

There are two wrappers available:

`keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)`, which implements the Scikit-Learn classifier interface,

`keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)`, which implements the Scikit-Learn regressor interface.

Also, the more general `Model` Keras models can be used with the generic `keras.wrappers.scikit_learn.BaseWrapper(build_fn=None, **sk_params)` wrapper, but they will not be compatible with Scikit-Learn unless restricted to single-input and single-output.

### Arguments

- __build_fn__: callable function or class instance
- __sk_params__: model parameters & fitting parameters

The `build_fn` should construct, compile and return a Keras model, which
will then be used to fit/predict. It must accept `input_shape` and
`output_shape` as arguments, both of which are tuples of integers or
dictionaries (for named multi-input/output models). One of the following three
values could be passed to `build_fn`:

1. A function
2. An instance of a class that implements the `__call__` method
3. None. This means you implement a class that inherits from either
`BaseWrapper`, `KerasClassifier` or `KerasRegressor`. The `__call__` method of
the present class will then be treated as the default `build_fn`.

`sk_params` takes both model parameters and fitting parameters. Legal model
parameters are the arguments of `build_fn`. Note that like all other
estimators in scikit-learn, `build_fn` should provide default values for
its arguments, so that you could create the estimator without passing any
values to `sk_params`.

`sk_params` could also accept parameters for calling `fit`, `predict`,
`predict_proba`, and `score` methods (e.g., `epochs`, `batch_size`).
fitting (predicting) parameters are selected in the following order:

1. Values passed to the dictionary arguments of
`fit`, `predict`, `predict_proba`, and `score` methods
2. Values passed to `sk_params`
3. The default values of the `keras.models.Sequential`
`fit`, `predict`, `predict_proba` and `score` methods, or the default values of
the `keras.models.Model` `fit`, `predict` and `score` methods.

When using scikit-learn's `grid_search` API, legal tunable parameters are
those you could pass to `sk_params`, including fitting parameters.
In other words, you could use `grid_search` to search for the best
`batch_size` or `epochs` as well as the model parameters.
