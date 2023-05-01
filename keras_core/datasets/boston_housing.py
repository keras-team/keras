from keras_core.api_export import keras_core_export


@keras_core_export("keras_core.datasets.boston_housing.load_data")
def load_data(path="boston_housing.npz", test_split=0.2, seed=113):
    raise NotImplementedError(
        "The Boston Housing dataset is no longer distributed with Keras. "
        "We recommend that you use instead the "
        "California Housing dataset, available via "
        "`keras_core.datasets.california_housing.load_data()`."
    )
