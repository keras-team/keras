import numpy as np

from keras.src.api_export import keras_export
from keras.src.utils.file_utils import get_file


@keras_export("keras.datasets.boston_housing.load_data")
def load_data(path="boston_housing.npz", test_split=0.2, seed=113):
    """Loads the Boston Housing dataset.

    This is a dataset taken from the StatLib library which is maintained at
    Carnegie Mellon University.

    **WARNING:** This dataset has an ethical problem: the authors of this
    dataset included a variable, "B", that may appear to assume that racial
    self-segregation influences house prices. As such, we strongly discourage
    the use of this dataset, unless in the context of illustrating ethical
    issues in data science and machine learning.

    Samples contain 13 attributes of houses at different locations around the
    Boston suburbs in the late 1970s. Targets are the median values of
    the houses at a location (in k$).

    The attributes themselves are defined in the
    [StatLib website](http://lib.stat.cmu.edu/datasets/boston).

    Args:
        path: path where to cache the dataset locally
            (relative to `~/.keras/datasets`).
        test_split: fraction of the data to reserve as test set.
        seed: Random seed for shuffling the data
            before computing the test split.

    Returns:
        Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train, x_test**: NumPy arrays with shape `(num_samples, 13)`
        containing either the training samples (for x_train),
        or test samples (for y_train).

    **y_train, y_test**: NumPy arrays of shape `(num_samples,)` containing the
        target scalars. The targets are float scalars typically between 10 and
        50 that represent the home prices in k$.
    """
    assert 0 <= test_split < 1
    origin_folder = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    )
    path = get_file(
        path,
        origin=f"{origin_folder}boston_housing.npz",
        file_hash=(  # noqa: E501
            "f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5"
        ),
    )
    with np.load(path, allow_pickle=True) as f:
        x = f["x"]
        y = f["y"]

    rng = np.random.RandomState(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x_train = np.array(x[: int(len(x) * (1 - test_split))])
    y_train = np.array(y[: int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)) :])
    y_test = np.array(y[int(len(x) * (1 - test_split)) :])
    return (x_train, y_train), (x_test, y_test)
