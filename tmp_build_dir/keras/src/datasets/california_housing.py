"""Boston housing price regression dataset."""

import numpy as np

from keras.src.api_export import keras_export
from keras.src.utils.file_utils import get_file


@keras_export("keras.datasets.california_housing.load_data")
def load_data(
    version="large", path="california_housing.npz", test_split=0.2, seed=113
):
    """Loads the California Housing dataset.

    This dataset was obtained from the [StatLib repository](
    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).

    It's a continuous regression dataset with 20,640 samples with
    8 features each.

    The target variable is a scalar: the median house value
    for California districts, in dollars.

    The 8 input features are the following:

    - MedInc: median income in block group
    - HouseAge: median house age in block group
    - AveRooms: average number of rooms per household
    - AveBedrms: average number of bedrooms per household
    - Population: block group population
    - AveOccup: average number of household members
    - Latitude: block group latitude
    - Longitude: block group longitude

    This dataset was derived from the 1990 U.S. census, using one row
    per census block group. A block group is the smallest geographical
    unit for which the U.S. Census Bureau publishes sample data
    (a block group typically has a population of 600 to 3,000 people).

    A household is a group of people residing within a home.
    Since the average number of rooms and bedrooms in this dataset are
    provided per household, these columns may take surprisingly large
    values for block groups with few households and many empty houses,
    such as vacation resorts.

    Args:
        version: `"small"` or `"large"`. The small version
            contains 600 samples, the large version contains
            20,640 samples. The purpose of the small version is
            to serve as an approximate replacement for the
            deprecated `boston_housing` dataset.
        path: path where to cache the dataset locally
            (relative to `~/.keras/datasets`).
        test_split: fraction of the data to reserve as test set.
        seed: Random seed for shuffling the data
            before computing the test split.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    **`x_train`, `x_test`**: numpy arrays with shape `(num_samples, 8)`
      containing either the training samples (for `x_train`),
      or test samples (for `y_train`).

    **`y_train`, `y_test`**: numpy arrays of shape `(num_samples,)`
        containing the target scalars. The targets are float scalars
        typically between 25,000 and 500,000 that represent
        the home prices in dollars.
    """
    assert 0 <= test_split < 1
    origin_folder = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    )
    path = get_file(
        path,
        origin=f"{origin_folder}california_housing.npz",
        file_hash=(  # noqa: E501
            "1a2e3a52e0398de6463aebe6f4a8da34fb21fbb6b934cf88c3425e766f2a1a6f"
        ),
    )
    with np.load(path, allow_pickle=True) as f:
        x = f["x"]
        y = f["y"]

    if version == "small":
        x = x[:600]
        y = y[:600]
    elif version != "large":
        raise ValueError(
            "Argument `version` must be one of 'small', 'large'. "
            f"Received: version={version}"
        )

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
