from ..utils.data_utils import get_file
import numpy as np


def load_data(path='boston_housing.npz', seed=113, test_split=0.2):
    """Loads the Boston Housing dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
        seed: Random seed for shuffling the data
            before computing the test split.
        test_split: fraction of the data to reserve as test set.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    assert 0 <= test_split < 1
    path = get_file(path,
                    origin='https://s3.amazonaws.com/keras-datasets/boston_housing.npz',
                    file_hash='f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5')
    f = np.load(path)
    x = f['x']
    y = f['y']
    f.close()

    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)
