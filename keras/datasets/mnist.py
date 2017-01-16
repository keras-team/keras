import gzip
from ..utils.data_utils import get_file
from six.moves import cPickle
import sys


def load_data(path='mnist.pkl.gz'):
    """Loads the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path, origin='https://s3.amazonaws.com/img-datasets/mnist.pkl.gz')

    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')

    f.close()
    return data  # (x_train, y_train), (x_test, y_test)
