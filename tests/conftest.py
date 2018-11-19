import pytest
from keras import backend as K


@pytest.fixture(autouse=True)
def clear_session_after_test():
    """Test wrapper to clean up after TensorFlow and CNTK tests.

    This wrapper runs for all the tests in the keras test suite.
    """
    yield
    if K.backend() == 'tensorflow' or K.backend() == 'cntk':
        K.clear_session()
