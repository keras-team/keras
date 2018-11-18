import warnings
import pytest
from keras import backend as K

warning_msg = r'your data is of type "float64", but your input ' \
              r'variable (.*) expects (.*). Please convert your ' \
              r'data beforehand to speed up training.'


@pytest.fixture(autouse=True)
def clear_session_after_test():
    """Test wrapper to clean up after TensorFlow and CNTK tests.

    This wrapper runs for all the tests in the keras test suite.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=warning_msg, category=UserWarning)
        yield

    if K.backend() == 'tensorflow' or K.backend() == 'cntk':
        K.clear_session()
