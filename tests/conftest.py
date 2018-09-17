import pytest
from keras import backend as K


@pytest.fixture(autouse=True)
def clear_session_after_test():
    yield
    if K.backend() == 'tensorflow' or K.backend() == 'cntk':
        K.clear_session()
