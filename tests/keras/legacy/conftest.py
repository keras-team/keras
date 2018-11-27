import warnings
import pytest


@pytest.fixture(autouse=True)
def clear_session_after_test():
    """This wrapper runs for all the tests in the legacy directory (recursively).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'(.+) Keras 2 ',
                                category=UserWarning)
        yield
