import sys
import pytest
import numpy as np
from keras.utils.generic_utils import custom_object_scope
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import Progbar
from keras.utils.test_utils import keras_test
from keras import activations
from keras import regularizers


@keras_test
def test_progbar():
    n = 2
    input_arr = np.random.random((n, n, n))
    bar = Progbar(n)

    for i, arr in enumerate(input_arr):
        bar.update(i, list(arr))


def test_custom_objects_scope():

    def custom_fn():
        pass

    class CustomClass(object):
        pass

    with custom_object_scope({'CustomClass': CustomClass,
                              'custom_fn': custom_fn}):
        act = activations.get('custom_fn')
        assert act == custom_fn
        cl = regularizers.get('CustomClass')
        assert cl.__class__ == CustomClass


@pytest.mark.parametrize('fn, name, accept_all, expected', [
    ('f(x)', 'x', False, True),
    ('f(x)', 'y', False, False),
    ('f(x)', 'y', True, False),
    ('f(x, y)', 'y', False, True),
    ('f(x, y=1)', 'y', False, True),
    ('f(x, **kwargs)', 'x', False, True),
    ('f(x, **kwargs)', 'y', False, False),
    ('f(x, **kwargs)', 'y', True, True),
    ('f(x, y=1, **kwargs)', 'y', False, True),
    # Keyword-only arguments (Python 3 only)
    ('f(x, *args, y=1)', 'y', False, True),
    ('f(x, *args, y=1)', 'z', True, False),
    ('f(x, *, y=1)', 'x', False, True),
    ('f(x, *, y=1)', 'y', False, True),
    # lambda
    (lambda x: x, 'x', False, True),
    (lambda x: x, 'y', False, False),
    (lambda x: x, 'y', True, False),
])
def test_has_arg(fn, name, accept_all, expected):
    if isinstance(fn, str):
        context = dict()
        try:
            exec('def {}: pass'.format(fn), context)
        except SyntaxError:
            if sys.version_info >= (3,):
                raise
            pytest.skip('Function is not compatible with Python 2')
        context.pop('__builtins__', None)  # Sometimes exec adds builtins to the context
        fn, = context.values()

    assert has_arg(fn, name, accept_all) is expected


@pytest.mark.xfail(sys.version_info < (3, 3),
                   reason='inspect API does not reveal positional-only arguments')
def test_has_arg_positional_only():
    assert has_arg(pow, 'x') is False


if __name__ == '__main__':
    pytest.main([__file__])
