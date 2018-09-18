import sys
import pytest
import numpy as np
import marshal
from keras.utils.generic_utils import custom_object_scope
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras import activations
from keras import regularizers


def test_progbar():
    values_s = [None,
                [['key1', 1], ['key2', 1e-4]],
                [['key3', 1], ['key2', 1e-4]]]

    for target in (len(values_s) - 1, None):
        for verbose in (0, 1, 2):
            bar = Progbar(target, width=30, verbose=verbose, interval=0.05)
            for current, values in enumerate(values_s):
                bar.update(current, values=values)


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
        # Sometimes exec adds builtins to the context
        context.pop('__builtins__', None)
        fn, = context.values()

    assert has_arg(fn, name, accept_all) is expected


@pytest.mark.xfail(sys.version_info < (3, 3),
                   reason='inspect API does not reveal positional-only arguments')
def test_has_arg_positional_only():
    assert has_arg(pow, 'x') is False


@pytest.mark.parametrize(
    'test_function_type',
    ('simple function', 'closured function'))
def test_func_dump_and_load(test_function_type):

    if test_function_type == 'simple function':
        def test_func():
            return r'\u'

    elif test_function_type == 'closured function':
        def get_test_func():
            x = r'\u'

            def test_func():
                return x
            return test_func
        test_func = get_test_func()
    else:
        raise Exception('Unknown test case for test_func_dump_and_load')

    serialized = func_dump(test_func)
    deserialized = func_load(serialized)
    assert deserialized.__code__ == test_func.__code__
    assert deserialized.__defaults__ == test_func.__defaults__
    assert deserialized.__closure__ == test_func.__closure__


def test_func_dump_and_load_closure():
    y = 0
    test_func = lambda x: x + y
    serialized, _, closure = func_dump(test_func)
    deserialized = func_load(serialized, closure=closure)
    assert deserialized.__code__ == test_func.__code__
    assert deserialized.__defaults__ == test_func.__defaults__
    assert deserialized.__closure__ == test_func.__closure__


@pytest.mark.parametrize(
    'test_func', [activations.softmax, np.argmax, lambda x: x**2, lambda x: x])
def test_func_dump_and_load_backwards_compat(test_func):
    # this test ensures that models serialized prior to version 2.1.2 can still be
    # deserialized

    # see:
    # https://github.com/evhub/keras/blob/2.1.1/keras/utils/generic_utils.py#L166
    serialized = marshal.dumps(test_func.__code__).decode('raw_unicode_escape')

    deserialized = func_load(serialized, defaults=test_func.__defaults__)
    assert deserialized.__code__ == test_func.__code__
    assert deserialized.__defaults__ == test_func.__defaults__
    assert deserialized.__closure__ == test_func.__closure__

if __name__ == '__main__':
    pytest.main([__file__])
