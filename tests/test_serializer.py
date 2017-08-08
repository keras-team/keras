from keras.serializer import _slash_get, _slash_set
from keras.utils.test_utils import keras_test


@keras_test
def test_slash_get():
    assert _slash_get({'a': {'b': 'c'}}, 'a/b') == 'c'

    assert _slash_get({'a': {'b': 'c'}}, '/a') == {'b': 'c'}


@keras_test
def test_slash_set():
    a = {}
    _slash_set(a, 'a/b', 'c')
    assert a == {'a': {'b': 'c'}}

    _slash_set(a, 'a/b', 'd')
    assert a == {'a': {'b': 'd'}}

    _slash_set(a, 'a/c', 'd')
    assert a == {'a': {'b': 'd', 'c': 'd'}}

    a = {}
    _slash_set(a, '/a', 'c')
    assert a == {'a': 'c'}
