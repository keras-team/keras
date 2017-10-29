from docs.autogen import *


def test_post_process_signature():
    signature = 'foo.layers.core.Bar(p1, p2)'
    actual = post_process_signature(signature)
    expected = 'keras.layers.Bar(p1, p2)'
    assert actual == expected


def test_get_class_signature():
    class Foo:
        def __init__(self, p1, p2, *args):
            pass

    actual = get_class_signature(Foo)
    expected = 'test_autogen.Foo(p1, p2)'
    assert actual == expected


if __name__ == '__main__':
    pytest.main([__file__])
