from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json

class MisbehavingTextSubtype(text_type):
    def __str__(self):
        return "FAIL!"

class MisbehavingBytesSubtype(binary_type):
    def decode(self, encoding=None):
        return "bad decode"
    def __str__(self):
        return "bad __str__"
    def __bytes__(self):
        return b("bad __bytes__")

def as_text_type(s):
    if PY3 and isinstance(s, bytes):
        return s.decode('ascii')
    return s

def decode_iso_8859_15(b):
    return b.decode('iso-8859-15')

class TestDump(TestCase):
    def test_dump(self):
        sio = StringIO()
        json.dump({}, sio)
        self.assertEqual(sio.getvalue(), '{}')

    def test_constants(self):
        for c in [None, True, False]:
            self.assertTrue(json.loads(json.dumps(c)) is c)
            self.assertTrue(json.loads(json.dumps([c]))[0] is c)
            self.assertTrue(json.loads(json.dumps({'a': c}))['a'] is c)

    def test_stringify_key(self):
        items = [(b('bytes'), 'bytes'),
                 (1.0, '1.0'),
                 (10, '10'),
                 (True, 'true'),
                 (False, 'false'),
                 (None, 'null'),
                 (long_type(100), '100')]
        for k, expect in items:
            self.assertEqual(
                json.loads(json.dumps({k: expect})),
                {expect: expect})
            self.assertEqual(
                json.loads(json.dumps({k: expect}, sort_keys=True)),
                {expect: expect})
        self.assertRaises(TypeError, json.dumps, {json: 1})
        for v in [{}, {'other': 1}, {b('derp'): 1, 'herp': 2}]:
            for sort_keys in [False, True]:
                v0 = dict(v)
                v0[json] = 1
                v1 = dict((as_text_type(key), val) for (key, val) in v.items())
                self.assertEqual(
                    json.loads(json.dumps(v0, skipkeys=True, sort_keys=sort_keys)),
                    v1)
                self.assertEqual(
                    json.loads(json.dumps({'': v0}, skipkeys=True, sort_keys=sort_keys)),
                    {'': v1})
                self.assertEqual(
                    json.loads(json.dumps([v0], skipkeys=True, sort_keys=sort_keys)),
                    [v1])

    def test_dumps(self):
        self.assertEqual(json.dumps({}), '{}')

    def test_encode_truefalse(self):
        self.assertEqual(json.dumps(
                 {True: False, False: True}, sort_keys=True),
                 '{"false": true, "true": false}')
        self.assertEqual(
            # load first because the keys are not sorted
            json.loads(json.dumps({'k1': {False: 5}, 'k2': {0: 5}})),
            {'k1': {'false': 5}, 'k2': {'0': 5}},
        )

        self.assertEqual(
            json.dumps(
                {2: 3.0,
                 4.0: long_type(5),
                 False: 1,
                 long_type(6): True,
                 "7": 0},
                sort_keys=True),
            '{"2": 3.0, "4.0": 5, "6": true, "7": 0, "false": 1}')

    def test_ordered_dict(self):
        # http://bugs.python.org/issue6105
        items = [('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)]
        s = json.dumps(json.OrderedDict(items))
        self.assertEqual(
            s,
            '{"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}')

    def test_indent_unknown_type_acceptance(self):
        """
        A test against the regression mentioned at `github issue 29`_.

        The indent parameter should accept any type which pretends to be
        an instance of int or long when it comes to being multiplied by
        strings, even if it is not actually an int or long, for
        backwards compatibility.

        .. _github issue 29:
           http://github.com/simplejson/simplejson/issue/29
        """

        class AwesomeInt(object):
            """An awesome reimplementation of integers"""

            def __init__(self, *args, **kwargs):
                if len(args) > 0:
                    # [construct from literals, objects, etc.]
                    # ...

                    # Finally, if args[0] is an integer, store it
                    if isinstance(args[0], int):
                        self._int = args[0]

            # [various methods]

            def __mul__(self, other):
                # [various ways to multiply AwesomeInt objects]
                # ... finally, if the right-hand operand is not awesome enough,
                # try to do a normal integer multiplication
                if hasattr(self, '_int'):
                    return self._int * other
                else:
                    raise NotImplementedError("To do non-awesome things with"
                        " this object, please construct it from an integer!")

        s = json.dumps([0, 1, 2], indent=AwesomeInt(3))
        self.assertEqual(s, '[\n   0,\n   1,\n   2\n]')

    def test_accumulator(self):
        # the C API uses an accumulator that collects after 100,000 appends
        lst = [0] * 100000
        self.assertEqual(json.loads(json.dumps(lst)), lst)

    def test_sort_keys(self):
        # https://github.com/simplejson/simplejson/issues/106
        for num_keys in range(2, 32):
            p = dict((str(x), x) for x in range(num_keys))
            sio = StringIO()
            json.dump(p, sio, sort_keys=True)
            self.assertEqual(sio.getvalue(), json.dumps(p, sort_keys=True))
            self.assertEqual(json.loads(sio.getvalue()), p)

    def test_misbehaving_text_subtype(self):
        # https://github.com/simplejson/simplejson/issues/185
        text = "this is some text"
        self.assertEqual(
            json.dumps(MisbehavingTextSubtype(text)),
            json.dumps(text)
        )
        self.assertEqual(
            json.dumps([MisbehavingTextSubtype(text)]),
            json.dumps([text])
        )
        self.assertEqual(
            json.dumps({MisbehavingTextSubtype(text): 42}),
            json.dumps({text: 42})
        )

    def test_misbehaving_bytes_subtype(self):
        data = b("this is some data \xe2\x82\xac")
        self.assertEqual(
            json.dumps(MisbehavingBytesSubtype(data)),
            json.dumps(data)
        )
        self.assertEqual(
            json.dumps([MisbehavingBytesSubtype(data)]),
            json.dumps([data])
        )
        self.assertEqual(
            json.dumps({MisbehavingBytesSubtype(data): 42}),
            json.dumps({data: 42})
        )

    def test_bytes_toplevel(self):
        self.assertEqual(json.dumps(b('\xe2\x82\xac')), r'"\u20ac"')
        self.assertRaises(UnicodeDecodeError, json.dumps, b('\xa4'))
        self.assertEqual(json.dumps(b('\xa4'), encoding='iso-8859-1'),
                         r'"\u00a4"')
        self.assertEqual(json.dumps(b('\xa4'), encoding='iso-8859-15'),
                         r'"\u20ac"')
        if PY3:
            self.assertRaises(TypeError, json.dumps, b('\xe2\x82\xac'),
                              encoding=None)
            self.assertRaises(TypeError, json.dumps, b('\xa4'),
                              encoding=None)
            self.assertEqual(json.dumps(b('\xa4'), encoding=None,
                                        default=decode_iso_8859_15),
                            r'"\u20ac"')
        else:
            self.assertEqual(json.dumps(b('\xe2\x82\xac'), encoding=None),
                             r'"\u20ac"')
            self.assertRaises(UnicodeDecodeError, json.dumps, b('\xa4'),
                              encoding=None)
            self.assertRaises(UnicodeDecodeError, json.dumps, b('\xa4'),
                              encoding=None, default=decode_iso_8859_15)

    def test_bytes_nested(self):
        self.assertEqual(json.dumps([b('\xe2\x82\xac')]), r'["\u20ac"]')
        self.assertRaises(UnicodeDecodeError, json.dumps, [b('\xa4')])
        self.assertEqual(json.dumps([b('\xa4')], encoding='iso-8859-1'),
                         r'["\u00a4"]')
        self.assertEqual(json.dumps([b('\xa4')], encoding='iso-8859-15'),
                         r'["\u20ac"]')
        if PY3:
            self.assertRaises(TypeError, json.dumps, [b('\xe2\x82\xac')],
                              encoding=None)
            self.assertRaises(TypeError, json.dumps, [b('\xa4')],
                              encoding=None)
            self.assertEqual(json.dumps([b('\xa4')], encoding=None,
                                        default=decode_iso_8859_15),
                             r'["\u20ac"]')
        else:
            self.assertEqual(json.dumps([b('\xe2\x82\xac')], encoding=None),
                             r'["\u20ac"]')
            self.assertRaises(UnicodeDecodeError, json.dumps, [b('\xa4')],
                              encoding=None)
            self.assertRaises(UnicodeDecodeError, json.dumps, [b('\xa4')],
                              encoding=None, default=decode_iso_8859_15)

    def test_bytes_key(self):
        self.assertEqual(json.dumps({b('\xe2\x82\xac'): 42}), r'{"\u20ac": 42}')
        self.assertRaises(UnicodeDecodeError, json.dumps, {b('\xa4'): 42})
        self.assertEqual(json.dumps({b('\xa4'): 42}, encoding='iso-8859-1'),
                         r'{"\u00a4": 42}')
        self.assertEqual(json.dumps({b('\xa4'): 42}, encoding='iso-8859-15'),
                         r'{"\u20ac": 42}')
        if PY3:
            self.assertRaises(TypeError, json.dumps, {b('\xe2\x82\xac'): 42},
                              encoding=None)
            self.assertRaises(TypeError, json.dumps, {b('\xa4'): 42},
                              encoding=None)
            self.assertRaises(TypeError, json.dumps, {b('\xa4'): 42},
                              encoding=None, default=decode_iso_8859_15)
            self.assertEqual(json.dumps({b('\xa4'): 42}, encoding=None,
                                        skipkeys=True),
                             r'{}')
        else:
            self.assertEqual(json.dumps({b('\xe2\x82\xac'): 42}, encoding=None),
                             r'{"\u20ac": 42}')
            self.assertRaises(UnicodeDecodeError, json.dumps, {b('\xa4'): 42},
                              encoding=None)
            self.assertRaises(UnicodeDecodeError, json.dumps, {b('\xa4'): 42},
                              encoding=None, default=decode_iso_8859_15)
            self.assertRaises(UnicodeDecodeError, json.dumps, {b('\xa4'): 42},
                              encoding=None, skipkeys=True)
