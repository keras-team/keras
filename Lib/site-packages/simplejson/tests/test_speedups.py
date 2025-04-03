from __future__ import with_statement

import sys
import unittest
from unittest import TestCase

import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b


def has_speedups():
    return encoder.c_make_encoder is not None


def skip_if_speedups_missing(func):
    def wrapper(*args, **kwargs):
        if not has_speedups():
            if hasattr(unittest, 'SkipTest'):
                raise unittest.SkipTest("C Extension not available")
            else:
                sys.stdout.write("C Extension not available")
                return
        return func(*args, **kwargs)

    return wrapper


class BadBool:
    def __bool__(self):
        1/0
    __nonzero__ = __bool__


class TestDecode(TestCase):
    @skip_if_speedups_missing
    def test_make_scanner(self):
        self.assertRaises(AttributeError, scanner.c_make_scanner, 1)

    @skip_if_speedups_missing
    def test_bad_bool_args(self):
        def test(value):
            decoder.JSONDecoder(strict=BadBool()).decode(value)
        self.assertRaises(ZeroDivisionError, test, '""')
        self.assertRaises(ZeroDivisionError, test, '{}')
        if not PY3:
            self.assertRaises(ZeroDivisionError, test, u'""')
            self.assertRaises(ZeroDivisionError, test, u'{}')

class TestEncode(TestCase):
    @skip_if_speedups_missing
    def test_make_encoder(self):
        self.assertRaises(
            TypeError,
            encoder.c_make_encoder,
            None,
            ("\xCD\x7D\x3D\x4E\x12\x4C\xF9\x79\xD7"
             "\x52\xBA\x82\xF2\x27\x4A\x7D\xA0\xCA\x75"),
            None
        )

    @skip_if_speedups_missing
    def test_bad_str_encoder(self):
        # Issue #31505: There shouldn't be an assertion failure in case
        # c_make_encoder() receives a bad encoder() argument.
        import decimal
        def bad_encoder1(*args):
            return None
        enc = encoder.c_make_encoder(
                None, lambda obj: str(obj),
                bad_encoder1, None, ': ', ', ',
                False, False, False, {}, False, False, False,
                None, None, 'utf-8', False, False, decimal.Decimal, False)
        self.assertRaises(TypeError, enc, 'spam', 4)
        self.assertRaises(TypeError, enc, {'spam': 42}, 4)

        def bad_encoder2(*args):
            1/0
        enc = encoder.c_make_encoder(
                None, lambda obj: str(obj),
                bad_encoder2, None, ': ', ', ',
                False, False, False, {}, False, False, False,
                None, None, 'utf-8', False, False, decimal.Decimal, False)
        self.assertRaises(ZeroDivisionError, enc, 'spam', 4)

    @skip_if_speedups_missing
    def test_bad_bool_args(self):
        def test(name):
            encoder.JSONEncoder(**{name: BadBool()}).encode({})
        self.assertRaises(ZeroDivisionError, test, 'skipkeys')
        self.assertRaises(ZeroDivisionError, test, 'ensure_ascii')
        self.assertRaises(ZeroDivisionError, test, 'check_circular')
        self.assertRaises(ZeroDivisionError, test, 'allow_nan')
        self.assertRaises(ZeroDivisionError, test, 'sort_keys')
        self.assertRaises(ZeroDivisionError, test, 'use_decimal')
        self.assertRaises(ZeroDivisionError, test, 'namedtuple_as_object')
        self.assertRaises(ZeroDivisionError, test, 'tuple_as_array')
        self.assertRaises(ZeroDivisionError, test, 'bigint_as_string')
        self.assertRaises(ZeroDivisionError, test, 'for_json')
        self.assertRaises(ZeroDivisionError, test, 'ignore_nan')
        self.assertRaises(ZeroDivisionError, test, 'iterable_as_array')

    @skip_if_speedups_missing
    def test_int_as_string_bitcount_overflow(self):
        long_count = long_type(2)**32+31
        def test():
            encoder.JSONEncoder(int_as_string_bitcount=long_count).encode(0)
        self.assertRaises((TypeError, OverflowError), test)

    if PY3:
        @skip_if_speedups_missing
        def test_bad_encoding(self):
            with self.assertRaises(UnicodeEncodeError):
                encoder.JSONEncoder(encoding='\udcff').encode({b('key'): 123})
