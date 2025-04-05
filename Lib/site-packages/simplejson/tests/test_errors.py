import sys, pickle
from unittest import TestCase

import simplejson as json
from simplejson.compat import text_type, b

class TestErrors(TestCase):
    def test_string_keys_error(self):
        data = [{'a': 'A', 'b': (2, 4), 'c': 3.0, ('d',): 'D tuple'}]
        try:
            json.dumps(data)
        except TypeError:
            err = sys.exc_info()[1]
        else:
            self.fail('Expected TypeError')
        self.assertEqual(str(err),
                'keys must be str, int, float, bool or None, not tuple')

    def test_not_serializable(self):
        try:
            json.dumps(json)
        except TypeError:
            err = sys.exc_info()[1]
        else:
            self.fail('Expected TypeError')
        self.assertEqual(str(err),
                'Object of type module is not JSON serializable')

    def test_decode_error(self):
        err = None
        try:
            json.loads('{}\na\nb')
        except json.JSONDecodeError:
            err = sys.exc_info()[1]
        else:
            self.fail('Expected JSONDecodeError')
        self.assertEqual(err.lineno, 2)
        self.assertEqual(err.colno, 1)
        self.assertEqual(err.endlineno, 3)
        self.assertEqual(err.endcolno, 2)

    def test_scan_error(self):
        err = None
        for t in (text_type, b):
            try:
                json.loads(t('{"asdf": "'))
            except json.JSONDecodeError:
                err = sys.exc_info()[1]
            else:
                self.fail('Expected JSONDecodeError')
            self.assertEqual(err.lineno, 1)
            self.assertEqual(err.colno, 10)

    def test_error_is_pickable(self):
        err = None
        try:
            json.loads('{}\na\nb')
        except json.JSONDecodeError:
            err = sys.exc_info()[1]
        else:
            self.fail('Expected JSONDecodeError')
        s = pickle.dumps(err)
        e = pickle.loads(s)

        self.assertEqual(err.msg, e.msg)
        self.assertEqual(err.doc, e.doc)
        self.assertEqual(err.pos, e.pos)
        self.assertEqual(err.end, e.end)
