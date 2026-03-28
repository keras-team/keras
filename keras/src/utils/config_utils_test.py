"""Tests for keras.src.utils.config (Config class)."""

import json

from keras.src import testing
from keras.src.utils.config import Config


class ConfigInitTest(testing.TestCase):
    def test_empty_config(self):
        c = Config()
        self.assertEqual(len(c), 0)
        self.assertEqual(list(c.keys()), [])

    def test_init_with_kwargs(self):
        c = Config(lr=0.1, momentum=0.9)
        self.assertEqual(c.lr, 0.1)
        self.assertEqual(c.momentum, 0.9)
        self.assertEqual(len(c), 2)

    def test_init_not_frozen(self):
        c = Config(a=1)
        self.assertFalse(c.frozen)


class ConfigAttributeAccessTest(testing.TestCase):
    def test_getattr(self):
        c = Config(x=42)
        self.assertEqual(c.x, 42)

    def test_setattr(self):
        c = Config()
        c.foo = "bar"
        self.assertEqual(c.foo, "bar")

    def test_getattr_unknown_raises(self):
        c = Config(learning_rate=0.01)
        with self.assertRaisesRegex(AttributeError, "Unknown attribute"):
            _ = c.nonexistent

    def test_getattr_suggests_close_match(self):
        c = Config(learning_rate=0.01)
        with self.assertRaisesRegex(
            AttributeError, "Did you mean 'learning_rate'"
        ):
            _ = c.learning_rat

    def test_overwrite_existing_attr(self):
        c = Config(x=1)
        c.x = 2
        self.assertEqual(c.x, 2)


class ConfigDictAccessTest(testing.TestCase):
    def test_getitem(self):
        c = Config(a=10)
        self.assertEqual(c["a"], 10)

    def test_setitem(self):
        c = Config()
        c["b"] = 20
        self.assertEqual(c["b"], 20)

    def test_getitem_missing_raises(self):
        c = Config()
        with self.assertRaises(KeyError):
            _ = c["missing"]

    def test_delitem(self):
        c = Config(a=1, b=2)
        del c["a"]
        self.assertNotIn("a", c)
        self.assertEqual(len(c), 1)

    def test_contains(self):
        c = Config(x=1)
        self.assertIn("x", c)
        self.assertNotIn("y", c)

    def test_keys(self):
        c = Config(a=1, b=2)
        self.assertEqual(set(c.keys()), {"a", "b"})

    def test_values(self):
        c = Config(a=1, b=2)
        self.assertEqual(set(c.values()), {1, 2})

    def test_items(self):
        c = Config(a=1, b=2)
        self.assertEqual(set(c.items()), {("a", 1), ("b", 2)})

    def test_get_existing(self):
        c = Config(x=5)
        self.assertEqual(c.get("x"), 5)

    def test_get_missing_default(self):
        c = Config()
        self.assertIsNone(c.get("missing"))
        self.assertEqual(c.get("missing", 42), 42)

    def test_pop(self):
        c = Config(a=1, b=2)
        val = c.pop("a")
        self.assertEqual(val, 1)
        self.assertNotIn("a", c)

    def test_pop_missing_with_default(self):
        c = Config()
        self.assertEqual(c.pop("missing", 99), 99)

    def test_update_with_dict(self):
        c = Config(a=1)
        c.update({"b": 2, "c": 3})
        self.assertEqual(c["b"], 2)
        self.assertEqual(c["c"], 3)

    def test_update_with_kwargs(self):
        c = Config(a=1)
        c.update(b=2)
        self.assertEqual(c["b"], 2)


class ConfigIterationTest(testing.TestCase):
    def test_iter_sorted(self):
        c = Config(c=3, a=1, b=2)
        self.assertEqual(list(c), ["a", "b", "c"])

    def test_len(self):
        c = Config(a=1, b=2, c=3)
        self.assertEqual(len(c), 3)


class ConfigFreezeTest(testing.TestCase):
    def test_freeze_prevents_setattr(self):
        c = Config(a=1)
        c.freeze()
        self.assertTrue(c.frozen)
        with self.assertRaisesRegex(ValueError, "frozen"):
            c.b = 2

    def test_freeze_prevents_setitem(self):
        c = Config(a=1)
        c.freeze()
        with self.assertRaisesRegex(ValueError, "frozen"):
            c["b"] = 2

    def test_freeze_prevents_delitem(self):
        c = Config(a=1)
        c.freeze()
        with self.assertRaisesRegex(ValueError, "frozen"):
            del c["a"]

    def test_freeze_prevents_pop(self):
        c = Config(a=1)
        c.freeze()
        with self.assertRaisesRegex(ValueError, "frozen"):
            c.pop("a")

    def test_freeze_prevents_update(self):
        c = Config(a=1)
        c.freeze()
        with self.assertRaisesRegex(ValueError, "frozen"):
            c.update(b=2)

    def test_unfreeze(self):
        c = Config(a=1)
        c.freeze()
        c.unfreeze()
        self.assertFalse(c.frozen)
        c.b = 2
        self.assertEqual(c.b, 2)

    def test_freeze_allows_read(self):
        c = Config(a=1)
        c.freeze()
        self.assertEqual(c.a, 1)
        self.assertEqual(c["a"], 1)
        self.assertEqual(list(c.keys()), ["a"])


class ConfigSerializationTest(testing.TestCase):
    def test_to_json(self):
        c = Config(x=1, y="hello")
        j = c.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["x"], 1)
        self.assertEqual(parsed["y"], "hello")

    def test_as_dict(self):
        c = Config(a=1, b=2)
        d = c.as_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d, {"a": 1, "b": 2})

    def test_as_dict_is_copy(self):
        c = Config(a=1)
        d = c.as_dict()
        d["a"] = 999
        self.assertEqual(c.a, 1)  # Original unchanged


class ConfigReprTest(testing.TestCase):
    def test_repr(self):
        c = Config(x=1)
        r = repr(c)
        self.assertIn("Config", r)
        self.assertIn("x", r)


class ConfigEdgeCasesTest(testing.TestCase):
    def test_none_value(self):
        c = Config(x=None)
        self.assertIsNone(c.x)
        self.assertIn("x", c)

    def test_nested_dict_value(self):
        c = Config(opts={"lr": 0.01, "wd": 0.001})
        self.assertEqual(c.opts["lr"], 0.01)

    def test_list_value(self):
        c = Config(layers=[64, 128, 256])
        self.assertEqual(c.layers, [64, 128, 256])

    def test_bool_value(self):
        c = Config(debug=True)
        self.assertTrue(c.debug)

    def test_empty_string_key(self):
        c = Config()
        c[""] = "empty"
        self.assertEqual(c[""], "empty")

    def test_numeric_like_string_key(self):
        c = Config()
        c["123"] = "numeric"
        self.assertEqual(c["123"], "numeric")


if __name__ == "__main__":
    testing.run_tests()
