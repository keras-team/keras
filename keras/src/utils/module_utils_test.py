"""Tests for keras.src.utils.module_utils (LazyModule, OrbaxLazyModule)."""

from keras.src import testing
from keras.src.utils.module_utils import LazyModule


class LazyModuleAvailableTest(testing.TestCase):
    def test_available_for_existing_module(self):
        m = LazyModule("json")
        self.assertTrue(m.available)

    def test_not_available_for_missing_module(self):
        m = LazyModule("nonexistent_module_xyz_12345")
        self.assertFalse(m.available)

    def test_available_caches_result(self):
        m = LazyModule("json")
        _ = m.available
        self.assertTrue(m.available)
        self.assertIsNotNone(m.module)


class LazyModuleInitializeTest(testing.TestCase):
    def test_initialize_loads_module(self):
        m = LazyModule("os")
        m.initialize()
        self.assertIsNotNone(m.module)

    def test_initialize_missing_raises(self):
        m = LazyModule("nonexistent_xyz")
        with self.assertRaises(ImportError):
            m.initialize()

    def test_custom_error_message(self):
        msg = "Custom error: install xyz"
        m = LazyModule("nonexistent_xyz", import_error_msg=msg)
        with self.assertRaisesRegex(ImportError, "Custom error"):
            m.initialize()

    def test_default_error_message_includes_pip_name(self):
        m = LazyModule("nonexistent_xyz", pip_name="xyz-package")
        with self.assertRaisesRegex(ImportError, "xyz-package"):
            m.initialize()


class LazyModuleGetAttrTest(testing.TestCase):
    def test_getattr_delegates_to_module(self):
        m = LazyModule("os.path")
        self.assertTrue(callable(m.join))

    def test_getattr_auto_initializes(self):
        m = LazyModule("json")
        # Module is not loaded yet
        self.assertIsNone(m.module)
        # Accessing an attribute should trigger initialization
        _ = m.dumps
        self.assertIsNotNone(m.module)

    def test_getattr_missing_attr_raises(self):
        m = LazyModule("json")
        with self.assertRaises(AttributeError):
            _ = m.nonexistent_attr_xyz

    def test_api_export_path_raises(self):
        """_api_export_path is intercepted specially."""
        m = LazyModule("json")
        with self.assertRaises(AttributeError):
            _ = m._api_export_path


class LazyModuleReprTest(testing.TestCase):
    def test_repr(self):
        m = LazyModule("json")
        self.assertEqual(repr(m), "LazyModule(json)")


class LazyModuleInitAttributesTest(testing.TestCase):
    def test_name(self):
        m = LazyModule("json")
        self.assertEqual(m.name, "json")

    def test_pip_name_default(self):
        m = LazyModule("json")
        self.assertEqual(m.pip_name, "json")

    def test_pip_name_custom(self):
        m = LazyModule("json", pip_name="python-json")
        self.assertEqual(m.pip_name, "python-json")


if __name__ == "__main__":
    testing.run_tests()
