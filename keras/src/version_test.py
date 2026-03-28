"""Tests for keras.src.version."""

from keras.src import testing
from keras.src.version import __version__
from keras.src.version import version


class VersionTest(testing.TestCase):
    def test_version_function_returns_string(self):
        v = version()
        self.assertIsInstance(v, str)

    def test_version_matches_module_var(self):
        self.assertEqual(version(), __version__)

    def test_version_format(self):
        """Version should follow semver-like pattern X.Y.Z."""
        v = version()
        parts = v.split(".")
        self.assertGreaterEqual(len(parts), 2)
        # Major and minor should be numeric
        self.assertTrue(parts[0].isdigit())
        self.assertTrue(parts[1].isdigit())

    def test_version_not_empty(self):
        self.assertTrue(len(version()) > 0)


if __name__ == "__main__":
    testing.run_tests()
