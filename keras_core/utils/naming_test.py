from keras_core.testing import test_case
from keras_core.utils import naming


class NamingUtilsTest(test_case.TestCase):
    def test_auto_name(self):
        self.assertEqual(naming.auto_name("unique_name"), "unique_name")
        self.assertEqual(naming.auto_name("unique_name"), "unique_name_1")
        self.assertEqual(naming.auto_name("unique_name"), "unique_name_2")

    def test_get_uid(self):
        self.assertEqual(naming.get_uid("very_unique_name"), 1)
        self.assertEqual(naming.get_uid("very_unique_name"), 2)
        self.assertEqual(naming.get_uid("very_unique_name"), 3)
