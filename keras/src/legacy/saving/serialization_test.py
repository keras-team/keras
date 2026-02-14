from keras.src import testing
from keras.src.legacy.saving import serialization


def custom_fn(x):
    """A simple custom function."""
    return x**2


class DummyLayer:
    """A dummy layer class for testing."""

    pass


class SerializationTest(testing.TestCase):
    def test_deserialize_function_from_module_objects(self):
        module_objects = {"custom_fn": custom_fn, "DummyLayer": DummyLayer}

        config = {
            "class_name": "DummyLayer",
            "config": {"name": "test", "callback": "custom_fn"},
        }

        cls, cls_config = (
            serialization.class_and_config_for_serialized_keras_object(
                config,
                module_objects=module_objects,
                custom_objects=None,
                printable_module_name="dummy_layer",
            )
        )

        self.assertTrue(callable(cls_config["callback"]))
        self.assertIs(cls_config["callback"], custom_fn)
        self.assertIs(cls, DummyLayer)

    def test_deserialize_function_not_in_module_objects(self):
        module_objects = {"DummyLayer": DummyLayer}

        config = {
            "class_name": "DummyLayer",
            "config": {"name": "test", "callback": "custom_fn"},
        }

        cls, cls_config = (
            serialization.class_and_config_for_serialized_keras_object(
                config,
                module_objects=module_objects,
                custom_objects=None,
                printable_module_name="dummy_layer",
            )
        )

        self.assertIsInstance(cls_config["callback"], str)
        self.assertEqual(cls_config["callback"], "custom_fn")
