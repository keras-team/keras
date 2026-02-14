from keras.src import testing


class TypesTest(testing.TestCase):
    def test_tensor_importable(self):
        from keras.src.types import Tensor

        self.assertIsNotNone(Tensor)

    def test_shape_importable(self):
        from keras.src.types import Shape

        self.assertIsNotNone(Shape)

    def test_dtype_importable(self):
        from keras.src.types import DType

        self.assertIsNotNone(DType)

    def test_api_export(self):
        import keras

        self.assertTrue(hasattr(keras, "types"))
        self.assertTrue(hasattr(keras.types, "Tensor"))
        self.assertTrue(hasattr(keras.types, "Shape"))
        self.assertTrue(hasattr(keras.types, "DType"))
