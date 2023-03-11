"""Test Model inference and save/load with an ExtensionType."""

import typing

import tensorflow.compat.v2 as tf

import keras
from keras.engine.input_layer import Input
from keras.engine.training import Model
from keras.saving.saving_api import load_model
from keras.testing_infra import test_utils


class MaskedTensor(tf.experimental.BatchableExtensionType):
    """Example subclass of ExtensionType, used for testing.

    This version adds Keras required properties to MaskedTensor and its Spec
    class, to test Keras integration.
    """

    __name__ = "tf.test.MaskedTensor.Spec"

    values: typing.Union[tf.Tensor, tf.RaggedTensor]
    mask: typing.Union[tf.Tensor, tf.RaggedTensor]

    def __init__(self, values, mask):
        if isinstance(values, tf.RaggedTensor):
            assert isinstance(mask, tf.RaggedTensor)
            assert mask.dtype == tf.dtypes.bool
        else:
            values = tf.convert_to_tensor(values)
            mask = tf.convert_to_tensor(mask, tf.dtypes.bool)
        self.values = values
        self.mask = mask

    # Required by assert_input_compatibility in keras/engine/input_spec.py
    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    class Spec:

        # Required by KerasTensor.shape in keras/engine/keras_tensor.py
        @property
        def shape(self):
            return self.values._shape


class ExtensionTypeTest(tf.test.TestCase):
    @test_utils.run_v2_only
    def testKerasModel(self):
        mt_spec = MaskedTensor.Spec(
            tf.TensorSpec(shape=[None, 1], dtype=tf.dtypes.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.dtypes.bool),
        )
        model_input = Input(type_spec=mt_spec)
        model_output = keras.layers.Lambda(
            lambda x: tf.identity(x, name="output")
        )(model_input)
        model = Model(inputs=model_input, outputs=model_output)
        mt = MaskedTensor([[1], [2], [3]], [[True], [False], [True]])
        self.assertEqual(model(mt), mt)
        ds = tf.data.Dataset.from_tensors(mt)
        self.assertEqual(model.predict(ds), mt)

        with self.subTest("keras save"):
            path = self.create_tempdir().full_path
            model.save(path)
            loaded_model = load_model(path)
            self.assertEqual(loaded_model.input.type_spec, mt_spec)
            self.assertEqual(loaded_model(mt), mt)

            loaded_fn = tf.saved_model.load(path)
            self.assertEqual(loaded_fn(mt), mt)
            with self.assertRaisesRegex(
                ValueError,
                "Could not find matching concrete function to call "
                "loaded from the SavedModel",
            ):
                loaded_fn(MaskedTensor([1, 2, 3], [True, False, True]))

            # The serving_fn use flatten signature
            serving_fn = loaded_fn.signatures["serving_default"]
            self.assertEqual(
                serving_fn(args_0=mt.values, args_0_1=mt.mask)["lambda"], mt
            )


if __name__ == "__main__":
    tf.test.main()
