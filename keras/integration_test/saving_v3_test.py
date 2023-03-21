"""Test Model.fit across a diverse range of models."""

import os

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.integration_test.models import bert
from keras.integration_test.models import dcgan
from keras.integration_test.models import edge_case_model
from keras.integration_test.models import input_spec
from keras.integration_test.models import low_level_model
from keras.integration_test.models import mini_unet
from keras.integration_test.models import mini_xception
from keras.integration_test.models import retinanet
from keras.integration_test.models import structured_data_classification
from keras.integration_test.models import text_classification
from keras.integration_test.models import timeseries_forecasting
from keras.integration_test.models import vae
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


def get_dataset(data_specs, batch_size):
    values = tf.nest.map_structure(input_spec.spec_to_value, data_specs)
    dataset = (
        tf.data.Dataset.from_tensor_slices(values)
        .prefetch(batch_size * 2)
        .batch(batch_size)
    )
    return dataset


@test_utils.run_v2_only
class SavingV3Test(test_combinations.TestCase):
    @parameterized.named_parameters(
        ("bert", bert),
        ("edge_case_model", edge_case_model),
        # ("efficientnet_v2", efficientnet_v2),  # Too expensive to run on CI
        ("low_level_model", low_level_model),
        ("mini_unet", mini_unet),
        ("mini_xception", mini_xception),
        ("retinanet", retinanet),
        ("structured_data_classification", structured_data_classification),
        ("text_classification", text_classification),
        ("timeseries_forecasting", timeseries_forecasting),
    )
    def test_saving_v3(self, module):
        batch_size = 2
        data_specs = module.get_data_spec(batch_size * 2)
        dataset = get_dataset(data_specs, batch_size)
        for batch in dataset.take(1):
            pass
        if isinstance(batch, tuple):
            batch = batch[0]

        model = module.get_model(
            build=True,
            compile=True,
            jit_compile=False,
            include_preprocessing=True,
        )
        model.fit(dataset, epochs=1, steps_per_epoch=1)
        temp_filepath = os.path.join(
            self.get_temp_dir(), f"{module.__name__}.keras"
        )
        model.save(temp_filepath, save_format="keras_v3")
        with tf.keras.utils.custom_object_scope(module.get_custom_objects()):
            new_model = tf.keras.models.load_model(temp_filepath)

        # Test model weights
        self.assertIs(new_model.__class__, model.__class__)
        self.assertEqual(len(model.get_weights()), len(new_model.get_weights()))
        for w1, w2 in zip(model.get_weights(), new_model.get_weights()):
            if w1.dtype == "object":
                self.assertEqual(str(w1), str(w2))
            else:
                self.assertAllClose(w1, w2, atol=1e-6)

        # Test forward pass
        self.assertAllClose(new_model(batch), model(batch), atol=1e-6)

        # Test optimizer state
        if hasattr(model, "optimizer"):
            self.assertEqual(
                len(model.optimizer.variables()),
                len(new_model.optimizer.variables()),
            )
            for v1, v2 in zip(
                model.optimizer.variables(), new_model.optimizer.variables()
            ):
                self.assertAllClose(v1.numpy(), v2.numpy(), atol=1e-6)

        # Test training still works
        new_model.fit(dataset, epochs=1, steps_per_epoch=1)

    @parameterized.named_parameters(("dcgan", dcgan), ("vae", vae))
    def test_saving_v3_no_call(self, module):
        batch_size = 2
        data_specs = module.get_data_spec(batch_size * 2)
        dataset = get_dataset(data_specs, batch_size)

        model = module.get_model(
            build=True,
            compile=True,
            jit_compile=False,
            include_preprocessing=True,
        )
        temp_filepath = os.path.join(
            self.get_temp_dir(), f"{module.__name__}.keras"
        )
        model.save(temp_filepath, save_format="keras_v3")
        with tf.keras.utils.custom_object_scope(module.get_custom_objects()):
            new_model = tf.keras.models.load_model(temp_filepath)

        # Test model weights
        self.assertIs(new_model.__class__, model.__class__)
        self.assertEqual(len(model.get_weights()), len(new_model.get_weights()))
        for w1, w2 in zip(model.get_weights(), new_model.get_weights()):
            if w1.dtype == "object":
                self.assertEqual(str(w1), str(w2))
            else:
                self.assertAllClose(w1, w2, atol=1e-6)

        # Test training still works
        new_model.fit(dataset, epochs=1, steps_per_epoch=1)


if __name__ == "__main__":
    tf.test.main()
