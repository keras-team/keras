import numpy as np

from keras.src import testing
from keras.src.saving import saving_lib_test


class VariableMappingTest(testing.TestCase):
    def test_basics(self):
        model = saving_lib_test._get_basic_functional_model()
        model.optimizer.build(model.trainable_variables)
        variable_map = model._get_variable_map()

        self.assertIn("first_dense/kernel", variable_map)
        self.assertIn("second_dense/bias", variable_map)
        self.assertIn("adam/learning_rate", variable_map)

        model = saving_lib_test._get_basic_sequential_model()
        model.build((None, 1))
        model.optimizer.build(model.trainable_variables)
        variable_map = model._get_variable_map()
        self.assertIn("sequential/dense_1/bias", variable_map)
        self.assertIn("adam/learning_rate", variable_map)

        model = saving_lib_test._get_subclassed_model()
        model(np.ones((1, 1)))
        model.optimizer.build(model.trainable_variables)
        variable_map = model._get_variable_map()
        self.assertIn("custom_model_x/my_dense_1/dense/kernel", variable_map)
        self.assertIn("custom_model_x/my_dense_1/my_dict_weight", variable_map)
        self.assertIn(
            "custom_model_x/my_dense_1/my_additional_weight", variable_map
        )
        self.assertIn("adam/learning_rate", variable_map)
