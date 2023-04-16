import numpy as np
from keras_core import backend
from keras_core import testing
from keras_core import metrics as metrics_module
from keras_core.trainers.compile_utils import CompileMetrics

class TestCompileMetrics(testing.TestCase):
    
    def test_single_output_case(self):
        compile_metrics = CompileMetrics(
            metrics=[metrics_module.MeanSquareError()],
            weighted_metrics=[metrics_module.MeanSquareError()],
        )
        # Test symbolic build
        y_true, y_pred = backend.KerasTensor((3, 4)), backend.KerasTensor((3, 4))
        compile_metrics.build(y_true, y_pred)
        # Test eager build
        y_true, y_pred = np.ones((3, 4)), np.ones((3, 4))
        compile_metrics.build(y_true, y_pred)

        # Test update / result / reset flow
        sample_weight = np.array([1, 0, 1])
        compile_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)
        y_pred += 1
        compile_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)
        result = compile_metrics.result()
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 2)
        self.assertAllClose(result["mean_square_error"], 0.5)
        self.assertAllClose(result["weighted_mean_square_error"], 0.5)


    # def test_metric_identifier_conversion(self):
    #     compile_metrics = CompileMetrics(
    #         metrics=[
            
    #         ],
    #     )
