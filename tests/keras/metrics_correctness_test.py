"""Tests for Keras metrics correctness."""

import numpy as np

import keras
from keras import layers
from keras import losses
from keras import metrics
from keras import backend as K


def get_multi_io_model():
    inp_1 = layers.Input(shape=(1,), name='input_1')
    inp_2 = layers.Input(shape=(1,), name='input_2')
    dense = layers.Dense(3, kernel_initializer='ones', trainable=False)
    x_1 = dense(inp_1)
    x_2 = dense(inp_2)
    out_1 = layers.Dense(
        1, kernel_initializer='ones', name='output_1', trainable=False)(x_1)
    out_2 = layers.Dense(
        1, kernel_initializer='ones', name='output_2', trainable=False)(x_2)
    return keras.Model([inp_1, inp_2], [out_1, out_2])


def custom_generator_multi_io(sample_weights=None):
    batch_size = 2
    num_samples = 4
    inputs = np.asarray([[1.], [2.], [3.], [4.]])
    targets_1 = np.asarray([[2.], [4.], [6.], [8.]])
    targets_2 = np.asarray([[1.], [2.], [3.], [4.]])
    w1 = sample_weights[0] if sample_weights else None
    w2 = sample_weights[1] if sample_weights else None
    i = 0
    while True:
        batch_index = i * batch_size % num_samples
        i += 1
        start = batch_index
        end = start + batch_size
        x = [inputs[start:end], inputs[start:end]]
        y = [targets_1[start:end], targets_2[start:end]]
        if sample_weights:
            w = [
                None if w1 is None else w1[start:end],
                None if w2 is None else w2[start:end]
            ]
        else:
            w = None
        yield x, y, w


class TestMetricsCorrectnessMultiIO(object):

    def _get_compiled_multi_io_model(self):
        model = get_multi_io_model()
        model.compile(
            optimizer='rmsprop',
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError(name='mean_squared_error')],
            weighted_metrics=[
                metrics.MeanSquaredError(name='mean_squared_error_2')
            ])
        return model

    def setUp(self):
        self.x = np.asarray([[1.], [2.], [3.], [4.]])
        self.y1 = np.asarray([[2.], [4.], [6.], [8.]])
        self.y2 = np.asarray([[1.], [2.], [3.], [4.]])
        self.sample_weight_1 = np.asarray([2., 3., 4., 5.])
        self.sample_weight_2 = np.asarray([3.5, 2.5, 1.5, 0.5])
        self.class_weight_1 = {2: 2, 4: 3, 6: 4, 8: 5}
        self.class_weight_2 = {1: 3.5, 2: 2.5, 3: 1.5, 4: 0.5}

        # y_true_1 = [[2.], [4.], [6.], [8.]], y_pred = [[3.], [6.], [9.], [12.]]
        # y_true_2 = [[1.], [2.], [3.], [4.]], y_pred = [[3.], [6.], [9.], [12.]]

        # Weighted metric `output_1`:
        #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
        #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
        #         = 130
        #   Count = (2 + 3) + (4 + 5)
        #   Result = 9.2857141

        # Weighted metric `output_2`:
        #   Total = ((3 - 1)^2 * 3.5 + (6 - 2)^2 * 2.5) +
        #           ((9 - 3)^2 * 1.5 + (12 - 4)^2 * 0.5)
        #         = 140
        #   Count = (3.5 + 2.5) + (1.5 + 0.5)
        #   Result = 17.5

        # Loss `output_1` with weights:
        #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
        #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
        #         = 130
        #   Count = 2 + 2
        #   Result = 32.5

        # Loss `output_1` without weights/Metric `output_1`:
        #   Total = ((3 - 2)^2 + (6 - 4)^2) + ((9 - 6)^2 + (12 - 8)^2) = 30
        #   Count = 2 + 2
        #   Result = 7.5

        # Loss `output_2` with weights:
        #   Total = ((3 - 1)^2 * 3.5 + (6 - 2)^2 * 2.5) +
        #           ((9 - 3)^2 * 1.5 + (12 - 4)^2 * 0.5)
        #         = 140
        #   Count = 2 + 2
        #   Result = 35

        # Loss `output_2` without weights/Metric `output_2`:
        #   Total = ((3 - 1)^2 + (6 - 2)^2) + ((9 - 3)^2 + (12 - 4)^2) = 120
        #   Count = 2 + 2
        #   Result = 30

        # Total loss with weights = 32.5 + 35 = 67.5
        # Total loss without weights = 7.5 + 30 = 37.5

        self.expected_fit_result_with_weights = {
            'output_1_mean_squared_error': [7.5, 7.5],
            'output_2_mean_squared_error': [30, 30],
            'output_1_mean_squared_error_2': [9.286, 9.286],
            'output_2_mean_squared_error_2': [17.5, 17.5],
            'loss': [67.5, 67.5],
            'output_1_loss': [32.5, 32.5],
            'output_2_loss': [35, 35],
        }

        self.expected_fit_result_with_weights_output_2 = {
            'output_1_mean_squared_error': [7.5, 7.5],
            'output_2_mean_squared_error': [30, 30],
            'output_1_mean_squared_error_2': [7.5, 7.5],
            'output_2_mean_squared_error_2': [17.5, 17.5],
            'loss': [42.5, 42.5],
            'output_1_loss': [7.5, 7.5],
            'output_2_loss': [35, 35],
        }

        self.expected_fit_result = {
            'output_1_mean_squared_error': [7.5, 7.5],
            'output_2_mean_squared_error': [30, 30],
            'output_1_mean_squared_error_2': [7.5, 7.5],
            'output_2_mean_squared_error_2': [30, 30],
            'loss': [37.5, 37.5],
            'output_1_loss': [7.5, 7.5],
            'output_2_loss': [30, 30],
        }

        # In the order: 'loss', 'output_1_loss', 'output_2_loss',
        # 'output_1_mean_squared_error', 'output_1_mean_squared_error_2',
        # 'output_2_mean_squared_error', 'output_2_mean_squared_error_2'
        self.expected_batch_result_with_weights = [
            67.5, 32.5, 35, 7.5, 9.286, 30, 17.5
        ]
        self.expected_batch_result_with_weights_output_2 = [
            42.5, 7.5, 35, 7.5, 7.5, 30, 17.5
        ]
        self.expected_batch_result = [37.5, 7.5, 30, 7.5, 7.5, 30, 30]

    def test_fit(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        history = model.fit([self.x, self.x], [self.y1, self.y2],
                            batch_size=2,
                            epochs=2,
                            shuffle=False)
        for key, value in self.expected_fit_result.items():
            np.allclose(history.history[key], value, 1e-3)

    def test_fit_with_sample_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        history = model.fit([self.x, self.x], [self.y1, self.y2],
                            sample_weight={
                                'output_1': self.sample_weight_1,
                                'output_2': self.sample_weight_2},
                            batch_size=2,
                            epochs=2,
                            shuffle=False)
        for key, value in self.expected_fit_result_with_weights.items():
            np.allclose(history.history[key], value, 1e-3)

        # Set weights for one output (use batch size).
        history = model.fit([self.x, self.x], [self.y1, self.y2],
                            sample_weight={'output_2': self.sample_weight_2},
                            batch_size=2,
                            epochs=2,
                            shuffle=False)

        for key, value in self.expected_fit_result_with_weights_output_2.items():
            np.allclose(history.history[key], value, 1e-3)

    def test_fit_with_class_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        history = model.fit([self.x, self.x], [self.y1, self.y2],
                            class_weight={
                                'output_1': self.class_weight_1,
                                'output_2': self.class_weight_2},
                            batch_size=2,
                            epochs=2,
                            shuffle=False)
        for key, value in self.expected_fit_result_with_weights.items():
            np.allclose(history.history[key], value, 1e-3)

        # Set weights for one output.
        history = model.fit([self.x, self.x], [self.y1, self.y2],
                            class_weight={'output_2': self.class_weight_2},
                            batch_size=2,
                            epochs=2,
                            shuffle=False)

        for key, value in self.expected_fit_result_with_weights_output_2.items():
            np.allclose(history.history[key], value, 1e-3)

    def test_eval(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                     batch_size=2)
        np.allclose(eval_result, self.expected_batch_result, 1e-3)

    def test_eval_with_sample_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                     batch_size=2,
                                     sample_weight={
                                         'output_1': self.sample_weight_1,
                                         'output_2': self.sample_weight_2})
        np.allclose(eval_result, self.expected_batch_result_with_weights,
                    1e-3)

        # Set weights for one output.
        model = self._get_compiled_multi_io_model()
        eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                     batch_size=2,
                                     sample_weight={
                                         'output_2': self.sample_weight_2})
        np.allclose(eval_result,
                    self.expected_batch_result_with_weights_output_2, 1e-3)

        # Verify that metric value is same with arbitrary weights and batch size.
        x = np.random.random((50, 1))
        y = np.random.random((50, 1))
        w = np.random.random((50,))
        mse1 = model.evaluate([x, x], [y, y], sample_weight=[w, w], batch_size=5)[3]
        mse2 = model.evaluate([x, x], [y, y], sample_weight=[w, w],
                              batch_size=10)[3]
        np.allclose(mse1, mse2, 1e-3)

    def test_train_on_batch(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        result = model.train_on_batch([self.x, self.x], [self.y1, self.y2])
        np.allclose(result, self.expected_batch_result, 1e-3)

    def test_train_on_batch_with_sample_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                      sample_weight={
                                          'output_1': self.sample_weight_1,
                                          'output_2': self.sample_weight_2})
        np.allclose(result, self.expected_batch_result_with_weights, 1e-3)

        # Set weights for one output.
        result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                      sample_weight={
                                          'output_2': self.sample_weight_2})
        np.allclose(result, self.expected_batch_result_with_weights_output_2, 1e-3)

    def test_train_on_batch_with_class_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                      class_weight={
                                          'output_1': self.class_weight_1,
                                          'output_2': self.class_weight_2})
        np.allclose(result, self.expected_batch_result_with_weights, 1e-3)

        # Set weights for one output.
        result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                      class_weight={
                                          'output_2': self.class_weight_2})
        np.allclose(result,
                    self.expected_batch_result_with_weights_output_2, 1e-3)

    def test_test_on_batch(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        result = model.test_on_batch([self.x, self.x], [self.y1, self.y2])
        np.allclose(result, self.expected_batch_result, 1e-3)

    def test_test_on_batch_with_sample_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        result = model.test_on_batch([self.x, self.x], [self.y1, self.y2],
                                     sample_weight={
                                         'output_1': self.sample_weight_1,
                                         'output_2': self.sample_weight_2})
        np.allclose(result, self.expected_batch_result_with_weights, 1e-3)

        # Set weights for one output.
        result = model.test_on_batch([self.x, self.x], [self.y1, self.y2],
                                     sample_weight={
                                         'output_2': self.sample_weight_2})
        np.allclose(result,
                    self.expected_batch_result_with_weights_output_2, 1e-3)

    def test_fit_generator(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        history = model.fit_generator(
            custom_generator_multi_io(), steps_per_epoch=2, epochs=2)
        for key, value in self.expected_fit_result.items():
            np.allclose(history.history[key], value, 1e-3)

    def test_fit_generator_with_sample_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        history = model.fit_generator(
            custom_generator_multi_io(
                sample_weights=[self.sample_weight_1, self.sample_weight_2]),
            steps_per_epoch=2,
            epochs=2)
        for key, value in self.expected_fit_result_with_weights.items():
            np.allclose(history.history[key], value, 1e-3)

        # Set weights for one output.
        history = model.fit_generator(
            custom_generator_multi_io(sample_weights=[None, self.sample_weight_2]),
            steps_per_epoch=2,
            epochs=2)
        for key, value in self.expected_fit_result_with_weights_output_2.items():
            np.allclose(history.history[key], value, 1e-3)

    def test_fit_generator_with_class_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        history = model.fit_generator(
            custom_generator_multi_io(),
            class_weight={
                'output_1': self.class_weight_1,
                'output_2': self.class_weight_2,
            },
            steps_per_epoch=2,
            epochs=2)
        for key, value in self.expected_fit_result_with_weights.items():
            np.allclose(history.history[key], value, 1e-3)

        # Set weights for one output.
        history = model.fit_generator(
            custom_generator_multi_io(),
            class_weight={'output_2': self.class_weight_2},
            steps_per_epoch=2,
            epochs=2)
        for key, value in self.expected_fit_result_with_weights_output_2.items():
            np.allclose(history.history[key], value, 1e-3)

    def test_eval_generator(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        eval_result = model.evaluate_generator(custom_generator_multi_io(), steps=2)
        np.allclose(eval_result, self.expected_batch_result, 1e-3)

    def test_eval_generator_with_sample_weight(self):
        self.setUp()
        model = self._get_compiled_multi_io_model()
        eval_result = model.evaluate_generator(
            custom_generator_multi_io(
                sample_weights=[self.sample_weight_1, self.sample_weight_2]),
            steps=2)
        np.allclose(eval_result, self.expected_batch_result_with_weights, 1e-3)

        # Set weights for one output.
        eval_result = model.evaluate_generator(
            custom_generator_multi_io(sample_weights=[None, self.sample_weight_2]),
            steps=2)
        np.allclose(eval_result,
                    self.expected_batch_result_with_weights_output_2, 1e-3)
