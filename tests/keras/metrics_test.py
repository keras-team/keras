import pytest
import numpy as np

import keras
from keras import metrics
from keras import backend as K
from keras.utils.test_utils import keras_test

all_metrics = [
    metrics.binary_accuracy,
    metrics.categorical_accuracy,
    metrics.mean_squared_error,
    metrics.mean_absolute_error,
    metrics.mean_absolute_percentage_error,
    metrics.mean_squared_logarithmic_error,
    metrics.squared_hinge,
    metrics.hinge,
    metrics.categorical_crossentropy,
    metrics.binary_crossentropy,
    metrics.poisson,
    metrics.cosine_proximity,
    metrics.logcosh,
]

all_sparse_metrics = [
    metrics.sparse_categorical_accuracy,
    metrics.sparse_categorical_crossentropy,
]


@keras_test
def test_metrics():
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    for metric in all_metrics:
        output = metric(y_a, y_b)
        print(metric.__name__)
        assert K.eval(output).shape == (6,)


@keras_test
def test_sparse_metrics():
    for metric in all_sparse_metrics:
        y_a = K.variable(np.random.randint(0, 7, (6,)), dtype=K.floatx())
        y_b = K.variable(np.random.random((6, 7)), dtype=K.floatx())
        assert K.eval(metric(y_a, y_b)).shape == (6,)


def test_serialize():
    '''This is a mock 'round trip' of serialize and deserialize.
    '''

    class MockMetric:
        def __init__(self):
            self.__name__ = "mock_metric"

    mock = MockMetric()
    found = metrics.serialize(mock)
    assert found == "mock_metric"

    found = metrics.deserialize('mock_metric',
                                custom_objects={'mock_metric': True})
    assert found is True


def test_invalid_get():

    with pytest.raises(ValueError):
        metrics.get(5)


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='CNTK backend does not support top_k yet')
@keras_test
def test_top_k_categorical_accuracy():
    y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
    y_true = K.variable(np.array([[0, 1, 0], [1, 0, 0]]))
    success_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=3))
    assert success_result == 1
    partial_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=2))
    assert partial_result == 0.5
    failure_result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred,
                                                               k=1))
    assert failure_result == 0


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='CNTK backend does not support top_k yet')
@keras_test
def test_sparse_top_k_categorical_accuracy():
    y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
    y_true = K.variable(np.array([[1], [0]]))
    success_result = K.eval(
        metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3))

    assert success_result == 1
    partial_result = K.eval(
        metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=2))

    assert partial_result == 0.5
    failure_result = K.eval(
        metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1))

    assert failure_result == 0


@keras_test
@pytest.mark.parametrize('metrics_mode', ['list', 'dict'])
def test_stateful_metrics(metrics_mode):
    np.random.seed(1334)

    class BinaryTruePositives(keras.layers.Layer):
        """Stateful Metric to count the total true positives over all batches.

        Assumes predictions and targets of shape `(samples, 1)`.

        # Arguments
            name: String, name for the metric.
        """

        def __init__(self, name='true_positives', **kwargs):
            super(BinaryTruePositives, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.true_positives = K.variable(value=0, dtype='int32')

        def reset_states(self):
            K.set_value(self.true_positives, 0)

        def __call__(self, y_true, y_pred):
            """Computes the number of true positives in a batch.

            # Arguments
                y_true: Tensor, batch_wise labels
                y_pred: Tensor, batch_wise predictions

            # Returns
                The total number of true positives seen this epoch at the
                    completion of the batch.
            """
            y_true = K.cast(y_true, 'int32')
            y_pred = K.cast(K.round(y_pred), 'int32')
            correct_preds = K.cast(K.equal(y_pred, y_true), 'int32')
            true_pos = K.cast(K.sum(correct_preds * y_true), 'int32')
            current_true_pos = self.true_positives * 1
            self.add_update(K.update_add(self.true_positives,
                                         true_pos),
                            inputs=[y_true, y_pred])
            return current_true_pos + true_pos

    metric_fn = BinaryTruePositives()
    config = metrics.serialize(metric_fn)
    metric_fn = metrics.deserialize(
        config, custom_objects={'BinaryTruePositives': BinaryTruePositives})

    # Test on simple model
    inputs = keras.Input(shape=(2,))
    outputs = keras.layers.Dense(1, activation='sigmoid', name='out')(inputs)
    model = keras.Model(inputs, outputs)

    if metrics_mode == 'list':
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['acc', metric_fn])
    elif metrics_mode == 'dict':
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics={'out': ['acc', metric_fn]})

    samples = 1000
    x = np.random.random((samples, 2))
    y = np.random.randint(2, size=(samples, 1))

    val_samples = 10
    val_x = np.random.random((val_samples, 2))
    val_y = np.random.randint(2, size=(val_samples, 1))

    # Test fit and evaluate
    history = model.fit(x, y, validation_data=(val_x, val_y), epochs=1, batch_size=10)
    outs = model.evaluate(x, y, batch_size=10)
    preds = model.predict(x)

    def ref_true_pos(y_true, y_pred):
        return np.sum(np.logical_and(y_pred > 0.5, y_true == 1))

    # Test correctness (e.g. updates should have been run)
    np.testing.assert_allclose(outs[2], ref_true_pos(y, preds), atol=1e-5)

    # Test correctness of the validation metric computation
    val_preds = model.predict(val_x)
    val_outs = model.evaluate(val_x, val_y, batch_size=10)
    np.testing.assert_allclose(val_outs[2], ref_true_pos(val_y, val_preds), atol=1e-5)
    np.testing.assert_allclose(val_outs[2], history.history['val_true_positives'][-1], atol=1e-5)

    # Test with generators
    gen = [(np.array([x0]), np.array([y0])) for x0, y0 in zip(x, y)]
    val_gen = [(np.array([x0]), np.array([y0])) for x0, y0 in zip(val_x, val_y)]
    history = model.fit_generator(iter(gen), epochs=1, steps_per_epoch=samples,
                                  validation_data=iter(val_gen), validation_steps=val_samples)
    outs = model.evaluate_generator(iter(gen), steps=samples)
    preds = model.predict_generator(iter(gen), steps=samples)

    # Test correctness of the metric re ref_true_pos()
    np.testing.assert_allclose(outs[2], ref_true_pos(y, preds), atol=1e-5)

    # Test correctness of the validation metric computation
    val_preds = model.predict_generator(iter(val_gen), steps=val_samples)
    val_outs = model.evaluate_generator(iter(val_gen), steps=val_samples)
    np.testing.assert_allclose(val_outs[2], ref_true_pos(val_y, val_preds), atol=1e-5)
    np.testing.assert_allclose(val_outs[2], history.history['val_true_positives'][-1], atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
