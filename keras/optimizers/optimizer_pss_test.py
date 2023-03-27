"""Tests for calling optimizer on ParameterServerStrategy."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.optimizers import adadelta
from keras.optimizers import adagrad
from keras.optimizers import adam
from keras.optimizers import adamax
from keras.optimizers import adamw
from keras.optimizers import ftrl
from keras.optimizers import lion
from keras.optimizers import nadam
from keras.optimizers import rmsprop
from keras.optimizers import sgd
from keras.utils import dataset_creator
from keras.utils import losses_utils

ds_combinations = tf.__internal__.distribute.combinations

STRATEGIES = [
    ds_combinations.parameter_server_strategy_3worker_2ps_cpu,
    ds_combinations.parameter_server_strategy_3worker_2ps_1gpu,
]

adadelta_fn = tf.__internal__.test.combinations.NamedObject(
    "adadelta",
    lambda: adadelta.Adadelta(
        0.002, use_ema=True, ema_overwrite_frequency=None
    ),
)
adagrad_fn = tf.__internal__.test.combinations.NamedObject(
    "adagrad", lambda: adagrad.Adagrad(0.002)
)
adam_fn = tf.__internal__.test.combinations.NamedObject(
    "adam", lambda: adam.Adam(0.002)
)
adamax_fn = tf.__internal__.test.combinations.NamedObject(
    "adamax", lambda: adamax.Adamax(0.002)
)
adamw_fn = tf.__internal__.test.combinations.NamedObject(
    "adamw", lambda: adamw.AdamW(0.002, weight_decay=0.004)
)
ftrl_fn = tf.__internal__.test.combinations.NamedObject(
    "ftrl", lambda: ftrl.Ftrl(0.002)
)
lion_fn = tf.__internal__.test.combinations.NamedObject(
    "lion", lambda: lion.Lion(0.002)
)
nadam_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentnadam", lambda: nadam.Nadam(0.002)
)
rmsprop_fn = tf.__internal__.test.combinations.NamedObject(
    "rmsprop", lambda: rmsprop.RMSprop(0.002)
)
sgd_fn = tf.__internal__.test.combinations.NamedObject(
    "sgdaverage",
    lambda: sgd.SGD(0.002, use_ema=True, ema_overwrite_frequency=1),
)

OPTIMIZER_FN = [
    adadelta_fn,
    adagrad_fn,
    adam_fn,
    adamax_fn,
    adamw_fn,
    ftrl_fn,
    lion_fn,
    nadam_fn,
    rmsprop_fn,
    sgd_fn,
]


# TODO(b/228209527): Combine this test with optimizer_test after
# fixing the NCCL issue.
class OptimizerPssTest(tf.test.TestCase, parameterized.TestCase):
    def _get_model(self):
        return keras.Sequential(
            [keras.layers.Input(shape=(1,)), keras.layers.Dense(1)]
        )

    def _get_dataset_fn(self):
        def dataset_fn(_):
            x, y = [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            ds = ds.repeat().batch(6)
            return ds

        return dataset_fn

    def _verify_accumulators_updated(self, optimizer):
        variables = optimizer.variables
        for var in variables:
            if "iteration" not in var.name and "learning_rate" not in var.name:
                # Find a variable not iteration or learning_rate, and verify its
                # value is updated (not 0).
                self.assertNotAllEqual(var, 0)

    @ds_combinations.generate(
        tf.__internal__.test.combinations.combine(
            strategy=STRATEGIES, optimizer_fn=OPTIMIZER_FN
        )
    )
    def testGetGradientsInModelPss(self, strategy, optimizer_fn):
        with strategy.scope():
            model = self._get_model()
            optimizer = optimizer_fn()
        ds_fn = self._get_dataset_fn()
        if isinstance(strategy, tf.distribute.ParameterServerStrategy):
            ds = dataset_creator.DatasetCreator(ds_fn)
        else:
            ds = ds_fn(None)
        model.compile(loss="mse", optimizer=optimizer)
        model.fit(ds, epochs=1, steps_per_epoch=5)

        self._verify_accumulators_updated(optimizer)

    @ds_combinations.generate(
        tf.__internal__.test.combinations.combine(
            strategy=STRATEGIES, optimizer_fn=OPTIMIZER_FN
        )
    )
    def testGetGradientsInCustomTrainingLoopPss(self, strategy, optimizer_fn):
        coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
            strategy
        )

        with strategy.scope():
            model = self._get_model()
            optimizer = optimizer_fn()

            def per_worker_dataset_fn():
                return strategy.distribute_datasets_from_function(
                    self._get_dataset_fn()
                )

            ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)

            @tf.function
            def train_step(iterator):
                def replica_fn(data):
                    features, labels = data
                    with tf.GradientTape() as tape:
                        output = model(tf.expand_dims(features, axis=1))
                        loss = keras.losses.MeanSquaredError(
                            reduction=losses_utils.ReductionV2.NONE
                        )(labels, output)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables)
                    )

                strategy.run(replica_fn, args=(next(iterator),))

            for _ in range(3):
                coordinator.schedule(train_step, args=(iter(ds),))
                coordinator.join()
            self.assertEqual(self.evaluate(optimizer.iterations), 3)
            self._verify_accumulators_updated(optimizer)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
