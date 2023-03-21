# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for ClusterCoordinator and Keras models."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.distribute import multi_worker_testing_utils
from keras.distribute import strategy_combinations
from keras.engine import base_layer


class ShardedVariableTest(tf.test.TestCase, parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.strategy = tf.distribute.experimental.ParameterServerStrategy(
            multi_worker_testing_utils.make_parameter_server_cluster(3, 2),
            variable_partitioner=tf.distribute.experimental.partitioners.FixedShardsPartitioner(  # noqa: E501
                2
            ),
        )

    def assert_list_all_equal(self, list1, list2):
        """Used in lieu of `assertAllEqual`.

        This is used to replace standard `assertAllEqual` for the cases where
        `list1` and `list2` contain `AggregatingVariable`. Lists with
        `AggregatingVariable` are not convertible to numpy array via `np.array`
        calls as numpy would raise `ValueError: setting an array element with a
        sequence.`

        Args:
          list1: The first list to compare equality.
          list2: The second list to compare equality.
        """
        for lhs, rhs in zip(list1, list2):
            self.assertEqual(lhs, rhs)

    def test_keras_layer_setattr(self):
        class Layer(base_layer.Layer):
            def __init__(self):
                super().__init__()
                self.w = tf.Variable([0, 1])
                self.b = tf.Variable([2, 3], trainable=False)

        with self.strategy.scope():
            layer = Layer()

        self.assertLen(layer.trainable_weights, 2)
        self.assertEqual(layer.trainable_weights[0], [0])
        self.assertEqual(layer.trainable_weights[1], [1])
        self.assertLen(layer.non_trainable_weights, 2)
        self.assertEqual(layer.non_trainable_weights[0], [2])
        self.assertEqual(layer.non_trainable_weights[1], [3])
        self.assert_list_all_equal(
            layer.weights, layer.trainable_weights + layer.non_trainable_weights
        )
        self.assert_list_all_equal(
            layer.trainable_weights, layer.trainable_variables
        )
        self.assert_list_all_equal(layer.weights, layer.variables)

        checkpoint_deps = set(layer._trackable_children().values())
        self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))

    def test_keras_layer_add_weight(self):
        class Layer(base_layer.Layer):
            def __init__(self):
                super().__init__()
                self.w = self.add_weight(
                    shape=(2,),
                    initializer=lambda shape, dtype: tf.constant(
                        [0.0, 1.0],
                    ),
                    trainable=True,
                )
                self.b = self.add_weight(
                    shape=(2,),
                    initializer=lambda shape, dtype: tf.constant([2.0, 3.0]),
                    trainable=False,
                )

        with self.strategy.scope():
            layer = Layer()

        self.assertLen(layer.trainable_weights, 2)
        self.assertEqual(layer.trainable_weights[0], [0.0])
        self.assertEqual(layer.trainable_weights[1], [1.0])
        self.assertLen(layer.non_trainable_weights, 2)
        self.assertEqual(layer.non_trainable_weights[0], [2.0])
        self.assertEqual(layer.non_trainable_weights[1], [3.0])
        self.assert_list_all_equal(
            layer.weights, layer.trainable_weights + layer.non_trainable_weights
        )
        self.assert_list_all_equal(
            layer.trainable_weights, layer.trainable_variables
        )
        self.assert_list_all_equal(layer.weights, layer.variables)

        checkpoint_deps = set(layer._trackable_children().values())
        self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))

    def test_keras_metrics(self):
        with self.strategy.scope():
            fp = keras.metrics.FalsePositives(thresholds=[0.2, 0.5, 0.7, 0.8])
            auc = keras.metrics.AUC(num_thresholds=10)

        @tf.function
        def update():
            fp.update_state([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.3, 0.9])
            auc.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])

        @tf.function
        def reset():
            fp.reset_state()
            auc.reset_state()

        update()
        self.assertEqual(auc.result(), 0.75)
        self.assertAllEqual(fp.result(), [2.0, 1.0, 1.0, 1.0])
        reset()
        self.assertEqual(auc.result(), 0.0)
        self.assertAllEqual(fp.result(), [0.0, 0.0, 0.0, 0.0])

        self.assertTrue(hasattr(auc.true_positives, "variables"))
        self.assertTrue(hasattr(fp.accumulator, "variables"))

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            shard_config=[
                [2, 2],
                [2, 3],
                [3, 2],
                [2, 1],
                [1, 1],
                [1, 2],
                [1, 3],
            ],
            model_type=["dense", "embedding"],
        )
    )
    def test_saved_model_combined(self, shard_config, model_type):
        """Test saving and loading models with various fixed numbers of shards.

        Args:
          shard_config: The number of shards to use per variable before and
            after loading. For example, [1, 3] means to create and save the
            model with 1 shard (i.e., no variable partitioning), and load it
            into 3 shards per variable.
          model_type: Either 'dense' or 'embedding', which simple model to test.
        """

        def create_embedding_model():
            inputs = keras.layers.Input(shape=(6,))
            embedding = keras.layers.Embedding(output_dim=2, input_dim=6)
            outputs = embedding(inputs)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="mean_squared_error")
            return model

        def create_dense_model():
            inputs = keras.layers.Input(shape=(6,))
            outputs = keras.layers.Dense(6)(inputs)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="mean_squared_error")
            return model

        # Maybe create new strategy with different number of shards
        if shard_config[0] > 2:
            strategy = tf.distribute.experimental.ParameterServerStrategy(
                multi_worker_testing_utils.make_parameter_server_cluster(3, 3),
                variable_partitioner=tf.distribute.experimental.partitioners.FixedShardsPartitioner(  # noqa: E501
                    shard_config[0]
                ),
            )
        elif shard_config[0] == 2:
            strategy = self.strategy
        else:
            # Just one shard, so use default strategy
            strategy = tf.distribute.get_strategy()

        x = tf.cast(tf.expand_dims(tf.range(6), 0), tf.float32)
        with strategy.scope():
            model = (
                create_dense_model()
                if model_type == "dense"
                else create_embedding_model()
            )
            expect = model(x)

        # Dense layers have two variables (kernel and bias), embedding layers
        # have 1
        n_expected_variables = shard_config[0] * (
            2 if model_type == "dense" else 1
        )
        self.assertLen(model.variables, n_expected_variables)
        model_weights = [v.numpy() for v in model.variables]

        saved_dir = self.get_temp_dir()
        model.save(saved_dir)

        if shard_config[1] > 2:
            strategy2 = tf.distribute.experimental.ParameterServerStrategy(
                multi_worker_testing_utils.make_parameter_server_cluster(3, 3),
                variable_partitioner=tf.distribute.experimental.partitioners.FixedShardsPartitioner(  # noqa: E501
                    shard_config[1]
                ),
            )
        elif shard_config[1] == 2:
            strategy2 = self.strategy
        else:
            # Just one shard, so use default strategy
            strategy2 = tf.distribute.get_strategy()

        with strategy2.scope():
            loaded_model = keras.models.load_model(saved_dir)
            got = loaded_model(x)

            self.assertAllClose(got, expect)
            n_expected_variables = shard_config[1] * (
                2 if model_type == "dense" else 1
            )
            self.assertLen(loaded_model.variables, n_expected_variables)
            loaded_model_weights = [v.numpy() for v in loaded_model.variables]
            self.assertAllClose(
                np.concatenate([w.flatten() for w in model_weights]),
                np.concatenate([w.flatten() for w in loaded_model_weights]),
            )

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            distribution=strategy_combinations.strategies_minus_tpu,
            model_type=["dense", "embedding"],
        )
    )
    def test_saved_model_load_non_pss(self, model_type, distribution):
        def create_embedding_model():
            inputs = keras.layers.Input(shape=(6,))
            embedding = keras.layers.Embedding(output_dim=2, input_dim=6)
            outputs = embedding(inputs)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="mean_squared_error")
            return model

        def create_dense_model():
            inputs = keras.layers.Input(shape=(6,))
            outputs = keras.layers.Dense(6)(inputs)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="mean_squared_error")
            return model

        x = tf.cast(tf.expand_dims(tf.range(6), 0), tf.float32)
        with self.strategy.scope():
            model = (
                create_dense_model()
                if model_type == "dense"
                else create_embedding_model()
            )
            expect = model(x)

        model_weights = [v.numpy() for v in model.variables]

        saved_dir = self.get_temp_dir()
        model.save(saved_dir)

        with distribution.scope():
            loaded_model = keras.models.load_model(saved_dir)
            got = loaded_model(x)

            self.assertAllClose(got, expect)
            n_expected_variables = 2 if model_type == "dense" else 1
            self.assertLen(loaded_model.variables, n_expected_variables)
            loaded_model_weights = [v.numpy() for v in loaded_model.variables]
            self.assertAllClose(
                np.concatenate([w.flatten() for w in model_weights]),
                np.concatenate([w.flatten() for w in loaded_model_weights]),
            )

    def test_slot_variable_checkpointing(self):

        with self.strategy.scope():
            # Set a name so the ShardedVariable is well-named for slot var
            # keying
            var = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="test")

        opt = keras.optimizers.legacy.adam.Adam()

        # Run once to trigger apply_gradients to populate optimizer slot
        # variables.
        def train_step():
            with tf.GradientTape() as tape:
                loss = sum(var)
            opt.minimize(loss, var.variables, tape=tape)

        self.strategy.run(train_step)

        # Check that we can call get_slot using each slot, before and after
        # Checkpointing, and get the same results
        pre_ckpt_slots = []
        for slot in opt.get_slot_names():
            pre_ckpt_slots.extend([v.numpy() for v in opt.get_slot(var, slot)])

        ckpt = tf.train.Checkpoint(var=var, opt=opt)

        # Assert that checkpoint has slots for each shard and the
        # ShardedVariable
        self.assertLen(ckpt.opt._slots, 3)
        for var_name in ckpt.opt._slots.keys():
            self.assertLen(ckpt.opt._slots[var_name], 2)
            self.assertEqual(ckpt.opt._slots[var_name].keys(), {"m", "v"})
            if hasattr(ckpt.opt._slots[var_name]["m"], "variables"):
                self.assertLen(ckpt.opt._slots[var_name]["m"].variables, 2)
                self.assertLen(ckpt.opt._slots[var_name]["v"].variables, 2)

        saved_dir = self.get_temp_dir()
        ckpt_prefix = f"{saved_dir}/ckpt"
        ckpt.save(ckpt_prefix)

        # Run once more to alter slot variables and ensure checkpoint restores
        # the earlier values.
        self.strategy.run(train_step)

        changed_ckpt_slots = []
        for slot in opt.get_slot_names():
            changed_ckpt_slots.extend(
                [v.numpy() for v in opt.get_slot(var, slot)]
            )
        self.assertNotAllClose(pre_ckpt_slots, changed_ckpt_slots)

        ckpt.restore(tf.train.latest_checkpoint(saved_dir))

        post_ckpt_slots = []
        for slot in opt.get_slot_names():
            post_ckpt_slots.extend([v.numpy() for v in opt.get_slot(var, slot)])

        self.assertAllClose(pre_ckpt_slots, post_ckpt_slots)

    def test_slot_variable_checkpoint_load_with_diff_shards(self):

        with self.strategy.scope():
            # Set a name so the ShardedVariable is well-named for slot var
            # keying
            var = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="test")

        opt = keras.optimizers.legacy.adam.Adam()

        # Run once to trigger apply_gradients to populate optimizer slot
        # variables.
        def train_step():
            with tf.GradientTape() as tape:
                loss = sum(var)
            opt.minimize(loss, var.variables, tape=tape)

        self.strategy.run(train_step)

        # Check that we can call get_slot using each slot, before and after
        # Checkpointing, and get the same results
        pre_ckpt_slots = []
        for slot in opt.get_slot_names():
            pre_ckpt_slots.extend(
                tf.concat(list(opt.get_slot(var, slot)), axis=0).numpy()
            )

        ckpt = tf.train.Checkpoint(var=var, opt=opt)
        saved_dir = self.get_temp_dir()
        ckpt_prefix = f"{saved_dir}/ckpt"
        ckpt.save(ckpt_prefix)

        # Create new strategy with different number of shards
        strategy2 = tf.distribute.experimental.ParameterServerStrategy(
            multi_worker_testing_utils.make_parameter_server_cluster(3, 2),
            variable_partitioner=tf.distribute.experimental.partitioners.FixedShardsPartitioner(  # noqa: E501
                3
            ),
        )

        # Create new variable with different values, to be overwritten by ckpt.
        with strategy2.scope():
            var = tf.Variable([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], name="test")

        opt = keras.optimizers.legacy.adam.Adam()
        # Run once to trigger apply_gradients to populate optimizer slot
        # variables.
        strategy2.run(train_step)

        new_ckpt = tf.train.Checkpoint(var=var, opt=opt)
        new_ckpt.restore(tf.train.latest_checkpoint(saved_dir))
        post_ckpt_slots = []
        for slot in new_ckpt.opt.get_slot_names():
            post_ckpt_slots.extend(
                tf.concat(
                    list(new_ckpt.opt.get_slot(var, slot)), axis=0
                ).numpy()
            )
        self.assertAllClose(pre_ckpt_slots, post_ckpt_slots)


class ShardedVariableMixedPartitioningTest(tf.test.TestCase):
    def test_saved_model_min_size_partitioner(self):

        # set min_shard_bytes such that Dense kernel is split into 2 and bias
        # into 1
        partitioner = (
            tf.distribute.experimental.partitioners.MinSizePartitioner(
                min_shard_bytes=(6 * 6 * 4) // 2, max_shards=2
            )
        )

        cluster_resolver = (
            multi_worker_testing_utils.make_parameter_server_cluster(3, 2)
        )
        strategy = tf.distribute.experimental.ParameterServerStrategy(
            cluster_resolver, variable_partitioner=partitioner
        )

        def create_dense_model():
            inputs = keras.layers.Input(shape=(6,))
            outputs = keras.layers.Dense(6)(inputs)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="mean_squared_error")
            return model

        x = tf.cast(tf.expand_dims(tf.range(6), 0), tf.float32)
        with strategy.scope():
            model = create_dense_model()
            expect = model(x)

        # 2 kernel variables, 1 bias
        self.assertLen(model.variables, 3)

        saved_dir = self.get_temp_dir()
        model.save(saved_dir)

        # set min_shard_bytes such that Dense kernel is split into 3 and bias
        # into 1
        partitioner2 = (
            tf.distribute.experimental.partitioners.MinSizePartitioner(
                min_shard_bytes=(6 * 6 * 4) // 3, max_shards=3
            )
        )
        strategy2 = tf.distribute.experimental.ParameterServerStrategy(
            cluster_resolver, variable_partitioner=partitioner2
        )

        with strategy2.scope():
            loaded_model = keras.models.load_model(saved_dir)
            got = loaded_model(x)

            self.assertAllClose(got, expect)
            # 3 kernel variables, 1 bias
            self.assertLen(loaded_model.variables, 4)


if __name__ == "__main__":
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
