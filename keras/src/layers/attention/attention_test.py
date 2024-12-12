import numpy as np

from keras.src import layers
from keras.src import ops
from keras.src import testing


class AttentionTest(testing.TestCase):
    def test_attention_basics(self):
        # No scale, no concat.
        self.run_layer_test(
            layers.Attention,
            init_kwargs={
                "score_mode": "dot",
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )
        # Scale and concat.
        self.run_layer_test(
            layers.Attention,
            init_kwargs={
                "use_scale": True,
                "score_mode": "concat",
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

    def test_attention_correctness(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        # Dot.
        layer = layers.Attention(score_mode="dot")
        output, scores = layer(
            [query, value, key],
            return_attention_scores=True,
        )
        self.assertAllClose(
            output, [[[2.462, 3.462], [1.538, 2.538]]], atol=1e-3
        )
        self.assertAllClose(
            scores, [[[0.269, 0.731], [0.731, 0.269]]], atol=1e-3
        )

        # Concat.
        layer = layers.Attention(score_mode="concat")
        output, scores = layer(
            [query, value, key],
            return_attention_scores=True,
        )
        self.assertAllClose(
            output, [[[1.727, 2.727], [2.272, 3.272]]], atol=1e-3
        )
        self.assertAllClose(
            scores, [[[0.636, 0.363], [0.363, 0.636]]], atol=1e-3
        )

    def test_attention_with_mask(self):
        layer = layers.Attention()
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        value = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        query_mask = np.array([[True, False]])
        value_mask = np.array([[True, False]])
        output, scores = layer(
            [query, value],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        self.assertAllClose(output, [[[1.0, 1.0], [0.0, 0.0]]])
        self.assertAllClose(scores, [[[1.0, 0.0], [1.0, 0.0]]])

    def test_attention_errors(self):
        layer = layers.Attention()
        tensor = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        with self.assertRaisesRegex(ValueError, "must be called on a list"):
            layer(tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor, tensor, tensor])

        with self.assertRaisesRegex(ValueError, "layer mask must be a list"):
            layer([tensor, tensor], mask=tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor], mask=[tensor])

    def test_attention_with_dropout(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        value = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        layer_with_dropout = layers.Attention(dropout=0.2)
        layer_without_dropout = layers.Attention()

        output1, scores1 = layer_with_dropout(
            [query, value], return_attention_scores=True, training=True
        )
        output2, scores2 = layer_without_dropout(
            [query, value], return_attention_scores=True, training=True
        )
        self.assertNotAllClose(output1, output2)
        self.assertNotAllClose(scores1, scores2)

    def test_attention_invalid_score_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value for argument score_mode. "
            "Expected one of {'dot', 'concat'}",
        ):
            layers.Attention(score_mode="invalid_mode")

    def test_attention_calculate_scores_with_scale(self):
        query = np.random.random((2, 3, 4))
        key = np.random.random((2, 4, 4))
        layer = layers.Attention(use_scale=True, score_mode="dot")
        layer.build(input_shape=[(2, 3, 4), (2, 4, 4)])
        expected_scores = np.matmul(query, key.transpose((0, 2, 1)))
        expected_scores *= layer.scale.numpy()
        actual_scores = layer._calculate_scores(query, key)
        self.assertAllClose(actual_scores, expected_scores)

    def test_attention_calculate_score_mask_no_causal_no_vmask(self):
        scores = np.random.random((2, 3, 4))
        layer = layers.Attention()
        mask = layer._calculate_score_mask(
            scores, v_mask=None, use_causal_mask=False
        )
        self.assertIsNone(
            mask,
            "Mask should be None when no causal mask and no value mask "
            "are used",
        )

    def test_attention_calculate_score_mask_with_causal_no_vmask(self):
        scores = np.random.random((2, 3, 4))
        layer = layers.Attention()

        causal_mask = layer._calculate_score_mask(
            scores, v_mask=None, use_causal_mask=True
        )
        expected_causal_mask = np.tril(
            np.ones((1, scores.shape[1], scores.shape[2])), k=0
        )
        self.assertAllClose(causal_mask, expected_causal_mask, atol=1e-6)

    def test_attention_calculate_score_mask_with_causal_and_vmask(self):
        scores = np.random.random((2, 3, 4))
        layer = layers.Attention()
        v_mask = np.array([[True, False, True, False]])

        combined_mask = layer._calculate_score_mask(
            scores, v_mask=v_mask, use_causal_mask=True
        )
        expected_causal_mask = np.tril(
            np.ones((1, scores.shape[1], scores.shape[2])), k=0
        )
        expected_combined_mask = np.logical_and(
            expected_causal_mask, v_mask[:, np.newaxis, :]
        )
        self.assertAllClose(combined_mask, expected_combined_mask, atol=1e-6)

    def test_attention_compute_mask_with_no_mask(self):
        layer = layers.Attention()
        dummy_inputs = [
            np.random.random((2, 3, 4)),
            np.random.random((2, 4, 4)),
        ]
        self.assertIsNone(
            layer.compute_mask(inputs=dummy_inputs, mask=None),
            "compute_mask should return None when mask is None",
        )

    def test_attention_compute_mask_with_first_element_none(self):
        layer = layers.Attention()
        dummy_inputs = [
            np.random.random((2, 3, 4)),
            np.random.random((2, 4, 4)),
        ]
        mask = [None, np.array([True, False, True])]
        self.assertIsNone(
            layer.compute_mask(inputs=dummy_inputs, mask=mask),
            "compute_mask should return None when the first element is None",
        )

    def test_attention_compute_mask_does_not_return_none_with_valid_mask(self):
        layer = layers.Attention()
        dummy_inputs = [
            np.random.random((2, 3, 4)),
            np.random.random((2, 4, 4)),
        ]
        valid_mask = np.array([True, False, True])
        mask = [valid_mask, np.array([False, True, False])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        self.assertIsNotNone(
            computed_mask,
            "compute_mask should not return None with a valid mask",
        )

    def test_attention_compute_mask_returns_correct_tensor_with_valid_mask(
        self,
    ):
        layer = layers.Attention()
        dummy_inputs = [
            np.random.random((2, 3, 4)),
            np.random.random((2, 4, 4)),
        ]
        valid_mask = np.array([True, False, True])
        mask = [valid_mask, np.array([False, True, False])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        self.assertTrue(
            np.array_equal(computed_mask, valid_mask),
            "compute_mask did not return the correct mask tensor",
        )

    def test_attention_compute_mask_returns_correct_tensor_with_all_true_mask(
        self,
    ):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([True, True, True])
        mask = [valid_mask, np.array([True, True, True])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = np.array([True, True, True])
        self.assertTrue(
            np.array_equal(computed_mask, expected_mask),
            "compute_mask did not return the correct mask tensor",
        )

    def test_attention_compute_mask_returns_correct_tensor_with_all_false_mask(
        self,
    ):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([False, False, False])
        mask = [valid_mask, np.array([False, False, False])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = np.array([False, False, False])
        self.assertTrue(
            np.array_equal(computed_mask, expected_mask),
            "compute_mask did not return the correct mask tensor",
        )

    def test_attention_compute_mask_with_tolerance_1e_3(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([1.0, 0.0, 1.0], dtype=float)
        mask = [valid_mask, np.array([0.0, 1.0, 0.0], dtype=float)]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = valid_mask
        self.assertTrue(
            np.allclose(computed_mask, expected_mask, atol=1e-3),
            "Incorrect mask tensor within tolerance 1e-3",
        )

    def test_attention_compute_mask_with_tolerance_1e_5(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([1.0, 0.0, 1.0], dtype=float)
        mask = [valid_mask, np.array([0.0, 1.0, 0.0], dtype=float)]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = valid_mask
        self.assertTrue(
            np.allclose(computed_mask, expected_mask, atol=1e-5),
            "Incorrect mask tensor within tolerance 1e-5",
        )

    def test_attention_compute_mask_with_tolerance_1e_7(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([1.0, 0.0, 1.0], dtype=float)
        mask = [valid_mask, np.array([0.0, 1.0, 0.0], dtype=float)]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = valid_mask
        self.assertTrue(
            np.allclose(computed_mask, expected_mask, atol=1e-7),
            "Incorrect mask tensor within tolerance 1e-7 ",
        )

    def test_attention_compute_mask_with_single_element_masks(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([True])
        mask = [valid_mask, np.array([False])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_shape = (1,)
        self.assertEqual(computed_mask.shape, expected_shape)

    def test_attention_compute_mask_with_non_boolean_masks(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([1, 0, 1])
        mask = [valid_mask, np.array([0, 1, 0])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        self.assertTrue(np.array_equal(computed_mask, valid_mask))

    def test_attention_compute_mask_with_edge_case_masks(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        edge_case_masks = [
            np.array([True, True, True]),
            np.array([False, False, False]),
            np.array([True, False, True]),
        ]
        for mask in edge_case_masks:
            computed_mask = layer.compute_mask(
                inputs=dummy_inputs, mask=[mask, mask]
            )
            computed_mask = ops.convert_to_numpy(computed_mask)
            self.assertTrue(np.array_equal(computed_mask, mask))

    def test_attention_compute_mask_with_different_input_shapes(self):
        layer = layers.Attention()
        input_shapes = [(2, 3, 4), (3, 2, 5), (4, 1, 6)]
        valid_mask = np.array([True, False, True])
        for shape in input_shapes:
            dummy_inputs = [np.ones(shape), np.ones(shape)]
            mask = [valid_mask, np.array([False, True, False])]
            computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
            computed_mask = ops.convert_to_numpy(computed_mask)
            self.assertTrue(np.array_equal(computed_mask, valid_mask))

    def test_attention_compute_output_shape(self):
        layer = layers.Attention()

        query = np.random.random((2, 3, 4))
        value = np.random.random((2, 3, 5))
        key = np.random.random((2, 3, 4))
        layer = layers.Attention()
        output = layer([query, value, key])
        self.assertAllEqual(output.shape, value.shape)
        self.assertAllEqual(
            layer.compute_output_shape(
                input_shape=[query.shape, value.shape, key.shape]
            ),
            output.shape,
        )
