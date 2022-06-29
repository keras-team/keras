# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests Attention layer."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.layers import core
from keras.testing_infra import test_combinations


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class AttentionTest(tf.test.TestCase, parameterized.TestCase):
    def test_calculate_scores_one_dim(self):
        # Query tensor of shape [1, 1, 1]
        q = np.array([[[1.1]]], dtype=np.float32)
        # Key tensor of shape [1, 1, 1]
        k = np.array([[[1.6]]], dtype=np.float32)
        attention_layer = keras.layers.Attention()
        attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
        actual = attention_layer._calculate_scores(query=q, key=k)

        # Expected tensor of shape [1, 1, 1].
        # expected000 = 1.1*1.6 = 1.76
        expected = np.array([[[1.76]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_calculate_scores_multi_dim(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Key tensor of shape [1, 3, 4]
        k = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        attention_layer = keras.layers.Attention()
        attention_layer.build(input_shape=([1, 2, 4], [1, 3, 4]))
        actual = attention_layer._calculate_scores(query=q, key=k)

        # Expected tensor of shape [1, 2, 3].
        # expected000 = 1.*1.5+1.1*1.6+1.2*1.7+1.3*1.8 = 7.64
        # expected001 = 1.*2.5+1.1*2.6+1.2*2.7+1.3*2.8 = 12.24
        # expected002 = 1.*3.5+1.1*3.6+1.2*3.7+1.3*3.8 = 16.84
        # expected010 = 2.*1.5+2.1*1.6+2.2*1.7+2.3*1.8 = 14.24
        # expected011 = 2.*2.5+2.1*2.6+2.2*2.7+2.3*2.8 = 22.84
        # expected012 = 2.*3.5+2.1*3.6+2.2*3.7+2.3*3.8 = 31.44
        expected = np.array(
            [[[7.64, 12.24, 16.84], [14.24, 22.84, 31.44]]], dtype=np.float32
        )
        self.assertAllClose(expected, actual)

    def test_calculate_scores_multi_dim_concat(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Key tensor of shape [1, 3, 4]
        k = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        attention_layer = keras.layers.Attention(score_mode="concat")
        attention_layer.concat_score_weight = 1
        attention_layer.build(input_shape=([1, 2, 4], [1, 3, 4]))
        actual = keras.backend.get_value(
            attention_layer._calculate_scores(query=q, key=k)
        )

        # expected000 = tanh(1.+1.5) + tanh(1.1+1.6) + \
        #     tanh(1.2+1.7) + tanh(1.3+1.8) = 3.96753427840
        # expected001 = tanh(1.+2.5) + tanh(1.1+2.6) + \
        #     tanh(1.2+2.7) + tanh(1.3+2.8) = 3.99558784825
        # expected002 = tanh(1.+3.5) + tanh(1.1+3.6) + \
        #     tanh(1.2+3.7) + tanh(1.3+3.8) = 3.99940254147
        # expected010 = tanh(2.+1.5) + tanh(2.1+1.6) + \
        #     tanh(2.2+1.7) + tanh(2.3+1.8) = 3.99558784825
        # expected011 = tanh(2.+2.5) + tanh(2.1+2.6) + \
        #     tanh(2.2+2.7) + tanh(2.3+2.8) = 3.99940254147
        # expected012 = tanh(2.+3.5) + tanh(2.1+3.6) + \
        #     tanh(2.2+3.7) + tanh(2.3+3.8) = 3.99991913657
        expected = np.array(
            [
                [
                    [3.96753427840, 3.99558784825, 3.99940254147],
                    [3.99558784825, 3.99940254147, 3.99991913657],
                ]
            ],
            dtype=np.float32,
        )
        self.assertAllClose(expected, actual)

    def test_calculate_scores_one_dim_batch_size_two(self):
        # Query tensor of shape [2, 1, 1]
        q = np.array([[[1.1]], [[2.1]]], dtype=np.float32)
        # Key tensor of shape [2, 1, 1]
        k = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
        attention_layer = keras.layers.Attention()
        attention_layer.build(input_shape=([2, 1, 1], [2, 1, 1]))
        actual = attention_layer._calculate_scores(query=q, key=k)

        # Expected tensor of shape [2, 1, 1].
        # expected000 = 1.1*1.6 = 1.76
        # expected100 = 2.1*2.6 = 5.46
        expected = np.array([[[1.76]], [[5.46]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_calculate_scores_one_dim_with_scale(self):
        """Tests that scores are multiplied by scale."""
        # Query tensor of shape [1, 1, 1]
        q = np.array([[[1.1]]], dtype=np.float32)
        # Key tensor of shape [1, 1, 1]
        k = np.array([[[1.6]]], dtype=np.float32)
        attention_layer = keras.layers.Attention(use_scale=True)
        attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
        attention_layer.scale = -2.0
        actual = attention_layer._calculate_scores(query=q, key=k)

        # Expected tensor of shape [1, 1, 1].
        # expected000 = -2*1.1*1.6 = -3.52
        expected = np.array([[[-3.52]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_calculate_scores_one_dim_with_scale_concat(self):
        """Tests that scores are multiplied by scale."""
        # Query tensor of shape [1, 1, 1]
        q = np.array([[[1.1]]], dtype=np.float32)
        # Key tensor of shape [1, 1, 1]
        k = np.array([[[1.6]]], dtype=np.float32)
        attention_layer = keras.layers.Attention(
            use_scale=True, score_mode="concat"
        )
        attention_layer.concat_score_weight = 1
        attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
        attention_layer.scale = 2.0
        actual = keras.backend.get_value(
            attention_layer._calculate_scores(query=q, key=k)
        )

        # Expected tensor of shape [1, 1, 1].
        # expected000 = tanh(2*(1.1+1.6)) = 0.9999592018254402
        expected = np.array([[[0.999959202]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_shape(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Value tensor of shape [1, 3, 4]
        v = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.Attention()
        actual = attention_layer([q, v], mask=[None, v_mask])

        expected_shape = [1, 2, 4]
        self.assertAllEqual(expected_shape, tf.shape(actual))

    def test_shape_concat(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Value tensor of shape [1, 3, 4]
        v = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.Attention(score_mode="concat")
        attention_layer.concat_score_weight = 1
        actual = attention_layer([q, v], mask=[None, v_mask])

        expected_shape = [1, 2, 4]
        self.assertAllEqual(expected_shape, tf.shape(actual))

    def test_shape_with_key(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Value tensor of shape [1, 3, 4]
        v = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Key tensor of shape [1, 3, 4]
        k = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.Attention()
        actual = attention_layer([q, v, k], mask=[None, v_mask])

        expected_shape = [1, 2, 4]
        self.assertAllEqual(expected_shape, tf.shape(actual))

    def test_shape_with_key_concat(self):
        # Query tensor of shape [1, 2, 4]
        q = np.array(
            [[[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]], dtype=np.float32
        )
        # Value tensor of shape [1, 3, 4]
        v = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Key tensor of shape [1, 3, 4]
        k = np.array(
            [
                [
                    [1.5, 1.6, 1.7, 1.8],
                    [2.5, 2.6, 2.7, 2.8],
                    [3.5, 3.6, 3.7, 3.8],
                ]
            ],
            dtype=np.float32,
        )
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.Attention(score_mode="concat")
        attention_layer.concat_score_weight = 1
        actual = attention_layer([q, v, k], mask=[None, v_mask])

        expected_shape = [1, 2, 4]
        self.assertAllEqual(expected_shape, tf.shape(actual))

    def test_multi_dim(self):
        # Query tensor of shape [1, 1, 1]
        q = np.array([[[1.1]]], dtype=np.float32)
        # Value tensor of shape [1, 3, 1]
        v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.Attention()
        actual = attention_layer([q, v], mask=[None, v_mask])

        # Expected scores of shape [1, 1, 3]
        # scores = [[[1.1*1.6, 1.1*0.7, -1.1*0.8]]] = [[[1.76, 0.77, -0.88]]]
        # Expected attention distribution = softmax(scores) with zeros in
        # positions where v_mask == False.
        # => attention_distribution000 = exp(1.76)/(exp(1.76) + exp(0.77))
        #                              = 0.72908792234
        #    attention_distribution001 = exp(0.77)/(exp(1.76) + exp(0.77))
        #                              = 0.27091207765
        #    attention_distribution002 = 0
        #
        # Expected tensor of shape [1, 1, 1].
        # expected000 = 0.72908792234 * 1.6 + 0.27091207765 * 0.7 - 0 * 0.8
        #             = 1.3561791301
        expected = np.array([[[1.3561791301]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_multi_dim_with_key(self):
        # Query tensor of shape [1, 1, 1]
        q = np.array([[[1.1]]], dtype=np.float32)
        # Value tensor of shape [1, 3, 1]
        v = np.array([[[0.5], [0.8], [-0.3]]], dtype=np.float32)
        # Key tensor of shape [1, 3, 1]
        k = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.Attention()
        actual = attention_layer([q, v, k], mask=[None, v_mask])

        # Expected scores of shape [1, 1, 3]
        # scores = [[[1.1*1.6, 1.1*0.7, -1.1*0.8]]] = [[[1.76, 0.77, -0.88]]]
        # Expected attention distribution = softmax(scores) with zeros in
        # positions where v_mask == False.
        # => attention_distribution000 = exp(1.76)/(exp(1.76) + exp(0.77))
        #                              = 0.72908792234
        #    attention_distribution001 = exp(0.77)/(exp(1.76) + exp(0.77))
        #                              = 0.27091207765
        #    attention_distribution002 = 0
        #
        # Expected tensor of shape [1, 1, 1].
        # expected000 = 0.72908792234 * 0.5 + 0.27091207765 * 0.8 - 0 * 0.3
        #             = 0.58127362329
        expected = np.array([[[0.58127362329]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    @parameterized.named_parameters(
        ("", False),
        ("return_attention_scores", True),
    )
    def test_multi_dim_with_query_mask(self, return_attention_scores):
        # Query tensor of shape [1, 2, 1]
        q = np.array([[[1.1], [-0.5]]], dtype=np.float32)
        # Value tensor of shape [1, 3, 1]
        v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
        # Query mask tensor of shape [1, 2]
        q_mask = np.array([[True, False]], dtype=np.bool_)
        # Value mask tensor of shape [1, 3]
        v_mask = np.array([[True, True, False]], dtype=np.bool_)
        attention_layer = keras.layers.Attention()
        if return_attention_scores:
            actual, actual_scores = attention_layer(
                [q, v],
                mask=[q_mask, v_mask],
                return_attention_scores=return_attention_scores,
            )
        else:
            actual = attention_layer(
                [q, v],
                mask=[q_mask, v_mask],
                return_attention_scores=return_attention_scores,
            )

        # Expected scores of shape [1, 2, 3]
        # scores = [[[1.1*1.6, 1.1*0.7, -1.1*0.8],
        #            [-0.5*1.6, -0.5*0.7, 0.5*0.8]]]
        #        = [[[1.76, 0.77, -0.88], [-0.8, -0.35, 0.4]]]
        # Expected attention distribution = softmax(scores) with zeros in
        # positions where v_mask == False.
        # => attention_distribution000 = exp(1.76)/(exp(1.76) + exp(0.77))
        #                              = 0.72908792234
        #    attention_distribution001 = exp(0.77)/(exp(1.76) + exp(0.77))
        #                              = 0.27091207765
        #    attention_distribution002 = 0
        # => attention_distribution010 = exp(-0.8)/(exp(-0.8) + exp(-0.35))
        #                              = 0.38936076605
        #    attention_distribution011 = exp(-0.35)/(exp(-0.8) + exp(-0.35))
        #                              = 0.61063923394
        #    attention_distribution012 = 0
        if return_attention_scores:
            expected_scores = np.array(
                [
                    [
                        [0.72908792234, 0.27091207765, 0.0],
                        [0.38936076605, 0.61063923394, 0.0],
                    ]
                ],
                dtype=np.float32,
            )
            self.assertAllClose(expected_scores, actual_scores)
        # Expected tensor of shape [1, 2, 1] with zeros where  q_mask == False.
        # expected000 = 0.72908792234 * 1.6 + 0.27091207765 * 0.7 - 0 * 0.8
        #             = 1.3561791301
        # expected000 = 0
        expected = np.array([[[1.3561791301], [0.0]]], dtype=np.float32)
        self.assertAllClose(expected, actual)

    def test_scale_none(self):
        """Tests that scale is None by default."""
        attention_layer = keras.layers.Attention()
        attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
        self.assertIsNone(attention_layer.scale)

    def test_scale_init_eager(self):
        """Tests that scale initializes to 1 when use_scale=True."""
        if not tf.executing_eagerly():
            self.skipTest("Only run in eager mode")
        attention_layer = keras.layers.Attention(use_scale=True)
        attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
        self.assertAllClose(1.0, attention_layer.scale.value())

    def test_scale_init_graph(self):
        """Tests that scale initializes to 1 when use_scale=True."""
        with self.cached_session() as sess:
            attention_layer = keras.layers.Attention(use_scale=True)
            attention_layer.build(input_shape=([1, 1, 1], [1, 1, 1]))
            sess.run(attention_layer.scale.initializer)
            self.assertAllClose(1.0, attention_layer.scale.value())

    @parameterized.named_parameters(
        ("", False),
        ("return_attention_scores", True),
    )
    def test_self_attention_causal(self, return_attention_scores):
        # Query-value tensor of shape [1, 3, 1]
        q = np.array([[[0.5], [0.8], [-0.3]]], dtype=np.float32)
        attention_layer = keras.layers.Attention()
        if return_attention_scores:
            actual, actual_scores = attention_layer(
                [q, q],
                return_attention_scores=return_attention_scores,
                use_causal_mask=True,
            )
        else:
            actual = attention_layer(
                [q, q],
                return_attention_scores=return_attention_scores,
                use_causal_mask=True,
            )

        # Expected scores of shape [1, 3, 3]
        # scores = [[0.25, 0.4, -0.15],
        #           [0.4, 0.64, -0.24],
        #           [-0.15, -0.24, 0.09]]
        # Expected attention distribution = softmax(scores) lower triangular
        # => attention_distribution00 = [1., 0., 0.]
        #    attention_distribution01
        #      = [exp(0.4), exp(0.64), 0.] / (exp(0.4) + exp(0.64))
        #      = [0.44028635073, 0.55971364926, 0.]
        #    attention_distribution02
        #      = [exp(-0.15), exp(-0.24), exp(0.09)]
        #        / (exp(-0.15) + exp(-0.24) + exp(0.09))
        #      = [0.31395396638, 0.28693232061, 0.399113713]
        if return_attention_scores:
            expected_scores = np.array(
                [
                    [
                        [1.0, 0.0, 0.0],
                        [0.44028635073, 0.55971364926, 0.0],
                        [0.31395396638, 0.28693232061, 0.399113713],
                    ]
                ],
                dtype=np.float32,
            )
            self.assertAllClose(expected_scores, actual_scores)
        # Expected tensor of shape [1, 3, 1].
        # expected000 = 0.5
        # expected010 = 0.44028635073 * 0.5 + 0.55971364926 * 0.8
        #             = 0.66791409477
        # expected020 = 0.31395396638 * 0.5 + \
        #     0.28693232061 * 0.8 -0.399113713 * 0.3
        #             = 0.26678872577
        expected = np.array(
            [[[0.5], [0.66791409477], [0.26678872577]]], dtype=np.float32
        )
        self.assertAllClose(expected, actual)

    def test_self_attention_causal_deprecated(self):
        """Verify deprecated specification of causal masking still works."""
        # Query-value tensor of shape [1, 3, 1]
        q = np.array([[[0.5], [0.8], [-0.3]]], dtype=np.float32)
        attention_layer_new = keras.layers.Attention()
        new_scores = attention_layer_new(
            [q, q],
            use_causal_mask=True,
        )
        attention_layer_old = keras.layers.Attention(causal=True)
        old_scores = attention_layer_old(
            [q, q],
        )
        self.assertAllClose(new_scores, old_scores)

    def test_inputs_not_list(self):
        attention_layer = keras.layers.Attention()
        q = np.array([[[1.1]]], dtype=np.float32)
        with self.assertRaisesRegex(
            ValueError, "Attention layer must be called on a list of inputs"
        ):
            attention_layer(q)

    def test_inputs_too_short(self):
        attention_layer = keras.layers.Attention()
        q = np.array([[[1.1]]], dtype=np.float32)
        with self.assertRaisesRegex(
            ValueError, "Attention layer accepts inputs list of length 2 or 3"
        ):
            attention_layer([q])

    def test_inputs_too_long(self):
        attention_layer = keras.layers.Attention()
        q = np.array([[[1.1]]], dtype=np.float32)
        with self.assertRaisesRegex(
            ValueError, "Attention layer accepts inputs list of length 2 or 3"
        ):
            attention_layer([q, q, q, q])

    def test_mask_not_list(self):
        attention_layer = keras.layers.Attention()
        q = np.array([[[1.1]]], dtype=np.float32)
        mask = np.array([[True]], dtype=np.bool_)
        with self.assertRaisesRegex(
            ValueError, "Attention layer mask must be a list"
        ):
            attention_layer([q, q], mask=mask)

    def test_mask_too_short(self):
        attention_layer = keras.layers.Attention()
        q = np.array([[[1.1]]], dtype=np.float32)
        mask = np.array([[True]], dtype=np.bool_)
        with self.assertRaisesRegex(
            ValueError, "Attention layer mask must be a list of length 2"
        ):
            attention_layer([q, q], mask=[mask])

    def test_mask_too_long(self):
        attention_layer = keras.layers.Attention()
        q = np.array([[[1.1]]], dtype=np.float32)
        mask = np.array([[True]], dtype=np.bool_)
        with self.assertRaisesRegex(
            ValueError, "Attention layer mask must be a list of length 2"
        ):
            attention_layer([q, q], mask=[mask, mask, mask])

    def test_override_mask(self):
        attention_layer = keras.layers.Attention()
        q = core.Masking()(np.array([[[1.1]]], dtype=np.float32))
        mask = np.array([[False]], dtype=np.bool_)
        actual = attention_layer([q, q], mask=[mask, mask])
        self.assertAllClose([[[0]]], actual)

    def test_implicit_mask(self):
        attention_layer = keras.layers.Attention()
        q = core.Masking(1.1)(np.array([[[1.1], [1]]], dtype=np.float32))
        v = core.Masking(1.2)(np.array([[[1.2], [1]]], dtype=np.float32))
        actual = attention_layer([q, v])
        self.assertAllClose([[[0], [1]]], actual)

    @parameterized.named_parameters(
        ("", False),
        ("use_scale", True),
    )
    def test_serialization(self, use_scale):
        # Test serialization with use_scale
        layer = keras.layers.Attention(use_scale=use_scale)

        config = keras.layers.serialize(layer)
        new_layer = keras.layers.deserialize(config)
        self.assertEqual(new_layer.use_scale, use_scale)

        config = layer.get_config()
        new_layer = keras.layers.Attention.from_config(config)
        self.assertEqual(new_layer.use_scale, use_scale)


if __name__ == "__main__":
    tf.test.main()
