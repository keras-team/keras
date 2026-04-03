import os

import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.saving import saving_api


@pytest.mark.skipif(
    backend.backend() == "numpy", reason="Failing for numpy backend."
)
class IndexLookupLayerTest(testing.TestCase):
    def test_basics_string_vocab(self):
        # Case: adapt + list inputs
        adapt_data = ["one", "one", "one", "two", "two", "three"]
        input_data = ["one", "two", "four"]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: numpy array input
        output = layer(np.array(input_data))
        self.assertEqual(list(output), [2, 3, 1])

        # Case: fixed vocab + list inputs
        vocabulary = ["one", "two", "three"]
        layer = layers.IndexLookup(vocabulary=vocabulary, **kwargs)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: fixed vocab with special tokens + list inputs
        vocabulary_with_special_tokens = ["", "[OOV]", "one", "two", "three"]
        layer = layers.IndexLookup(
            vocabulary=vocabulary_with_special_tokens, **kwargs
        )
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: set vocabulary
        layer = layers.IndexLookup(**kwargs)
        layer.set_vocabulary(vocabulary)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: set vocabulary (with special tokens)
        layer = layers.IndexLookup(**kwargs)
        layer.set_vocabulary(vocabulary_with_special_tokens)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_basics_integer_vocab(self):
        # Case: adapt + list inputs
        adapt_data = [1, 1, 1, 2, 2, 3]
        input_data = [1, 2, 4]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": 0,
            "oov_token": -1,
            "vocabulary_dtype": "int64",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: numpy array input
        output = layer(np.array(input_data))
        self.assertEqual(list(output), [2, 3, 1])

        # Case: fixed vocab + list inputs
        vocabulary = [1, 2, 3]
        layer = layers.IndexLookup(vocabulary=vocabulary, **kwargs)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: fixed vocab with special tokens + list inputs
        vocabulary_with_special_tokens = [0, -1, 1, 2, 3]
        layer = layers.IndexLookup(
            vocabulary=vocabulary_with_special_tokens, **kwargs
        )
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: set vocabulary
        layer = layers.IndexLookup(**kwargs)
        layer.set_vocabulary(vocabulary)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

        # Case: set vocabulary (with special tokens)
        layer = layers.IndexLookup(**kwargs)
        layer.set_vocabulary(vocabulary_with_special_tokens)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2, 3])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2, 3],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_max_tokens_adapt(self):
        adapt_data = [1, 1, 1, 2, 2, 3]
        input_data = [1, 2, 3, 4]
        kwargs = {
            "max_tokens": 4,
            "num_oov_indices": 1,
            "mask_token": 0,
            "oov_token": -1,
            "vocabulary_dtype": "int64",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        self.assertEqual(layer.get_vocabulary(), [0, -1, 1, 2])
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            [1, 2],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_pad_to_max_tokens(self):
        vocabulary = [1, 2]
        input_data = [1, 2]
        kwargs = {
            "max_tokens": 5,
            "num_oov_indices": 1,
            "mask_token": 0,
            "oov_token": -1,
            "vocabulary_dtype": "int64",
            "vocabulary": vocabulary,
            "pad_to_max_tokens": True,
            "output_mode": "multi_hot",
        }
        layer = layers.IndexLookup(**kwargs)
        output = layer(input_data)
        self.assertAllClose(output, [0, 1, 1, 0, 0])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_output_modes(self):
        vocabulary = ["one", "two", "three"]
        single_sample_input_data = ["one", "two", "four"]
        batch_input_data = [["one", "two", "four", "two"]]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "vocabulary": vocabulary,
        }

        # int
        kwargs["output_mode"] = "int"
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [2, 3, 1])
        output = layer(batch_input_data)
        self.assertAllClose(output, [[2, 3, 1, 3]])

        # multi-hot
        kwargs["output_mode"] = "multi_hot"
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [1, 1, 1, 0])
        output = layer(batch_input_data)
        self.assertAllClose(output, [[1, 1, 1, 0]])

        # one-hot
        kwargs["output_mode"] = "one_hot"
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])

        # count
        kwargs["output_mode"] = "count"
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [1, 1, 1, 0])
        output = layer(batch_input_data)
        self.assertAllClose(output, [[1, 1, 2, 0]])

        # tf-idf
        kwargs["output_mode"] = "tf_idf"
        kwargs["idf_weights"] = np.array([0.1, 0.2, 0.3])
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertAllClose(output, [0.2, 0.1, 0.2, 0.0])
        output = layer(batch_input_data)
        self.assertAllClose(output, [[0.2, 0.1, 0.4, 0.0]])

    def test_one_hot_symbolic_output_shape_with_higher_rank_input(self):
        """Symbolic output shape for one_hot must preserve input dims + depth.

        Regression test for gh-22336: StringLookup/IntegerLookup with
        output_mode='one_hot' produced (None, depth) instead of
        (None, d1, ..., dN, depth) for nested inputs.
        """
        # IntegerLookup with one_hot and 3D input (batch, 2, 2)
        layer = layers.IntegerLookup(
            vocabulary=[1, 2, 3],
            output_mode="one_hot",
        )
        symbolic_input = layers.Input(shape=(2, 2), dtype="int32")
        symbolic_output = layer(symbolic_input)
        # Expected: (None, 2, 2, vocab_size) where vocab_size = 4 (3 + OOV)
        self.assertEqual(
            tuple(symbolic_output.shape),
            (None, 2, 2, 4),
            msg="one_hot symbolic output shape must be input_shape + (depth,)",
        )
        # Eager execution: same input shape -> same output shape
        eager_input = np.array([[[1, 2], [3, 0]], [[1, 2], [3, 0]]])
        eager_output = layer(eager_input)
        self.assertEqual(eager_output.shape, (2, 2, 2, 4))
        self.assertEqual(
            tuple(symbolic_output.shape)[1:],
            eager_output.shape[1:],
            msg="Symbolic and eager output shapes must match (except batch)",
        )

    def test_one_hot_compute_output_shape_multi_hot_consistency(self):
        """multi_hot/count/tf_idf last dim is sample in output shape."""
        kwargs = {
            "max_tokens": 10,
            "num_oov_indices": 1,
            "mask_token": None,
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "vocabulary": ["a", "b", "c"],
        }
        # depth = vocab size (3) + OOV (1) = 4 when pad_to_max_tokens is False
        depth = 4
        # multi_hot: (batch, sample_len) -> (batch, depth)
        layer_multi = layers.IndexLookup(**kwargs, output_mode="multi_hot")
        shape_multi = layer_multi.compute_output_shape((None, 5))
        self.assertEqual(shape_multi, (None, depth))
        # one_hot: (batch, d1, d2) -> (batch, d1, d2, depth)
        layer_one = layers.IndexLookup(**kwargs, output_mode="one_hot")
        shape_one = layer_one.compute_output_shape((None, 2, 2))
        self.assertEqual(shape_one, (None, 2, 2, depth))

    def test_one_hot_compute_output_spec_preserves_input_dims(self):
        """compute_output_spec for one_hot must preserve all input dims."""
        layer = layers.IntegerLookup(
            vocabulary=[1, 2, 3],
            output_mode="one_hot",
        )
        symbolic_input = layers.Input(shape=(3, 4), dtype="int32")
        output_spec = layer.compute_output_spec(symbolic_input)
        self.assertEqual(output_spec.shape, (None, 3, 4, 4))
        self.assertEqual(output_spec.dtype, backend.floatx())

    def test_sparse_outputs(self):
        # TODO
        pass

    def test_adapt_tf_idf(self):
        # Case: unbatched data
        adapt_data = ["one", "one", "one", "two", "two", "three"]
        input_data = ["one", "two", "four"]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "output_mode": "tf_idf",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        output = layer(input_data)
        # Document counts for one, two, three = [3, 2, 1]
        idf_weights = np.log(1 + len(adapt_data) / (1 + np.array([3, 2, 1])))
        self.assertAllClose(layer.idf_weights[1:], idf_weights)
        self.assertAllClose(output, [1.1337324, 0.91629076, 1.0986123, 0.0])
        # Case: batched data
        adapt_data = [["one", "one"], ["one", "two"], ["two", "three"]]
        input_data = [["one", "two"], ["two", "four"]]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "output_mode": "tf_idf",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        # Document counts for one, two, three = [2, 2, 1]
        idf_weights = np.log(1 + len(adapt_data) / (1 + np.array([2, 2, 1])))
        self.assertAllClose(layer.idf_weights[1:], idf_weights)
        output = layer(input_data)
        self.assertAllClose(
            output,
            [
                [0.0, 0.6931472, 0.6931472, 0.0],
                [0.76752836, 0.0, 0.6931472, 0.0],
            ],
        )

    def test_invert(self):
        vocabulary = ["one", "two", "three"]
        single_sample_input_data = [2, 3, 1]
        batch_input_data = [[2, 3, 1, 3]]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "vocabulary": vocabulary,
            "invert": True,
            "output_mode": "int",
        }
        layer = layers.IndexLookup(**kwargs)
        output = layer(single_sample_input_data)
        self.assertEqual(
            [w.decode("utf-8") for w in output.numpy()], ["one", "two", "[OOV]"]
        )
        output = layer(batch_input_data)
        self.assertEqual(
            [w.decode("utf-8") for w in output.numpy()[0]],
            ["one", "two", "[OOV]", "two"],
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires string input dtype"
    )
    def test_saving(self):
        # Test with adapt()
        vocabulary = ["one", "two", "three"]
        adapt_data = ["one", "one", "one", "two", "two", "three"]
        batch_input_data = np.array([["one", "two", "four"]])
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "output_mode": "int",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        model = models.Sequential(
            [
                layers.Input(shape=(None,), dtype="string"),
                layer,
            ]
        )
        output_1 = model(batch_input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        model = saving_api.load_model(path)
        output_2 = model(batch_input_data)
        self.assertAllClose(output_1, output_2)

        # Test when vocabulary is provided
        kwargs["vocabulary"] = vocabulary
        layer = layers.IndexLookup(**kwargs)
        model = models.Sequential(
            [
                layers.Input(shape=(None,), dtype="string"),
                layer,
            ]
        )
        output_1 = model(batch_input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        model = saving_api.load_model(path)
        output_2 = model(batch_input_data)
        self.assertAllClose(output_1, output_2)

    def test_adapt_with_tf_data(self):
        # Case: adapt + list inputs
        adapt_data = tf_data.Dataset.from_tensor_slices(
            ["one", "one", "one", "two", "two", "three"]
        ).batch(2)
        input_data = ["one", "two", "four"]
        kwargs = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
        }
        layer = layers.IndexLookup(**kwargs)
        layer.adapt(adapt_data)
        self.assertEqual(
            layer.get_vocabulary(), ["", "[OOV]", "one", "two", "three"]
        )
        self.assertEqual(
            layer.get_vocabulary(include_special_tokens=False),
            ["one", "two", "three"],
        )
        output = layer(input_data)
        self.assertEqual(list(output), [2, 3, 1])
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_max_tokens_less_than_two(self):
        with self.assertRaisesRegex(
            ValueError,
            "If set, `max_tokens` must be greater than 1.",
        ):
            layers.IndexLookup(
                max_tokens=1,
                num_oov_indices=1,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="int64",
            )

    def test_max_tokens_none_with_pad_to_max_tokens(self):
        with self.assertRaisesRegex(
            ValueError,
            "If pad_to_max_tokens is True, must set `max_tokens`.",
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="int64",
                pad_to_max_tokens=True,
            )

    def test_negative_num_oov_indices(self):
        with self.assertRaisesRegex(
            ValueError,
            "`num_oov_indices` must be greater than or equal to 0.",
        ):
            layers.IndexLookup(
                max_tokens=10,
                num_oov_indices=-1,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="int64",
            )

    def test_invert_with_non_int_output_mode(self):
        with self.assertRaisesRegex(
            ValueError, r"`output_mode` must be `'int'` when `invert` is true."
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="string",
                invert=True,
                output_mode="one_hot",  # Invalid combination
            )

    def test_sparse_true_with_int_output_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            r"`sparse` may only be true if `output_mode` is `'one_hot'`",
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="string",
                sparse=True,
                output_mode="int",  # Invalid combination
            )

    def test_idf_weights_set_with_non_tfidf_output_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            r"`idf_weights` should only be set if `output_mode` is `'tf_idf'`",
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="string",
                idf_weights=[
                    0.5,
                    0.1,
                    0.3,
                ],  # Should not be set for non-TF-IDF modes
                output_mode="int",
            )

    def test_unrecognized_kwargs(self):
        with self.assertRaisesRegex(
            ValueError, "Unrecognized keyword argument"
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="string",
                output_mode="int",
                # This is an unrecognized argument
                extra_arg=True,
            )

    def test_non_tf_idf_with_idf_weights(self):
        with self.assertRaisesRegex(
            ValueError,
            "`idf_weights` should only be set if `output_mode` is",
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="string",
                output_mode="multi_hot",
                idf_weights=[
                    0.5,
                    0.1,
                    0.3,
                ],  # idf_weights not valid for multi_hot mode
            )

    def test_vocabulary_file_does_not_exist(self):
        with self.assertRaisesRegex(
            ValueError,
            "Vocabulary file path/to/missing_vocab.txt does not exist",
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="string",
                output_mode="int",
                # Nonexistent file path
                vocabulary="path/to/missing_vocab.txt",
            )

    def test_repeated_tokens_in_vocabulary(self):
        with self.assertRaisesRegex(
            ValueError, "The passed vocabulary has at least one repeated term."
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token=None,
                oov_token=None,
                vocabulary_dtype="string",
                vocabulary=["token", "token", "unique"],
            )

    def test_mask_token_in_wrong_position(self):
        with self.assertRaisesRegex(
            ValueError,
            "Found reserved mask token at unexpected location in `vocabulary`.",
        ):
            layers.IndexLookup(
                num_oov_indices=1,
                max_tokens=None,
                mask_token="mask",
                oov_token=None,
                vocabulary_dtype="string",
                vocabulary=[
                    "token",
                    "mask",
                    "unique",
                ],  # 'mask' should be at the start if included explicitly
            )

    def test_ensure_known_vocab_size_without_vocabulary(self):
        kwargs = {
            "num_oov_indices": 1,
            # Assume empty string or some default token is valid.
            "mask_token": "",
            # Assume [OOV] or some default token is valid.
            "oov_token": "[OOV]",
            "output_mode": "multi_hot",
            "pad_to_max_tokens": False,
            "vocabulary_dtype": "string",
            "max_tokens": None,
        }
        layer = layers.IndexLookup(**kwargs)

        # Try calling the layer without setting the vocabulary.
        with self.assertRaisesRegex(
            RuntimeError, "When using `output_mode=multi_hot` and"
        ):
            input_data = ["sample", "data"]
            layer(input_data)

    def test_save_and_load_assets_string_vocab(self):
        kwargs = {
            "max_tokens": 10,
            "num_oov_indices": 1,
            "mask_token": "<mask>",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
        }
        layer = layers.IndexLookup(**kwargs)

        vocabulary = ["apple", "banana", "cherry"]
        layer.set_vocabulary(vocabulary)

        vocab_before = layer.get_vocabulary(include_special_tokens=True)
        vocab_before_no_special = layer.get_vocabulary(
            include_special_tokens=False
        )

        sample_input = ["apple", "banana", "unknown"]
        output_before = layer(sample_input).numpy()

        tmpdir = self.get_temp_dir()

        layer.save_assets(tmpdir)

        layer2 = layers.IndexLookup(**kwargs)
        layer2.load_assets(tmpdir)

        vocab_after = layer2.get_vocabulary(include_special_tokens=True)
        vocab_after_no_special = layer2.get_vocabulary(
            include_special_tokens=False
        )

        self.assertEqual(vocab_before, vocab_after)
        self.assertEqual(vocab_before_no_special, vocab_after_no_special)

        output_after = layer2(sample_input).numpy()
        np.testing.assert_array_equal(output_before, output_after)

    def test_save_and_load_assets_with_multiple_oov_indices(self):
        kwargs = {
            "max_tokens": 10,
            "num_oov_indices": 2,
            "mask_token": "<mask>",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
        }
        layer = layers.IndexLookup(**kwargs)

        vocabulary = ["apple", "banana"]
        layer.set_vocabulary(vocabulary)

        vocab_before = layer.get_vocabulary(include_special_tokens=True)

        self.assertEqual(len(vocab_before), 5)
        self.assertEqual(vocab_before[0], "<mask>")
        self.assertEqual(vocab_before[1], "[OOV]")
        self.assertEqual(vocab_before[2], "[OOV]")

        tmpdir = self.get_temp_dir()

        layer.save_assets(tmpdir)

        layer2 = layers.IndexLookup(**kwargs)
        layer2.load_assets(tmpdir)

        vocab_after = layer2.get_vocabulary(include_special_tokens=True)
        self.assertEqual(vocab_before, vocab_after)

    def test_load_assets_handles_trailing_newlines(self):
        kwargs = {
            "max_tokens": 10,
            "num_oov_indices": 1,
            "mask_token": "<mask>",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
        }
        layer = layers.IndexLookup(**kwargs)

        vocabulary = ["apple", "banana", "cherry"]
        layer.set_vocabulary(vocabulary)
        vocab_expected = layer.get_vocabulary(include_special_tokens=True)

        tmpdir = self.get_temp_dir()

        vocab_file = os.path.join(tmpdir, "vocabulary.txt")
        with open(vocab_file, "w") as f:
            f.write("<mask>\n[OOV]\napple\nbanana\ncherry\n")

        layer2 = layers.IndexLookup(**kwargs)
        layer2.load_assets(tmpdir)

        vocab_loaded = layer2.get_vocabulary(include_special_tokens=True)
        self.assertEqual(vocab_expected, vocab_loaded)

    def test_oov_method_ignored_for_string_dtype(self):
        vocabulary = ["cat", "dog", "fish"]
        oov_data = ["aaa", "bbb", "ccc", "ddd", "eee", "fff"]
        kwargs = {
            "max_tokens": 10,
            "num_oov_indices": 4,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "vocabulary": vocabulary,
        }
        layer_floormod = layers.IndexLookup(oov_method="floormod", **kwargs)
        layer_farmhash = layers.IndexLookup(oov_method="farmhash", **kwargs)
        out_floormod = backend.convert_to_numpy(layer_floormod(oov_data))
        out_farmhash = backend.convert_to_numpy(layer_farmhash(oov_data))
        self.assertAllClose(out_floormod, out_farmhash)

    def test_salt_config_serialization(self):
        layer = layers.IndexLookup(
            max_tokens=10,
            num_oov_indices=2,
            mask_token=None,
            oov_token=-1,
            vocabulary_dtype="int64",
            vocabulary=[1, 2, 3],
            oov_method="farmhash",
            salt=[137, 42],
        )
        if backend.backend() != "torch":
            self.run_class_serialization_test(layer)

    def test_salt_valid_and_invalid_formats(self):
        base_kwargs = dict(
            max_tokens=10,
            num_oov_indices=2,
            mask_token=None,
            oov_token=-1,
            vocabulary_dtype="int64",
            oov_method="farmhash",
        )
        # valid formats
        layer = layers.IndexLookup(salt=[137, 42], **base_kwargs)
        self.assertEqual(layer.salt, [137, 42])
        layer = layers.IndexLookup(salt=99, **base_kwargs)
        self.assertEqual(layer.salt, [99, 99])
        # invalid formats
        with self.assertRaises(ValueError):
            layers.IndexLookup(salt="bad", **base_kwargs)
        with self.assertRaises(ValueError):
            layers.IndexLookup(salt=[1, 2, 3], **base_kwargs)

    def test_salt_requires_farmhash(self):
        with self.assertRaises(ValueError):
            layers.IndexLookup(
                max_tokens=10,
                num_oov_indices=2,
                mask_token=None,
                oov_token=-1,
                vocabulary_dtype="int64",
                oov_method="floormod",
                salt=[137, 42],
            )

    def test_salt_produces_different_output_than_farmhash(self):
        oov_integers = [100, 200, 300, 400, 500, 600]
        for dtype, vocab, oov_data in [
            ("int64", [1, 2, 3], oov_integers),
            ("string", ["a", "b", "c"], ["x", "y", "z", "w", "v", "u"]),
        ]:
            base_kwargs = dict(
                max_tokens=10,
                num_oov_indices=4,
                mask_token=None,
                oov_token=-1 if dtype == "int64" else "[OOV]",
                vocabulary_dtype=dtype,
                vocabulary=vocab,
                oov_method="farmhash",
            )
            layer_farmhash = layers.IndexLookup(**base_kwargs)
            layer_siphash = layers.IndexLookup(salt=[137, 42], **base_kwargs)
            out_farmhash = backend.convert_to_numpy(layer_farmhash(oov_data))
            out_siphash = backend.convert_to_numpy(layer_siphash(oov_data))
            self.assertFalse(
                np.array_equal(out_farmhash, out_siphash),
                msg=f"Expected different outputs for dtype={dtype}",
            )
