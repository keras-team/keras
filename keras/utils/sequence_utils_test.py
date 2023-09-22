from keras import testing
from keras.utils import sequence_utils


class PadSequencesTest(testing.TestCase):
    def test_pad_sequences(self):
        a = [[1], [1, 2], [1, 2, 3]]

        # test padding
        b = sequence_utils.pad_sequences(a, maxlen=3, padding="pre")
        self.assertAllClose(b, [[0, 0, 1], [0, 1, 2], [1, 2, 3]])
        b = sequence_utils.pad_sequences(a, maxlen=3, padding="post")
        self.assertAllClose(b, [[1, 0, 0], [1, 2, 0], [1, 2, 3]])

        # test truncating
        b = sequence_utils.pad_sequences(a, maxlen=2, truncating="pre")
        self.assertAllClose(b, [[0, 1], [1, 2], [2, 3]])
        b = sequence_utils.pad_sequences(a, maxlen=2, truncating="post")
        self.assertAllClose(b, [[0, 1], [1, 2], [1, 2]])

        # test value
        b = sequence_utils.pad_sequences(a, maxlen=3, value=1)
        self.assertAllClose(b, [[1, 1, 1], [1, 1, 2], [1, 2, 3]])

    def test_pad_sequences_str(self):
        a = [["1"], ["1", "2"], ["1", "2", "3"]]

        # test padding
        b = sequence_utils.pad_sequences(
            a, maxlen=3, padding="pre", value="pad", dtype=object
        )
        self.assertAllEqual(
            b, [["pad", "pad", "1"], ["pad", "1", "2"], ["1", "2", "3"]]
        )
        b = sequence_utils.pad_sequences(
            a, maxlen=3, padding="post", value="pad", dtype="<U3"
        )
        self.assertAllEqual(
            b, [["1", "pad", "pad"], ["1", "2", "pad"], ["1", "2", "3"]]
        )

        # test truncating
        b = sequence_utils.pad_sequences(
            a, maxlen=2, truncating="pre", value="pad", dtype=object
        )
        self.assertAllEqual(b, [["pad", "1"], ["1", "2"], ["2", "3"]])
        b = sequence_utils.pad_sequences(
            a, maxlen=2, truncating="post", value="pad", dtype="<U3"
        )
        self.assertAllEqual(b, [["pad", "1"], ["1", "2"], ["1", "2"]])

        with self.assertRaisesRegex(
            ValueError, "`dtype` int32 is not compatible with "
        ):
            sequence_utils.pad_sequences(
                a, maxlen=2, truncating="post", value="pad"
            )

    def test_pad_sequences_vector(self):
        a = [[[1, 1]], [[2, 1], [2, 2]], [[3, 1], [3, 2], [3, 3]]]

        # test padding
        b = sequence_utils.pad_sequences(a, maxlen=3, padding="pre")
        self.assertAllClose(
            b,
            [
                [[0, 0], [0, 0], [1, 1]],
                [[0, 0], [2, 1], [2, 2]],
                [[3, 1], [3, 2], [3, 3]],
            ],
        )
        b = sequence_utils.pad_sequences(a, maxlen=3, padding="post")
        self.assertAllClose(
            b,
            [
                [[1, 1], [0, 0], [0, 0]],
                [[2, 1], [2, 2], [0, 0]],
                [[3, 1], [3, 2], [3, 3]],
            ],
        )

        # test truncating
        b = sequence_utils.pad_sequences(a, maxlen=2, truncating="pre")
        self.assertAllClose(
            b, [[[0, 0], [1, 1]], [[2, 1], [2, 2]], [[3, 2], [3, 3]]]
        )

        b = sequence_utils.pad_sequences(a, maxlen=2, truncating="post")
        self.assertAllClose(
            b, [[[0, 0], [1, 1]], [[2, 1], [2, 2]], [[3, 1], [3, 2]]]
        )

        # test value
        b = sequence_utils.pad_sequences(a, maxlen=3, value=1)
        self.assertAllClose(
            b,
            [
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [2, 1], [2, 2]],
                [[3, 1], [3, 2], [3, 3]],
            ],
        )
